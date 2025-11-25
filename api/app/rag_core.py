import os
import logging
import sys
import json
from typing import Generator

# --- LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_milvus import Milvus

# --- Reranking Imports ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# --- CONFIGURATION FROM ENV ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", "gemini-2.0-flash") 
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gpt-oss:20b") 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_demo")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")

# Logging Setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationalRAG:
    def __init__(self):
        logging.info("Initializing RAG with Streaming, Citations, and Reranking...")

        # 1. Initialize LLMs
        self.primary_llm = self._init_primary_llm()
        self.fallback_llm = self._init_fallback_llm()
        
        # 2. Initialize Base Retriever (Milvus)
        self.base_retriever = self._init_retriever()
        
        # 3. Initialize Reranker (FlashRank)
        self.reranker = None
        if self.base_retriever:
            try:
                logging.info("Loading FlashRank Reranker...")
                # FlashRank reranks documents locally using a lightweight model
                compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2") 
                self.reranker = ContextualCompressionRetriever(
                    base_compressor=compressor, 
                    base_retriever=self.base_retriever
                )
                logging.info("Reranker loaded successfully.")
            except Exception as e:
                logging.warning(f"Failed to load Reranker: {e}. Falling back to base retriever.")
                self.reranker = self.base_retriever

        # 4. Initialize History Chain (for query rewriting)
        self.history_chain = None
        if self.primary_llm:
            self.history_chain = self._create_history_chain(self.primary_llm)

    def _init_primary_llm(self):
        try:
            logging.info(f"Loading Primary LLM: {PRIMARY_LLM_MODEL}")
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            return ChatGoogleGenerativeAI(
                model=PRIMARY_LLM_MODEL, 
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                safety_settings=safety_settings,
                streaming=True # Enable Streaming capability
            )
        except Exception as e:
            logging.error(f"Failed to load Gemini: {e}")
            return None

    def _init_fallback_llm(self):
        try:
            logging.info(f"Loading Fallback LLM: {FALLBACK_LLM_MODEL}")
            return ChatOllama(
                model=FALLBACK_LLM_MODEL, 
                base_url=OLLAMA_BASE_URL,
                temperature=0.1
            )
        except Exception as e:
            logging.error(f"Failed to load Ollama: {e}")
            return None

    def _init_retriever(self):
        logging.info("Initializing Milvus Retriever...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
        URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        try:
            vector_store = Milvus(
                embedding_function=embeddings,
                collection_name=MILVUS_COLLECTION_NAME,
                connection_args={"uri": URI},
                consistency_level="Strong",
                auto_id=True
            )
            # Fetch top 10 results initially, let Reranker filter them down
            return vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 10})
        except Exception as e:
            logging.error(f"Milvus init failed: {e}")
            return None

    def _create_history_chain(self, llm):
        """Creates a chain to rewrite user questions based on history."""
        system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        return prompt | llm | StrOutputParser()

    def stream_answer(self, user_input, chat_history) -> Generator[str, None, None]:
        """
        Generator function that:
        1. Refines query using history.
        2. Retrieves and Reranks documents.
        3. Streams text chunks.
        4. Streams sources metadata at the end.
        """
        refined_query = user_input

        # 1. Refine Query
        if chat_history:
            try:
                if self.history_chain:
                    refined_query = self.history_chain.invoke({"chat_history": chat_history, "input": user_input})
                    logging.info(f"Refined Query: {refined_query}")
            except Exception as e:
                logging.error(f"History chain failed: {e}")

        # 2. Retrieve & Rerank Documents
        docs = []
        if self.reranker:
            try:
                # Retrieve top K and Rerank
                docs = self.reranker.invoke(refined_query)
                # Keep top 4 most relevant after reranking
                docs = docs[:4]
            except Exception as e:
                logging.error(f"Retrieval/Reranking failed: {e}")

        # 3. Extract Citations (Sources)
        # We extract unique sources from document metadata
        sources = []
        seen_sources = set()
        
        context_text = ""
        for d in docs:
            context_text += f"{d.page_content}\n\n"
            
            # Metadata handling (assumes ingestion puts 'source' and 'page')
            src = d.metadata.get("source", "Unknown File")
            page = d.metadata.get("page", "N/A")
            
            # Clean up filename (remove path)
            src_name = os.path.basename(src)
            identifier = f"{src_name}-{page}"
            
            if identifier not in seen_sources:
                sources.append({"file": src_name, "page": page})
                seen_sources.add(identifier)

        # 4. Prepare Generation Prompt
        system_prompt = (
            "You are a helpful financial AI assistant. Use the following retrieved context to answer the user's question.\n"
            "If the context is not relevant, state that you don't have the information.\n\n"
            "Context:\n" + context_text
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # 5. Stream Response
        llm = self.primary_llm if self.primary_llm else self.fallback_llm
        chain = prompt | llm | StrOutputParser()
        
        try:
            # Stream tokens
            for chunk in chain.stream({"chat_history": chat_history, "input": refined_query}):
                # Custom protocol: 'text:' prefix
                yield f"text:{chunk}\n"
            
            # Stream sources at the very end
            if sources:
                sources_json = json.dumps(sources)
                # Custom protocol: 'sources:' prefix
                yield f"sources:{sources_json}\n"
                
        except Exception as e:
            yield f"text:Error generating response: {e}\n"