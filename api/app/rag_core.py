import os
import logging
import sys
import time

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION FROM ENV ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", "gemini-2.0-flash")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gemma2:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_demo")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationalRAG:
    def __init__(self):
        logging.info("Initializing ConversationalRAG...")

        self.primary_llm = self._init_primary_llm()
        self.fallback_llm = self._init_fallback_llm()
        self.retriever = self._init_retriever()
        
        self.rag_chain_primary = None
        self.rag_chain_fallback = None

        try:
            if self.primary_llm and self.retriever:
                self.rag_chain_primary = self._create_full_rag_chain(self.primary_llm)
        except Exception as e:
            logging.warning(f"Cannot initialize Primary RAG chain: {e}")

        try:
            if self.fallback_llm and self.retriever:
                self.rag_chain_fallback = self._create_full_rag_chain(self.fallback_llm)
        except Exception as e:
            logging.warning(f"Cannot initialize Fallback RAG chain: {e}")
            
        if not self.rag_chain_primary and not self.rag_chain_fallback:
            logging.error("CRITICAL: No RAG chains available.")
        else:
            logging.info("ConversationalRAG is ready.")

    def _init_primary_llm(self):
        try:
            logging.info(f"Loading Primary LLM: {PRIMARY_LLM_MODEL}")
            return ChatGoogleGenerativeAI(
                model=PRIMARY_LLM_MODEL, 
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3
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
                temperature=0.3
            )
        except Exception as e:
            logging.error(f"Failed to load Ollama: {e}")
            return None

    def _init_retriever(self):
        logging.info("Initializing Retriever connection...")
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            base_url=OLLAMA_BASE_URL
        )
      
        max_retries = 5
        retry_delay = 3
        URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        
        for attempt in range(max_retries):
            try:
                logging.info(f"[Attempt {attempt + 1}] Connecting to Milvus at {URI}")
                
                vector_store = Milvus(
                    embedding_function=embeddings,
                    collection_name=MILVUS_COLLECTION_NAME,
                    connection_args={"uri": URI},
                    consistency_level="Strong",
                    index_params={"metric_type": "L2"},
                    auto_id=True
                )
                
                logging.info("✓ Connected to Milvus Collection.")
                
                return vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        'k': 3, 
                        'param': {'ef': 64} 
                    }
                )
                
            except Exception as e:
                logging.warning(f"Milvus connection failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logging.error("Failed to connect to Milvus after retries")
                    return None

    def _create_full_rag_chain(self, llm):
        # 1. Contextualize History
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm, self.retriever, contextualize_q_prompt
        )

        # 2. Answer Question
        qa_system_prompt = (
            "You are a helpful AI assistant. "
            "Use the following pieces of retrieved context to answer the question. "
            "\n\n"
            "{context}"
            "\n\n"
            "INSTRUCTIONS:"
            "\n1. If the context provides the answer, use it to answer in detail."
            "\n2. If the context is missing, empty, or not relevant, "
            "you MUST answer using your own internal knowledge."
            "\n3. CRITICAL: If you answer without using the provided context, "
            "start your response with: 'I don't have context, please be careful or find other sources'."
            "\n4. Provide a comprehensive answer. Do not limit length."
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def invoke(self, user_input, chat_history):
        logging.info(f"Processing question: {user_input}")
        chain_input = {"input": user_input, "chat_history": chat_history}
        
        if self.rag_chain_primary:
            try:
                logging.info("Invoking Primary Chain (Gemini)...")
                return self.rag_chain_primary.invoke(chain_input)
            except Exception as e:
                logging.error(f"Gemini Error: {e}. Switching to Fallback...")
        
        if self.rag_chain_fallback:
            try:
                logging.info("Invoking Fallback Chain (Ollama)...")
                return self.rag_chain_fallback.invoke(chain_input)
            except Exception as e:
                logging.error(f"Ollama Error: {e}")
                return {"answer": "Error: Both systems failed."}
        
        return {"answer": "System not initialized."}