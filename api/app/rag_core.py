import os
import logging
import sys
import json
from typing import Generator, List

# --- LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document # <--- Added for search result wrapping

# --- Tools Imports ---
from langchain_community.tools import DuckDuckGoSearchRun # <--- Added Search Tool

# --- Reranking Imports ---
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank

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
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ConversationalRAG:
    """
    RAG System with all features:
    1. Milvus vector database for internal knowledge
    2. DuckDuckGo for internet search (handled via tools - not used in streaming)
    3. Query rewriting with historical context using summary technique
    4. FlashRank reranking for better relevance
    5. Streaming responses with token-by-token delivery
    6. Citation tracking with source metadata
    """
    
    def __init__(self):
        logging.info("=" * 80)
        logging.info("Initializing ConversationalRAG with Full Feature Set")
        logging.info("=" * 80)

        # 1. Initialize LLMs (Primary + Fallback)
        self.primary_llm = self._init_primary_llm()
        self.fallback_llm = self._init_fallback_llm()
        
        # 2. Initialize Search Tool
        self.search_tool = self._init_search_tool() # <--- New Initialization Method

        # 3. Initialize Milvus Vector Store & Base Retriever
        self.base_retriever = self._init_milvus_retriever()
        
        # 4. Initialize FlashRank Reranker (wraps base retriever)
        self.reranker = self._init_reranker()
        
        # 5. Initialize Query Rewriting Chain (with summary support)
        self.query_rewriter = self._init_query_rewriter()
        
        # 6. Initialize Summarization Chain (for progressive memory)
        self.summarizer = self._init_summarizer()
        
        logging.info("✓ ConversationalRAG initialized successfully with all features")
        logging.info("=" * 80)

    # ==================== LLM INITIALIZATION ====================
    
    def _init_primary_llm(self):
        """Initialize primary LLM (Gemini) with streaming support."""
        try:
            logging.info(f"Loading Primary LLM: {PRIMARY_LLM_MODEL}")
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            llm = ChatGoogleGenerativeAI(
                model=PRIMARY_LLM_MODEL, 
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                safety_settings=safety_settings,
                streaming=True  # Enable streaming
            )
            logging.info("✓ Primary LLM (Gemini) loaded successfully")
            return llm
        except Exception as e:
            logging.error(f"✗ Failed to load Gemini: {e}")
            return None

    def _init_fallback_llm(self):
        """Initialize fallback LLM (Ollama)."""
        try:
            logging.info(f"Loading Fallback LLM: {FALLBACK_LLM_MODEL}")
            llm = ChatOllama(
                model=FALLBACK_LLM_MODEL, 
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
                streaming=True
            )
            logging.info("✓ Fallback LLM (Ollama) loaded successfully")
            return llm
        except Exception as e:
            logging.error(f"✗ Failed to load Ollama: {e}")
            return None
        
    # ==================== TOOLS & RETRIEVAL ====================
    
    def _init_search_tool(self):
        """Initialize DuckDuckGo search tool."""
        try:
            tool = DuckDuckGoSearchRun()
            logging.info("✓ Internet Search Tool (DuckDuckGo) loaded")
            return tool
        except Exception as e:
            logging.warning(f"⚠ Search tool failed to load: {e}")
            return None
        
    def _init_milvus_retriever(self):
        """Initialize Milvus vector database retriever."""
        logging.info("Initializing Milvus Vector Store...")
        try:
            embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL_NAME, 
                base_url=OLLAMA_BASE_URL
            )
            URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
            
            vector_store = Milvus(
                embedding_function=embeddings,
                collection_name=MILVUS_COLLECTION_NAME,
                connection_args={"uri": URI},
                consistency_level="Strong",
                auto_id=True
            )
            
            # Retrieve top 10 for reranking (will be filtered down to top 4)
            retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={'k': 10}
            )
            
            logging.info(f"✓ Milvus retriever initialized (collection: {MILVUS_COLLECTION_NAME})")
            return retriever
            
        except Exception as e:
            logging.error(f"✗ Milvus initialization failed: {e}")
            return None

    def _init_reranker(self):
        """Initialize FlashRank reranker for improved relevance."""
        if not self.base_retriever:
            logging.warning("⚠ Base retriever not available, skipping reranker")
            return None
            
        try:
            logging.info("Loading FlashRank Reranker (ms-marco-MiniLM-L-12-v2)...")
            compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
            reranker = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=self.base_retriever
            )
            logging.info("✓ FlashRank reranker loaded successfully")
            return reranker
        except Exception as e:
            logging.warning(f"⚠ Reranker failed to load: {e}. Using base retriever.")
            return self.base_retriever

    # ==================== QUERY REWRITING ====================
    
    def _init_query_rewriter(self):
        """Initialize chain for query rewriting with historical context."""
        llm = self.primary_llm if self.primary_llm else self.fallback_llm
        if not llm:
            logging.warning("⚠ No LLM available for query rewriting")
            return None
            
        try:
            system_prompt = (
                "You are a query reformulation assistant. Given a conversation summary and recent chat history, "
                "rewrite the user's latest question to be self-contained and include relevant context.\n\n"
                "Guidelines:\n"
                "- Incorporate key entities and topics from the summary and history\n"
                "- Resolve pronouns and references (e.g., 'it', 'that', 'the company')\n"
                "- Keep the reformulated question concise and focused\n"
                "- If the question is already clear and standalone, return it as-is\n\n"
                "Conversation Summary: {summary}\n"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Reformulate this question: {input}"),
            ])
            
            chain = prompt | llm | StrOutputParser()
            logging.info("✓ Query rewriter initialized")
            return chain
            
        except Exception as e:
            logging.error(f"✗ Failed to initialize query rewriter: {e}")
            return None

    # ==================== SUMMARIZATION ====================
    
    def _init_summarizer(self):
        """Initialize chain for progressive conversation summarization."""
        llm = self.primary_llm if self.primary_llm else self.fallback_llm
        if not llm:
            logging.warning("⚠ No LLM available for summarization")
            return None
            
        try:
            prompt_template = (
                "Progressively summarize the conversation below, adding to the existing summary.\n\n"
                "Guidelines:\n"
                "- Keep the summary concise but informative\n"
                "- Focus on key topics, decisions, and context needed for future questions\n"
                "- Preserve important entities, numbers, and facts\n\n"
                "EXAMPLE:\n"
                "Current summary: The user asked about Q3 revenue. It was $2.5M, up 15% YoY.\n"
                "New conversation:\n"
                "Human: What about Q4 projections?\n"
                "AI: Q4 is projected at $3.1M, driven by holiday sales.\n"
                "New summary: The user asked about Q3 revenue ($2.5M, +15% YoY) and Q4 projections ($3.1M, driven by holiday sales).\n\n"
                "Current summary:\n{summary}\n\n"
                "New conversation:\n{new_lines}\n\n"
                "New summary:"
            )
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm | StrOutputParser()
            logging.info("✓ Summarizer initialized")
            return chain
            
        except Exception as e:
            logging.error(f"✗ Failed to initialize summarizer: {e}")
            return None

    # ==================== SUMMARIZATION PUBLIC METHOD ====================
    
    def summarize_messages(self, current_summary: str, new_messages: List[BaseMessage]) -> str:
        """
        Progressively summarize new messages into existing summary.
        
        Args:
            current_summary: Existing conversation summary
            new_messages: List of new HumanMessage/AIMessage objects
            
        Returns:
            Updated summary string
        """
        if not self.summarizer:
            logging.warning("⚠ Summarizer not available, returning current summary")
            return current_summary
            
        # Convert messages to readable format
        new_lines = ""
        for msg in new_messages:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            new_lines += f"{role}: {msg.content}\n"
            
        try:
            logging.info(f"Summarizing {len(new_messages)} messages...")
            new_summary = self.summarizer.invoke({
                "summary": current_summary,
                "new_lines": new_lines
            })
            logging.info(f"✓ Summary updated: {len(current_summary)} → {len(new_summary)} chars")
            return new_summary
            
        except Exception as e:
            logging.error(f"✗ Summarization failed: {e}")
            return current_summary

    # ==================== CORE STREAMING METHOD ====================
    
    def stream_answer(
        self, 
        user_input: str, 
        chat_history: List[BaseMessage], 
        current_summary: str = ""
    ) -> Generator[str, None, None]:
        """
        Main streaming method with all features:
        1. Query rewriting with summary + history
        2. Milvus retrieval with reranking
        3. Token-by-token streaming
        4. Citation extraction and streaming
        
        Yields strings in custom protocol:
        - "text:<content>" for response tokens
        - "sources:<json>" for citations at the end
        
        Args:
            user_input: User's question
            chat_history: List of recent messages (buffer)
            current_summary: Compressed older conversation summary
        """
        
        logging.info("=" * 80)
        logging.info(f"Processing query: {user_input[:100]}...")
        
        # ===== STEP 1: QUERY REWRITING =====
        refined_query = user_input
        
        if (chat_history or current_summary) and self.query_rewriter:
            try:
                logging.info("Rewriting query with historical context...")
                refined_query = self.query_rewriter.invoke({
                    "summary": current_summary or "No previous context.",
                    "chat_history": chat_history,
                    "input": user_input
                })
                logging.info(f"✓ Query rewritten: {refined_query[:100]}...")
            except Exception as e:
                logging.error(f"✗ Query rewriting failed: {e}, using original")

        # ===== STEP 2: RETRIEVAL & RERANKING =====
        docs = []
        if self.reranker:
            try:
                logging.info("Retrieving documents from Milvus...")
                docs = self.reranker.invoke(refined_query)
                # Keep top 4 after reranking
                docs = docs[:4]
                logging.info(f"✓ Retrieved and reranked to {len(docs)} documents")
            except Exception as e:
                logging.error(f"✗ Retrieval/reranking failed: {e}")

        # ===== STEP 3: INTERNET SEARCH (EXTERNAL) =====
        if self.search_tool:
            try:
                logging.info("Searching internet for external context...")
                # Using refined_query ensures the search is context-aware
                search_result = self.search_tool.invoke(refined_query)
                
                if search_result:
                    # Wrap result as a Document to treat it uniformly
                    search_doc = Document(
                        page_content=f"Internet Search Results (Current Info):\n{search_result}",
                        metadata={"source": "Internet Search (DuckDuckGo)", "page": "Web"}
                    )
                    docs.append(search_doc)
                    logging.info("✓ Internet search results added to context")
            except Exception as e:
                logging.error(f"✗ Internet search failed: {e}")

        # ===== STEP 4: EXTRACT CITATIONS =====
        sources = []
        seen_sources = set()
        context_text = ""
        
        for doc in docs:
            context_text += f"{doc.page_content}\n\n"
            
            # Extract metadata for citations
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            src_name = os.path.basename(src)
            identifier = f"{src_name}|{page}"
            
            if identifier not in seen_sources:
                sources.append({
                    "file": src_name,
                    "page": str(page)
                })
                seen_sources.add(identifier)
        
        logging.info(f"✓ Extracted {len(sources)} unique sources")

        # ===== STEP 5: PREPARE GENERATION PROMPT =====
        system_prompt = (
            "You are a helpful and knowledgeable financial AI assistant.\n\n"
            "Use the following retrieved context to answer the user's question accurately. "
            "If the context doesn't contain relevant information, clearly state that you don't have that information "
            "rather than making assumptions.\n\n"
            "Retrieved Context:\n" + (context_text or "No relevant context found.")
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # ===== STEP 6: STREAM RESPONSE =====
        llm = self.primary_llm if self.primary_llm else self.fallback_llm
        
        if not llm:
            yield "text:Error: No LLM available.\n"
            return
            
        chain = prompt | llm | StrOutputParser()
        
        try:
            logging.info("Streaming response...")
            token_count = 0
            
            # Stream tokens with custom protocol
            for chunk in chain.stream({
                "chat_history": chat_history,
                "input": refined_query
            }):
                yield f"text:{chunk}\n"
                token_count += 1
            
            logging.info(f"✓ Streamed {token_count} tokens")
            
            # ===== STEP 7: STREAM CITATIONS =====
            if sources:
                sources_json = json.dumps(sources)
                yield f"sources:{sources_json}\n"
                logging.info(f"✓ Streamed {len(sources)} citations")
                
        except Exception as e:
            logging.error(f"✗ Streaming failed: {e}")
            yield f"text:Error generating response: {str(e)}\n"
        
        logging.info("=" * 80)