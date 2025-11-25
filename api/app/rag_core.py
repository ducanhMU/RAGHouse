import os
import logging
import sys
import time

# --- LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_milvus import Milvus

# --- Tools & Agents Imports ---
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

# Robust Import for Retriever Tool (Handles version differences)
try:
    from langchain.tools.retriever import create_retriever_tool
except ImportError:
    from langchain.agents.agent_toolkits import create_retriever_tool

# Robust Import for Agent Creation
# If this fails, it means LangChain < 0.1.0 is installed. 
# The requirements.txt update should fix this, but this check helps debug.
try:
    from langchain.agents import create_react_agent, AgentExecutor
except ImportError:
    logging.critical("CRITICAL ERROR: Your LangChain version is too old. Please rebuild Docker with 'docker-compose up -d --build --no-cache'.")
    raise

# --- CONFIGURATION FROM ENV ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", "gemini-2.0-flash") 
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gpt-oss:20b") 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Database Config
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_demo")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")

# Logging Setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationalRAG:
    def __init__(self):
        logging.info("Initializing ConversationalRAG with Agentic capabilities...")

        # 1. Initialize LLMs (Primary & Fallback)
        self.primary_llm = self._init_primary_llm()
        self.fallback_llm = self._init_fallback_llm()
        
        # 2. Initialize Retriever (Connection to Milvus)
        self.retriever = self._init_retriever()
        
        # 3. Initialize Tools (Search + Knowledge Base)
        self.tools = self._init_tools()
        
        # 4. Initialize History Awareness Chains
        # These chains are responsible for rewriting questions based on chat history
        self.history_chain_primary = None
        self.history_chain_fallback = None

        if self.primary_llm:
            self.history_chain_primary = self._create_history_chain(self.primary_llm)
        if self.fallback_llm:
            self.history_chain_fallback = self._create_history_chain(self.fallback_llm)

        # 5. Initialize Agents (ReAct Executors)
        self.agent_executor_primary = None
        self.agent_executor_fallback = None

        try:
            if self.primary_llm and self.tools:
                self.agent_executor_primary = self._create_agent_executor(self.primary_llm)
                logging.info("Primary Agent (Gemini + Tools) initialized.")
        except Exception as e:
            logging.warning(f"Cannot initialize Primary Agent: {e}")

        try:
            if self.fallback_llm and self.tools:
                self.agent_executor_fallback = self._create_agent_executor(self.fallback_llm)
                logging.info("Fallback Agent (Ollama + Tools) initialized.")
        except Exception as e:
            logging.warning(f"Cannot initialize Fallback Agent: {e}")
            
        if not self.agent_executor_primary and not self.agent_executor_fallback:
            logging.error("CRITICAL: No Agents available.")
        else:
            logging.info("ConversationalRAG is ready.")

    def _init_primary_llm(self):
        """Initializes Google Gemini with safety filters disabled."""
        try:
            logging.info(f"Loading Primary LLM: {PRIMARY_LLM_MODEL}")
            # Disable safety filters to prevent empty responses on benign topics
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            return ChatGoogleGenerativeAI(
                model=PRIMARY_LLM_MODEL, 
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3, # Low temp for factual accuracy
                safety_settings=safety_settings
            )
        except Exception as e:
            logging.error(f"Failed to load Gemini: {e}")
            return None

    def _init_fallback_llm(self):
        """Initializes Local Ollama Model."""
        try:
            logging.info(f"Loading Fallback LLM: {FALLBACK_LLM_MODEL}")
            return ChatOllama(
                model=FALLBACK_LLM_MODEL, 
                base_url=OLLAMA_BASE_URL,
                temperature=0.1 # Very low temp for stable tool calling
            )
        except Exception as e:
            logging.error(f"Failed to load Ollama: {e}")
            return None

    def _init_retriever(self):
        """Initializes connection to Milvus Vector DB."""
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
                    auto_id=True
                )
                
                # Force check connection
                # vector_store.col.num_entities 
                logging.info("Connected to Milvus Collection.")
                
                return vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 3}
                )
                
            except Exception as e:
                logging.warning(f"Milvus connection failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logging.error("Failed to connect to Milvus after retries")
                    return None

    def _init_tools(self):
        """Defines the tools available to the Agent."""
        tools = []
        
        # 1. Internet Search Tool (DuckDuckGo)
        try:
            search_tool = DuckDuckGoSearchRun(
                name="internet_search",
                description="Useful for when you need to answer questions about current events, news, realtime data, or topics NOT found in the internal database."
            )
            tools.append(search_tool)
            logging.info("Internet Search tool loaded.")
        except Exception as e:
            logging.error(f"Failed to load Search Tool: {e}")

        # 2. Internal Knowledge Tool (Milvus)
        if self.retriever:
            retriever_tool = create_retriever_tool(
                self.retriever,
                "internal_knowledge_base",
                "Useful for searching internal uploaded documents, company reports, and specific private data stored in Milvus."
            )
            tools.append(retriever_tool)
            logging.info("Milvus Retriever tool loaded.")
            
        return tools

    def _create_history_chain(self, llm):
        """Creates a chain to rewrite user questions based on history."""
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. \n"
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        return contextualize_q_prompt | llm | StrOutputParser()

    def _create_agent_executor(self, llm):
        """Creates a ReAct Agent Executor."""
        # Pull standard prompt from LangChain Hub
        prompt = hub.pull("hwchase17/react")
        
        agent = create_react_agent(llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=5
        )

    def invoke(self, user_input, chat_history):
        """
        Main entry point.
        1. Contextualize (Rewrite) the question using history.
        2. Pass the rewritten question to the Agent.
        """
        logging.info(f"Original Input: {user_input}")
        
        refined_query = user_input

        # --- STEP 1: QUERY REFINEMENT (HISTORY AWARENESS) ---
        if chat_history and len(chat_history) > 0:
            logging.info("History detected. Reformulating question...")
            try:
                # Prefer Primary LLM for better reformulation logic
                if self.history_chain_primary:
                    refined_query = self.history_chain_primary.invoke({
                        "chat_history": chat_history,
                        "input": user_input
                    })
                elif self.history_chain_fallback:
                    refined_query = self.history_chain_fallback.invoke({
                        "chat_history": chat_history,
                        "input": user_input
                    })
                logging.info(f"Refined Query: {refined_query}")
            except Exception as e:
                logging.error(f"Failed to contextualize query: {e}. Using original input.")
        
        # --- STEP 2: AGENT EXECUTION ---
        chain_input = {"input": refined_query}
        
        # Attempt 1: Primary Agent (Gemini)
        if self.agent_executor_primary:
            try:
                logging.info("Invoking Primary Agent (Gemini)...")
                result = self.agent_executor_primary.invoke(chain_input)
                return {"answer": result.get("output", "No output generated")}
            except Exception as e:
                logging.error(f"Gemini Agent Error: {e}. Switching to Fallback...")
        
        # Attempt 2: Fallback Agent (Ollama)
        if self.agent_executor_fallback:
            try:
                logging.info("Invoking Fallback Agent (Ollama)...")
                result = self.agent_executor_fallback.invoke(chain_input)
                return {"answer": result.get("output", "No output generated")}
            except Exception as e:
                logging.error(f"Ollama Agent Error: {e}")
                return {"answer": "Error: Both systems failed to process the request."}
        
        return {"answer": "System not initialized. Check logs."}