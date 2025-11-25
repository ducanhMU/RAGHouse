import os
import logging
import sys
import time
from typing import List

# --- LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_milvus import Milvus

# --- Tools & Agents ---
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

# Robust Imports
try:
    from langchain.tools.retriever import create_retriever_tool
except ImportError:
    from langchain.agents.agent_toolkits import create_retriever_tool

try:
    from langchain.agents import create_react_agent, AgentExecutor
except ImportError:
    logging.critical("CRITICAL ERROR: LangChain version too old.")
    raise

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", "gemini-2.0-flash") 
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gpt-oss:20b") 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Database Config
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_demo")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationalRAG:
    def __init__(self):
        logging.info("Initializing ConversationalRAG with Summary Memory...")

        self.primary_llm = self._init_primary_llm()
        self.fallback_llm = self._init_fallback_llm()
        self.retriever = self._init_retriever()
        self.tools = self._init_tools()
        
        # History Awareness Chains
        self.history_chain_primary = None
        self.history_chain_fallback = None
        if self.primary_llm:
            self.history_chain_primary = self._create_history_chain(self.primary_llm)
        if self.fallback_llm:
            self.history_chain_fallback = self._create_history_chain(self.fallback_llm)

        # Summarization Chains (New)
        self.summary_chain_primary = None
        self.summary_chain_fallback = None
        if self.primary_llm:
            self.summary_chain_primary = self._create_summary_chain(self.primary_llm)
        if self.fallback_llm:
            self.summary_chain_fallback = self._create_summary_chain(self.fallback_llm)

        # Agents
        self.agent_executor_primary = None
        self.agent_executor_fallback = None

        if self.primary_llm and self.tools:
            self.agent_executor_primary = self._create_agent_executor(self.primary_llm)

        if self.fallback_llm and self.tools:
            self.agent_executor_fallback = self._create_agent_executor(self.fallback_llm)
            
        logging.info("ConversationalRAG Ready.")

    def _init_primary_llm(self):
        try:
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
                safety_settings=safety_settings
            )
        except Exception as e:
            logging.error(f"Failed to load Gemini: {e}")
            return None

    def _init_fallback_llm(self):
        try:
            return ChatOllama(
                model=FALLBACK_LLM_MODEL, 
                base_url=OLLAMA_BASE_URL,
                temperature=0.1
            )
        except Exception as e:
            logging.error(f"Failed to load Ollama: {e}")
            return None

    def _init_retriever(self):
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
            return vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 3})
        except Exception:
            return None

    def _init_tools(self):
        tools = []
        try:
            tools.append(DuckDuckGoSearchRun(name="internet_search", description="Search for realtime info."))
        except Exception:
            pass
        if self.retriever:
            tools.append(create_retriever_tool(self.retriever, "internal_knowledge", "Search internal documents."))
        return tools

    def _create_history_chain(self, llm):
        # Updated prompt to include the Summary
        system_prompt = (
            "Given a conversation summary, chat history, and the latest user question, "
            "formulate a standalone question. \n"
            "Summary of older conversation: {summary}\n"
            "Recent Chat History: {chat_history}\n"
            "Do NOT answer, just rewrite the question."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        return prompt | llm | StrOutputParser()

    def _create_summary_chain(self, llm):
        """Creates a chain to summarize conversation."""
        # Standard LangChain summary prompt logic
        prompt_template = (
            "Progressively summarize the lines of conversation provided, adding to the previous summary returning a new summary.\n\n"
            "EXAMPLE\n"
            "Current summary: The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.\n"
            "New lines of conversation:\n"
            "Human: Why do you think artificial intelligence is a force for good?\n"
            "AI: Because artificial intelligence will help humans reach their full potential.\n"
            "New summary: The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.\n"
            "END OF EXAMPLE\n\n"
            "Current summary:\n{summary}\n\n"
            "New lines of conversation:\n{new_lines}\n\n"
            "New summary:"
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | llm | StrOutputParser()

    def _create_agent_executor(self, llm):
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True, max_iterations=5)

    def summarize_messages(self, current_summary: str, new_messages: List[BaseMessage]) -> str:
        """
        Calls the LLM to merge new messages into the existing summary.
        """
        # Convert messages to string format
        new_lines = ""
        for msg in new_messages:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            new_lines += f"{role}: {msg.content}\n"
            
        chain = self.summary_chain_primary if self.summary_chain_primary else self.summary_chain_fallback
        if not chain:
            return current_summary # Cannot summarize
        
        try:
            logging.info("Summarizing conversation...")
            return chain.invoke({"summary": current_summary, "new_lines": new_lines})
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return current_summary

    def invoke(self, user_input, chat_history, current_summary=""):
        """
        Main entry point. Now accepts current_summary.
        """
        refined_query = user_input

        # 1. REFINEMENT (Using Summary + Buffer)
        if (chat_history and len(chat_history) > 0) or current_summary:
            try:
                chain = self.history_chain_primary if self.history_chain_primary else self.history_chain_fallback
                if chain:
                    refined_query = chain.invoke({
                        "summary": current_summary,
                        "chat_history": chat_history,
                        "input": user_input
                    })
                    logging.info(f"Refined Query: {refined_query}")
            except Exception as e:
                logging.error(f"Contextualization failed: {e}")
        
        # 2. AGENT EXECUTION
        chain_input = {"input": refined_query}
        
        if self.agent_executor_primary:
            try:
                res = self.agent_executor_primary.invoke(chain_input)
                return {"answer": res.get("output", "No output")}
            except Exception:
                logging.warning("Gemini Agent failed, switching...")
        
        if self.agent_executor_fallback:
            try:
                res = self.agent_executor_fallback.invoke(chain_input)
                return {"answer": res.get("output", "No output")}
            except Exception:
                pass
        
        return {"answer": "System Error."}