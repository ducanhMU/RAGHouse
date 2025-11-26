# file: api/app/rag_core.py

import os
import logging
from typing import AsyncGenerator, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Internal Imports
from app.database import ChatEvent, SessionLocal
from app import ingest  # To reuse get_vector_store

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Logging
logger = logging.getLogger(__name__)

class HybridRAG:
    """
    Core RAG Engine for Version 1.
    Features:
    - Hybrid Inference (Gemini Primary -> Ollama Fallback)
    - Hierarchical Memory Assembly (The 3-5 Rule)
    - Milvus Vector Retrieval
    """
    
    def __init__(self):
        # 1. Primary Model: Gemini 1.5 Flash (Fast, Cheap, Long Context)
        self.primary_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True, # Often needed for Gemini compatibility
            stream=True
        )
        
        # 2. Fallback Model: Ollama (Local Llama 3 or Mistral)
        self.fallback_llm = ChatOllama(
            model="llama3.2:3b", # Ensure you pulled this model in docker-compose
            base_url=OLLAMA_BASE_URL,
            temperature=0.3
        )
        
        logger.info("HybridRAG Engine Initialized (Gemini + Ollama)")

    def build_hierarchical_context(self, db: Session, session_id: str) -> str:
        """
        Constructs the 'Memory Context' based on the 3-5 Rule.
        Structure: [Checkpoint] -> [Summaries] -> [Recent Raw Messages]
        """
        # 

        # A. Get latest Checkpoint (Global Context)
        # We look for the most recent event with type 'CHECKPOINT_5'
        checkpoint = db.query(ChatEvent).filter(
            ChatEvent.session_id == session_id,
            ChatEvent.event_type == 'CHECKPOINT_5'
        ).order_by(desc(ChatEvent.sequence_num)).first()
        
        checkpoint_txt = checkpoint.content if checkpoint else "No long-term history yet."
        last_seq = checkpoint.sequence_num if checkpoint else 0

        # B. Get Summaries generated *after* the last Checkpoint
        summaries = db.query(ChatEvent).filter(
            ChatEvent.session_id == session_id,
            ChatEvent.event_type == 'SUMMARY_3',
            ChatEvent.sequence_num > last_seq
        ).order_by(ChatEvent.sequence_num).all()
        
        summary_txt = "\n".join([f"- {s.content}" for s in summaries]) if summaries else "No mid-term summaries."
        
        # Update last_seq to the end of summaries (if any)
        if summaries:
            last_seq = summaries[-1].sequence_num

        # C. Get Recent Raw Messages (since last Summary or Checkpoint)
        # These are the immediate conversation turns that haven't been summarized yet
        raw_msgs = db.query(ChatEvent).filter(
            ChatEvent.session_id == session_id,
            ChatEvent.event_type == 'NORMAL',
            ChatEvent.sequence_num > last_seq
        ).order_by(ChatEvent.sequence_num).all()
        
        recent_txt = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in raw_msgs])

        # D. Assemble Final String
        return f"""
        [LONG-TERM MEMORY (CHECKPOINT)]:
        {checkpoint_txt}

        [MID-TERM CONTEXT (SUMMARIES)]:
        {summary_txt}

        [RECENT INTERACTION (RAW)]:
        {recent_txt}
        """

    def get_retrieval_context(self, query: str, k: int = 5) -> str:
        """
        Fetches relevant documents from Milvus.
        """
        try:
            vector_db = ingest.get_vector_store()
            docs = vector_db.similarity_search(query, k=k)
            
            if not docs:
                return "No relevant documents found in knowledge base."
            
            # Format docs with Source Metadata
            context_parts = []
            for d in docs:
                source = d.metadata.get('filename', 'Unknown Source')
                context_parts.append(f"[Source: {source}]: {d.page_content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Milvus Retrieval Failed: {e}")
            return "Error retrieving documents from knowledge base."

    async def generate_stream(self, memory_ctx: str, rag_ctx: str, user_query: str) -> AsyncGenerator[str, None]:
        """
        The Main Generator: Combines Context -> Prompts LLM -> Streams Response.
        Implements the Failover Logic.
        """
        
        # 1. Construct System Prompt
        system_instruction = f"""You are an intelligent AI assistant. 
        Answer the user's question using the provided Knowledge Base and Conversation History.
        
        GUIDELINES:
        1. Priority: Use [KNOWLEDGE BASE] facts first.
        2. Fallback: If facts are missing, use your general knowledge but state "Based on my general knowledge...".
        3. Memory: Use [HISTORY] to understand context (e.g., if user says "it", look at history).
        4. Citations: Mention the source filename if you use data from [KNOWLEDGE BASE].

        [KNOWLEDGE BASE]:
        {rag_ctx}

        [HISTORY & MEMORY]:
        {memory_ctx}
        """

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_query)
        ]

        # 2. Hybrid Inference Execution
        try:
            # Attempt Primary (Gemini)
            async for chunk in self.primary_llm.astream(messages):
                yield chunk.content
                
        except Exception as e:
            logger.error(f"Primary LLM (Gemini) failed: {e}. Switching to Fallback (Ollama).")
            try:
                # Attempt Fallback (Ollama)
                async for chunk in self.fallback_llm.astream(messages):
                    yield chunk.content
            except Exception as e2:
                logger.critical(f"Both LLMs failed. Ollama Error: {e2}")
                yield "I apologize, but I am currently unable to process your request due to system overload."

    def trigger_memory_consolidation(self, session_id: str):
        """
        Background Task: Implements the 'Write' side of the 3-5 Rule.
        - Every 3 turns (6 messages) -> Create Summary.
        - Every 5 summaries -> Create Checkpoint.
        """
        db = SessionLocal()
        try:
            # --- CHECK 1: Short-term Summary (Needs 3 completed turns = 6 normal messages) ---
            
            # Find the sequence number of the last Summary or Checkpoint
            last_summary_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type.in_(['SUMMARY_3', 'CHECKPOINT_5'])
            ).scalar() or 0

            # Count NORMAL messages after that point
            new_msgs = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'NORMAL',
                ChatEvent.sequence_num > last_summary_seq
            ).order_by(ChatEvent.sequence_num).all()

            # Rule: 3 Turns = 3 User + 3 Assistant = 6 Messages
            if len(new_msgs) >= 6:
                logger.info(f"Session {session_id}: Triggering Summary_3 generation.")
                self._create_summary(db, session_id, new_msgs, "SUMMARY_3")

            # --- CHECK 2: Long-term Checkpoint (Needs 5 Summaries) ---
            
            # Find the sequence number of the last Checkpoint
            last_checkpoint_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'CHECKPOINT_5'
            ).scalar() or 0

            # Count SUMMARY_3 events after that point
            new_summaries = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'SUMMARY_3',
                ChatEvent.sequence_num > last_checkpoint_seq
            ).order_by(ChatEvent.sequence_num).all()

            # Rule: 5 Summaries
            if len(new_summaries) >= 5:
                logger.info(f"Session {session_id}: Triggering Checkpoint_5 generation.")
                self._create_summary(db, session_id, new_summaries, "CHECKPOINT_5")
                
        except Exception as e:
            logger.error(f"Memory Consolidation Failed: {e}")
        finally:
            db.close()

    def _create_summary(self, db: Session, session_id: str, events: list, target_type: str):
        """
        Helper to call LLM to summarize a list of events and insert into DB.
        """
        # 1. Prepare Text
        text_chunk = "\n".join([f"{e.role}: {e.content}" for e in events])
        
        prompt = f"""
        Please summarize the following conversation segment concisely. 
        Focus on facts, user intent, and key decisions. 
        Keep it under 100 words.
        
        TEXT:
        {text_chunk}
        """

        # 2. Call LLM (Non-streaming)
        summary_content = "Summary generation failed."
        try:
            summary_content = self.primary_llm.invoke(prompt).content
        except:
            try:
                summary_content = self.fallback_llm.invoke(prompt).content
            except Exception as e:
                logger.error(f"Summarization failed completely: {e}")
                return # Skip insertion if failed

        # 3. Insert into DB (Hidden Event)
        # Get next sequence ID
        next_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
            ChatEvent.session_id == session_id
        ).scalar() + 1
        
        new_event = ChatEvent(
            session_id=session_id,
            sequence_num=next_seq,
            role='system',
            content=summary_content,
            event_type=target_type,
            visibility='HIDDEN' # User doesn't see this in UI
        )
        db.add(new_event)
        db.commit()
        logger.info(f"Session {session_id}: Created {target_type} successfully.")