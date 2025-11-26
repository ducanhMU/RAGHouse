# file: api/app/rag_core.py

import os
import logging
from typing import AsyncGenerator, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Internal Imports
from app.database import ChatEvent, SessionLocal
from app import ingest

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Logging
logger = logging.getLogger(__name__)

class HybridRAG:
    """
    Core RAG Engine for Version 1.
    
    Features:
    - Hybrid Inference (Gemini Primary -> Ollama Fallback)
    - Hierarchical Memory Assembly (The 3-5 Rule)
    - Milvus Vector Retrieval
    - Streaming Response Generation
    """
    
    def __init__(self):
        """Initialize the hybrid LLM setup."""
        try:
            # 1. Primary Model: Gemini 1.5 Flash (Fast, Long Context)
            if not GOOGLE_API_KEY:
                logger.warning("GOOGLE_API_KEY not set. Gemini will not be available.")
                self.primary_llm = None
            else:
                self.primary_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-latest",
                    temperature=0.3,
                    google_api_key=GOOGLE_API_KEY,
                    convert_system_message_to_human=True,
                    streaming=True,
                    max_output_tokens=2048
                )
                logger.info("Gemini 1.5 Flash initialized successfully")
            
            # 2. Fallback Model: Ollama (Local)
            self.fallback_llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.3,
                num_predict=2048
            )
            logger.info(f"Ollama ({OLLAMA_MODEL}) initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLMs: {e}")
            raise

    def build_hierarchical_context(self, db: Session, session_id: str) -> str:
        """
        Constructs the 'Memory Context' based on the 3-5 Rule.
        
        Structure:
        1. [CHECKPOINT] - Long-term conversation summary
        2. [SUMMARIES] - Mid-term summaries of conversation segments
        3. [RECENT] - Raw recent messages
        
        Args:
            db: Database session
            session_id: Current chat session ID
            
        Returns:
            Formatted context string
        """
        try:
            # A. Get latest Checkpoint (Global Context)
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
            
            # Update last_seq to the end of summaries
            if summaries:
                last_seq = summaries[-1].sequence_num

            # C. Get Recent Raw Messages (since last Summary or Checkpoint)
            # Limit to last 10 messages to avoid context overflow
            raw_msgs = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'NORMAL',
                ChatEvent.sequence_num > last_seq
            ).order_by(ChatEvent.sequence_num).limit(10).all()
            
            recent_txt = "\n".join([
                f"{msg.role.upper()}: {msg.content}" 
                for msg in raw_msgs
            ]) if raw_msgs else "No recent messages."

            # D. Assemble Final Context
            context = f"""[LONG-TERM MEMORY (CHECKPOINT)]:
{checkpoint_txt}

[MID-TERM CONTEXT (SUMMARIES)]:
{summary_txt}

[RECENT INTERACTION (RAW)]:
{recent_txt}"""
            
            logger.debug(f"Built hierarchical context for session {session_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to build hierarchical context: {e}")
            return "Error loading conversation history."

    def get_retrieval_context(self, query: str, k: int = 5) -> str:
        """
        Fetches relevant documents from Milvus vector store.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string with sources
        """
        try:
            vector_db = ingest.get_vector_store()
            docs = vector_db.similarity_search(query, k=k)
            
            if not docs:
                return "No relevant documents found in knowledge base."
            
            # Format with source metadata
            context_parts = []
            seen_sources = set()
            
            for d in docs:
                source = d.metadata.get('filename', 'Unknown Source')
                # Avoid duplicate sources
                if source not in seen_sources:
                    context_parts.append(
                        f"[Source: {source}]\n{d.page_content}"
                    )
                    seen_sources.add(source)
            
            logger.debug(f"Retrieved {len(context_parts)} unique sources for query")
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Milvus retrieval failed: {e}")
            return "Error retrieving documents from knowledge base."

    async def generate_stream(
        self, 
        memory_ctx: str, 
        rag_ctx: str, 
        user_query: str
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Main streaming generator with hybrid LLM failover.
        
        Args:
            memory_ctx: Hierarchical conversation context
            rag_ctx: Retrieved document context
            user_query: Current user question
            
        Yields:
            Tuples of (chunk, model_name)
        """
        # 1. Construct System Prompt
        system_instruction = f"""You are an intelligent AI assistant with access to a knowledge base and conversation history.

GUIDELINES:
1. **Priority**: Use information from [KNOWLEDGE BASE] when available and relevant
2. **Citations**: Always mention the source filename when using knowledge base information
3. **Fallback**: If the knowledge base doesn't contain relevant info, use your general knowledge but state "Based on my general knowledge..."
4. **Context**: Use [CONVERSATION HISTORY] to understand references (e.g., "it", "that", "the document")
5. **Accuracy**: Don't hallucinate facts. If unsure, say so
6. **Conciseness**: Be clear and concise in your answers

[KNOWLEDGE BASE]:
{rag_ctx}

[CONVERSATION HISTORY]:
{memory_ctx}

Answer the user's question below using the above context."""

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_query)
        ]

        # 2. Hybrid Inference with Failover
        model_used = None
        
        # Try Primary (Gemini) first
        if self.primary_llm:
            try:
                logger.info("Attempting generation with Gemini")
                async for chunk in self.primary_llm.astream(messages):
                    if chunk.content:
                        model_used = "gemini-1.5-flash"
                        yield chunk.content, model_used
                return  # Success, exit
                        
            except Exception as e:
                logger.warning(f"Gemini failed: {e}. Falling back to Ollama")
        else:
            logger.info("Gemini not available, using Ollama directly")
        
        # Fallback to Ollama
        try:
            logger.info(f"Generating with Ollama ({OLLAMA_MODEL})")
            async for chunk in self.fallback_llm.astream(messages):
                if chunk.content:
                    model_used = f"ollama-{OLLAMA_MODEL}"
                    yield chunk.content, model_used
                    
        except Exception as e2:
            logger.critical(f"Both LLMs failed. Ollama error: {e2}")
            error_msg = "I apologize, but I'm currently unable to process your request. Please try again later."
            yield error_msg, "error"

    def trigger_memory_consolidation(self, session_id: str):
        """
        Background task implementing the 3-5 Rule.
        
        Rules:
        - Every 3 turns (6 messages: 3 user + 3 assistant) -> Create SUMMARY_3
        - Every 5 summaries -> Create CHECKPOINT_5
        
        This runs asynchronously after each chat response.
        
        Args:
            session_id: Session to consolidate memory for
        """
        db = SessionLocal()
        try:
            logger.info(f"Starting memory consolidation for session {session_id}")
            
            # --- CHECK 1: Short-term Summary ---
            # Find last summary or checkpoint
            last_summary_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type.in_(['SUMMARY_3', 'CHECKPOINT_5'])
            ).scalar() or 0

            # Get new normal messages
            new_msgs = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'NORMAL',
                ChatEvent.sequence_num > last_summary_seq
            ).order_by(ChatEvent.sequence_num).all()

            # Rule: 3 turns = 6 messages (3 user + 3 assistant)
            if len(new_msgs) >= 6:
                logger.info(f"Session {session_id}: Creating SUMMARY_3 (found {len(new_msgs)} messages)")
                self._create_summary(db, session_id, new_msgs[:6], "SUMMARY_3")

            # --- CHECK 2: Long-term Checkpoint ---
            # Find last checkpoint
            last_checkpoint_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'CHECKPOINT_5'
            ).scalar() or 0

            # Get summaries after last checkpoint
            new_summaries = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'SUMMARY_3',
                ChatEvent.sequence_num > last_checkpoint_seq
            ).order_by(ChatEvent.sequence_num).all()

            # Rule: 5 summaries
            if len(new_summaries) >= 5:
                logger.info(f"Session {session_id}: Creating CHECKPOINT_5 (found {len(new_summaries)} summaries)")
                self._create_summary(db, session_id, new_summaries, "CHECKPOINT_5")
            
            logger.info(f"Memory consolidation completed for session {session_id}")
                
        except Exception as e:
            logger.error(f"Memory consolidation failed for session {session_id}: {e}")
        finally:
            db.close()

    def _create_summary(
        self, 
        db: Session, 
        session_id: str, 
        events: list, 
        target_type: str
    ):
        """
        Generate and store a summary using LLM.
        
        Args:
            db: Database session
            session_id: Current session ID
            events: List of events to summarize
            target_type: Type of summary (SUMMARY_3 or CHECKPOINT_5)
        """
        try:
            # 1. Prepare text for summarization
            if target_type == "SUMMARY_3":
                # Summarizing raw messages
                text_chunk = "\n".join([
                    f"{e.role.upper()}: {e.content}" 
                    for e in events
                ])
                prompt = f"""Summarize the following conversation segment concisely.
Focus on:
- Key facts discussed
- User's questions and intent
- Important decisions or conclusions
- Any action items

Keep it under 100 words.

CONVERSATION:
{text_chunk}

SUMMARY:"""
            else:
                # Summarizing summaries (meta-summary)
                text_chunk = "\n".join([
                    f"Segment {i+1}: {e.content}" 
                    for i, e in enumerate(events)
                ])
                prompt = f"""Create a comprehensive summary of the following conversation summaries.
Synthesize the key themes, facts, and progression of the conversation.
Focus on the most important information that provides context for future interactions.

Keep it under 200 words.

SUMMARIES:
{text_chunk}

COMPREHENSIVE SUMMARY:"""

            # 2. Call LLM (non-streaming, with failover)
            summary_content = None
            
            # Try Gemini first
            if self.primary_llm:
                try:
                    response = self.primary_llm.invoke([HumanMessage(content=prompt)])
                    summary_content = response.content
                    logger.info(f"Generated {target_type} using Gemini")
                except Exception as e:
                    logger.warning(f"Gemini summarization failed: {e}")
            
            # Fallback to Ollama
            if not summary_content:
                try:
                    response = self.fallback_llm.invoke([HumanMessage(content=prompt)])
                    summary_content = response.content
                    logger.info(f"Generated {target_type} using Ollama")
                except Exception as e:
                    logger.error(f"Ollama summarization also failed: {e}")
                    return  # Skip insertion on complete failure

            # 3. Insert into database as hidden event
            next_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id
            ).scalar() + 1
            
            new_event = ChatEvent(
                session_id=session_id,
                sequence_num=next_seq,
                role='system',
                content=summary_content,
                event_type=target_type,
                visibility='HIDDEN'  # Not shown in UI
            )
            db.add(new_event)
            db.commit()
            
            logger.info(f"Session {session_id}: Successfully created and stored {target_type}")
            
        except Exception as e:
            logger.error(f"Failed to create {target_type}: {e}")
            db.rollback()