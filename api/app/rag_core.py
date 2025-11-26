# file: api/app/rag_core.py

import os
import logging
from typing import AsyncGenerator, Tuple, List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

# SQL Integration
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Internal Imports
from app.database import ChatEvent, SessionLocal
from app import ingest

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
CLICKHOUSE_URL = os.getenv("CLICKHOUSE_URL", "clickhouse://default:@clickhouse:8123/default")
SUPERSET_BASE_URL = os.getenv("SUPERSET_BASE_URL", "http://superset:8088")

# Logging
logger = logging.getLogger(__name__)

class EnhancedRAGv2:
    """
    RAG V2 Engine with Advanced Features:
    - Hybrid Search (Vector + BM25)
    - Cross-Encoder Re-ranking
    - Text-to-SQL Analytics
    - Visualization Integration
    - Hierarchical Memory (3-5 Rule)
    """
    
    def __init__(self):
        """Initialize the enhanced RAG system."""
        try:
            # === LLM SETUP ===
            # Primary: Gemini 2.0 Flash (Fast, Long Context, Cost-effective)
            if not GOOGLE_API_KEY:
                logger.warning("GOOGLE_API_KEY not set. Gemini will not be available.")
                self.primary_llm = None
            else:
                self.primary_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",  # Latest experimental model
                    temperature=0.3,
                    google_api_key=GOOGLE_API_KEY,
                    convert_system_message_to_human=True,
                    streaming=True,
                    max_output_tokens=4096
                )
                logger.info("Gemini 2.0 Flash initialized successfully")
            
            # Fallback: Ollama (Local, works offline)
            self.fallback_llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.3,
                num_predict=4096
            )
            logger.info(f"Ollama ({OLLAMA_MODEL}) initialized successfully")
            
            # === RERANKER SETUP ===
            # Using lightweight cross-encoder for CPU efficiency
            try:
                self.cross_encoder = HuggingFaceCrossEncoder(
                    model_name="BAAI/bge-reranker-v2-m3"  # Multilingual, efficient
                )
                self.reranker = CrossEncoderReranker(
                    model=self.cross_encoder,
                    top_n=5  # Final top-k after reranking
                )
                logger.info("Cross-encoder reranker initialized")
            except Exception as e:
                logger.warning(f"Reranker initialization failed: {e}. Using without reranking.")
                self.reranker = None
            
            # === SQL DATABASE SETUP ===
            try:
                self.sql_db = SQLDatabase.from_uri(CLICKHOUSE_URL)
                self.sql_chain = None  # Lazy init when needed
                logger.info("ClickHouse database connected")
            except Exception as e:
                logger.warning(f"ClickHouse connection failed: {e}. SQL features disabled.")
                self.sql_db = None
            
            # === CACHE FOR BM25 ===
            self._bm25_retriever = None
            self._last_bm25_update = None
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG V2: {e}")
            raise

    def _get_hybrid_retriever(self, query: str, k: int = 20) -> List:
        """
        HYBRID SEARCH: Combines Dense Vector Search + BM25 Keyword Search
        
        Args:
            query: User query
            k: Number of candidates to retrieve
            
        Returns:
            List of documents from hybrid search
        """
        try:
            vector_db = ingest.get_vector_store()
            
            # === COMPONENT 1: Dense Vector Search ===
            vector_results = vector_db.similarity_search(query, k=k)
            
            # === COMPONENT 2: BM25 Keyword Search ===
            # Refresh BM25 index if needed (cache for 5 minutes)
            import time
            current_time = time.time()
            
            if (self._bm25_retriever is None or 
                self._last_bm25_update is None or 
                current_time - self._last_bm25_update > 300):
                
                # Get all documents from vector store for BM25 indexing
                # Note: This is simplified - in production, maintain separate BM25 index
                all_docs = vector_db.similarity_search("", k=1000)  # Get sample
                
                if all_docs:
                    self._bm25_retriever = BM25Retriever.from_documents(all_docs)
                    self._bm25_retriever.k = k
                    self._last_bm25_update = current_time
                    logger.debug("BM25 index refreshed")
            
            # Get BM25 results
            bm25_results = []
            if self._bm25_retriever:
                try:
                    bm25_results = self._bm25_retriever.get_relevant_documents(query)
                except Exception as e:
                    logger.warning(f"BM25 retrieval failed: {e}")
            
            # === ENSEMBLE: Combine both retrievers ===
            # Weighted combination: 60% vector, 40% BM25
            ensemble_retriever = EnsembleRetriever(
                retrievers=[
                    vector_db.as_retriever(search_kwargs={"k": k}),
                    self._bm25_retriever
                ] if self._bm25_retriever else [vector_db.as_retriever(search_kwargs={"k": k})],
                weights=[0.6, 0.4] if self._bm25_retriever else [1.0]
            )
            
            hybrid_results = ensemble_retriever.get_relevant_documents(query)
            
            logger.info(f"Hybrid search: {len(vector_results)} vector + {len(bm25_results)} BM25 = {len(hybrid_results)} combined")
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            # Fallback to simple vector search
            vector_db = ingest.get_vector_store()
            return vector_db.similarity_search(query, k=k)

    def _rerank_documents(self, query: str, documents: List, top_k: int = 5) -> List:
        """
        RERANKING: Use Cross-Encoder to refine relevance scores
        
        Args:
            query: User query
            documents: Candidate documents from hybrid search
            top_k: Final number of documents to return
            
        Returns:
            Top-k reranked documents
        """
        if not self.reranker or not documents:
            return documents[:top_k]
        
        try:
            # Create compression retriever with reranker
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=None  # We'll use it differently
            )
            
            # Rerank documents
            reranked = self.reranker.compress_documents(documents, query)
            
            logger.info(f"Reranked {len(documents)} docs -> {len(reranked)} final")
            return reranked[:top_k]
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original order.")
            return documents[:top_k]

    def get_retrieval_context(self, query: str, k: int = 5) -> str:
        """
        ENHANCED RETRIEVAL: Hybrid Search + Reranking
        
        Pipeline:
        1. Hybrid Search (Vector + BM25) -> Top 20 candidates
        2. Cross-Encoder Reranking -> Top 5 final
        3. Format with metadata
        
        Args:
            query: Search query
            k: Final number of documents
            
        Returns:
            Formatted context string
        """
        try:
            # Step 1: Hybrid Search (get more candidates for reranking)
            candidates = self._get_hybrid_retriever(query, k=20)
            
            if not candidates:
                return "No relevant documents found in knowledge base."
            
            # Step 2: Rerank to get best k
            final_docs = self._rerank_documents(query, candidates, top_k=k)
            
            # Step 3: Format with metadata
            context_parts = []
            seen_sources = set()
            
            for idx, doc in enumerate(final_docs, 1):
                source = doc.metadata.get('filename', 'Unknown Source')
                page = doc.metadata.get('page', 'N/A')
                
                # Build context string
                context_parts.append(
                    f"[Document {idx}: {source}, Page {page}]\n{doc.page_content}\n"
                )
                seen_sources.add(source)
            
            logger.info(f"Retrieved context from {len(seen_sources)} unique sources")
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            return "Error retrieving documents from knowledge base."

    def _detect_intent(self, query: str) -> str:
        """
        INTENT DETECTION: Classify user query type
        
        Returns:
            'sql' - Query requires database analysis
            'visualization' - User wants charts/dashboards
            'rag' - Standard document Q&A
        """
        query_lower = query.lower()
        
        # SQL indicators
        sql_keywords = ['doanh thu', 'revenue', 'sales', 'sum', 'count', 'average', 
                       'tổng', 'bao nhiêu', 'how many', 'thống kê', 'statistics',
                       'quý', 'quarter', 'tháng', 'month', 'năm', 'year']
        
        # Visualization indicators
        viz_keywords = ['biểu đồ', 'chart', 'graph', 'dashboard', 'visualize', 
                       'plot', 'vẽ', 'xu hướng', 'trend', 'so sánh', 'compare']
        
        if any(kw in query_lower for kw in viz_keywords):
            return 'visualization'
        elif any(kw in query_lower for kw in sql_keywords):
            return 'sql'
        else:
            return 'rag'

    async def _execute_sql_query(self, query: str) -> Dict:
        """
        TEXT-TO-SQL: Convert natural language to SQL and execute
        
        Args:
            query: Natural language query
            
        Returns:
            Dict with SQL query and results
        """
        if not self.sql_db:
            return {
                "error": "SQL database not configured",
                "suggestion": "Please set CLICKHOUSE_URL environment variable"
            }
        
        try:
            # Initialize SQL chain if needed
            if not self.sql_chain:
                llm = self.primary_llm if self.primary_llm else self.fallback_llm
                self.sql_chain = create_sql_query_chain(llm, self.sql_db)
            
            # Generate SQL query
            logger.info(f"Generating SQL for: {query}")
            sql_query = await self.sql_chain.ainvoke({"question": query})
            
            # Execute query
            executor = QuerySQLDataBaseTool(db=self.sql_db)
            result = executor.invoke(sql_query)
            
            logger.info(f"SQL executed successfully: {sql_query[:100]}...")
            
            return {
                "sql": sql_query,
                "result": result,
                "type": "sql_query"
            }
            
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return {
                "error": str(e),
                "type": "sql_error"
            }

    def _get_visualization_link(self, query: str) -> Optional[str]:
        """
        VISUALIZATION: Return Superset dashboard link based on intent
        
        Args:
            query: User query
            
        Returns:
            Dashboard URL or None
        """
        query_lower = query.lower()
        
        # Mapping of keywords to dashboard IDs
        # TODO: Configure these based on your actual Superset dashboards
        dashboard_map = {
            'revenue': '/superset/dashboard/revenue-overview/',
            'doanh thu': '/superset/dashboard/revenue-overview/',
            'sales': '/superset/dashboard/sales-analytics/',
            'customer': '/superset/dashboard/customer-insights/',
            'khách hàng': '/superset/dashboard/customer-insights/',
            'trend': '/superset/dashboard/trend-analysis/',
            'xu hướng': '/superset/dashboard/trend-analysis/',
        }
        
        for keyword, path in dashboard_map.items():
            if keyword in query_lower:
                full_url = f"{SUPERSET_BASE_URL}{path}"
                logger.info(f"Visualization link generated: {full_url}")
                return full_url
        
        return None

    def build_hierarchical_context(self, db: Session, session_id: str) -> str:
        """
        MEMORY MANAGEMENT: Hierarchical 3-5 Rule (unchanged from V1)
        """
        try:
            # A. Get latest Checkpoint
            checkpoint = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'CHECKPOINT_5'
            ).order_by(desc(ChatEvent.sequence_num)).first()
            
            checkpoint_txt = checkpoint.content if checkpoint else "No long-term history yet."
            last_seq = checkpoint.sequence_num if checkpoint else 0

            # B. Get Summaries after checkpoint
            summaries = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'SUMMARY_3',
                ChatEvent.sequence_num > last_seq
            ).order_by(ChatEvent.sequence_num).all()
            
            summary_txt = "\n".join([f"- {s.content}" for s in summaries]) if summaries else "No mid-term summaries."
            
            if summaries:
                last_seq = summaries[-1].sequence_num

            # C. Get Recent Raw Messages
            raw_msgs = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'NORMAL',
                ChatEvent.sequence_num > last_seq
            ).order_by(ChatEvent.sequence_num).limit(10).all()
            
            recent_txt = "\n".join([
                f"{msg.role.upper()}: {msg.content}" 
                for msg in raw_msgs
            ]) if raw_msgs else "No recent messages."

            # D. Assemble
            context = f"""[LONG-TERM MEMORY]:
{checkpoint_txt}

[MID-TERM SUMMARIES]:
{summary_txt}

[RECENT CONVERSATION]:
{recent_txt}"""
            
            logger.debug(f"Built hierarchical context for session {session_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to build context: {e}")
            return "Error loading conversation history."

    async def generate_stream(
        self, 
        memory_ctx: str, 
        rag_ctx: str, 
        user_query: str,
        intent: str = 'rag',
        sql_result: Optional[Dict] = None,
        viz_link: Optional[str] = None
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        MAIN GENERATION: Hybrid LLM with context-aware prompting
        
        Args:
            memory_ctx: Conversation history
            rag_ctx: Retrieved documents
            user_query: Current question
            intent: Query type (rag/sql/visualization)
            sql_result: SQL execution results
            viz_link: Dashboard URL
            
        Yields:
            Tuples of (chunk, model_name)
        """
        # Build system prompt based on intent
        if intent == 'sql' and sql_result:
            system_instruction = f"""You are a data analyst assistant. Answer the user's question using the SQL query results provided.

QUERY RESULTS:
SQL: {sql_result.get('sql', 'N/A')}
Data: {sql_result.get('result', 'N/A')}

Provide a clear, natural language answer interpreting these results. Include relevant numbers and insights.

CONVERSATION HISTORY:
{memory_ctx}"""

        elif intent == 'visualization' and viz_link:
            system_instruction = f"""You are a visualization assistant. The user wants to see charts/dashboards.

AVAILABLE DASHBOARD:
{viz_link}

Provide the dashboard link and explain what insights they can find there.

CONVERSATION HISTORY:
{memory_ctx}"""

        else:  # Standard RAG
            system_instruction = f"""You are an intelligent AI assistant with access to a knowledge base and conversation history.

GUIDELINES:
1. **Priority**: Use information from [KNOWLEDGE BASE] when available
2. **Citations**: Always mention source filename and page number
3. **Accuracy**: Don't hallucinate. If unsure, say so
4. **Context**: Use [CONVERSATION HISTORY] for references
5. **Format**: Be clear and concise

[KNOWLEDGE BASE]:
{rag_ctx}

[CONVERSATION HISTORY]:
{memory_ctx}

Answer the user's question using the context above."""

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_query)
        ]

        # Hybrid Inference with Failover
        model_used = None
        
        # Try Primary (Gemini)
        if self.primary_llm:
            try:
                logger.info("Generating with Gemini 2.0 Flash")
                async for chunk in self.primary_llm.astream(messages):
                    if chunk.content:
                        model_used = "gemini-2.0-flash"
                        yield chunk.content, model_used
                return
                        
            except Exception as e:
                logger.warning(f"Gemini failed: {e}. Falling back to Ollama")
        else:
            logger.info("Gemini not available, using Ollama")
        
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

    async def process_query(
        self,
        db: Session,
        session_id: str,
        user_query: str
    ) -> AsyncGenerator[Tuple[str, str, Dict], None]:
        """
        UNIFIED QUERY PROCESSOR: Routes query based on intent
        
        Args:
            db: Database session
            session_id: Chat session ID
            user_query: User question
            
        Yields:
            Tuples of (chunk, model_name, metadata)
        """
        # Step 1: Detect Intent
        intent = self._detect_intent(user_query)
        logger.info(f"Query intent detected: {intent}")
        
        # Step 2: Build Memory Context
        memory_ctx = self.build_hierarchical_context(db, session_id)
        
        # Step 3: Execute based on intent
        sql_result = None
        viz_link = None
        rag_ctx = ""
        
        if intent == 'sql':
            # Execute SQL query
            sql_result = await self._execute_sql_query(user_query)
            
        elif intent == 'visualization':
            # Get dashboard link
            viz_link = self._get_visualization_link(user_query)
            
        else:  # RAG
            # Enhanced retrieval (Hybrid + Reranking)
            rag_ctx = self.get_retrieval_context(user_query, k=5)
        
        # Step 4: Generate response
        metadata = {
            'intent': intent,
            'sql_result': sql_result,
            'viz_link': viz_link
        }
        
        async for chunk, model in self.generate_stream(
            memory_ctx, rag_ctx, user_query, intent, sql_result, viz_link
        ):
            yield chunk, model, metadata

    def trigger_memory_consolidation(self, session_id: str):
        """
        MEMORY CONSOLIDATION: 3-5 Rule (unchanged from V1)
        """
        db = SessionLocal()
        try:
            logger.info(f"Starting memory consolidation for session {session_id}")
            
            # Check for Summary_3
            last_summary_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type.in_(['SUMMARY_3', 'CHECKPOINT_5'])
            ).scalar() or 0

            new_msgs = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'NORMAL',
                ChatEvent.sequence_num > last_summary_seq
            ).order_by(ChatEvent.sequence_num).all()

            if len(new_msgs) >= 6:
                logger.info(f"Creating SUMMARY_3 ({len(new_msgs)} messages)")
                self._create_summary(db, session_id, new_msgs[:6], "SUMMARY_3")

            # Check for Checkpoint_5
            last_checkpoint_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'CHECKPOINT_5'
            ).scalar() or 0

            new_summaries = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == 'SUMMARY_3',
                ChatEvent.sequence_num > last_checkpoint_seq
            ).order_by(ChatEvent.sequence_num).all()

            if len(new_summaries) >= 5:
                logger.info(f"Creating CHECKPOINT_5 ({len(new_summaries)} summaries)")
                self._create_summary(db, session_id, new_summaries, "CHECKPOINT_5")
                
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
        finally:
            db.close()

    def _create_summary(self, db: Session, session_id: str, events: list, target_type: str):
        """
        SUMMARIZATION: Generate and store memory summaries
        """
        try:
            if target_type == "SUMMARY_3":
                text_chunk = "\n".join([f"{e.role}: {e.content}" for e in events])
                prompt = f"""Summarize this conversation segment concisely (under 100 words):

{text_chunk}

SUMMARY:"""
            else:
                text_chunk = "\n".join([f"Segment {i+1}: {e.content}" for i, e in enumerate(events)])
                prompt = f"""Create a comprehensive summary (under 200 words):

{text_chunk}

COMPREHENSIVE SUMMARY:"""

            # Generate summary
            summary_content = None
            
            if self.primary_llm:
                try:
                    response = self.primary_llm.invoke([HumanMessage(content=prompt)])
                    summary_content = response.content
                except Exception as e:
                    logger.warning(f"Gemini summarization failed: {e}")
            
            if not summary_content:
                try:
                    response = self.fallback_llm.invoke([HumanMessage(content=prompt)])
                    summary_content = response.content
                except Exception as e:
                    logger.error(f"Ollama summarization failed: {e}")
                    return

            # Store summary
            next_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id
            ).scalar() + 1
            
            new_event = ChatEvent(
                session_id=session_id,
                sequence_num=next_seq,
                role='system',
                content=summary_content,
                event_type=target_type,
                visibility='HIDDEN'
            )
            db.add(new_event)
            db.commit()
            
            logger.info(f"Created {target_type} successfully")
            
        except Exception as e:
            logger.error(f"Failed to create {target_type}: {e}")
            db.rollback()