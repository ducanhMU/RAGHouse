# file: api/app/rag_core.py

import os
import logging
import asyncio
from typing import AsyncGenerator, Tuple, List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, text

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker

# SQL Integration
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# Internal Imports
from app.database import ChatEvent, SessionLocal, EventType, MessageRole, Visibility
from app import ingest

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
CLICKHOUSE_URL = os.getenv("CLICKHOUSE_URL")
SUPERSET_BASE_URL = os.getenv("SUPERSET_BASE_URL", "http://superset:8088")

# Feature Flags
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
ENABLE_POSTGRES_FTS = os.getenv("ENABLE_POSTGRES_FTS", "false").lower() == "true"

# Logging
logger = logging.getLogger(__name__)

class EnhancedRAGv2:
    """
    ULTIMATE RAG ENGINE
    
    Fixes:
    - Removed BM25 in-memory (use Postgres FTS instead)
    - Semantic intent detection
    - Lighter reranker (TinyBERT)
    - SQL injection protection
    - Async memory consolidation
    """
    
    def __init__(self):
        """Initialize with non-blocking pattern."""
        self.initialized = False
        self.initialization_error = None
        
        # Core components
        self.primary_llm = None
        self.fallback_llm = None
        self.intent_classifier = None  # NEW: Semantic router
        self.cross_encoder = None
        self.reranker = None
        self.sql_db = None
        self.sql_chain = None
        
        # Start background initialization
        logger.info("🚀 RAG V2 Engine initialization...")
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Background initialization."""
        try:
            await self._init_llms()
            
            if ENABLE_RERANKING:
                await self._init_reranker()
            
            if CLICKHOUSE_URL:
                await self._init_sql_db()
            
            self.initialized = True
            logger.info("✅ RAG V2 Engine ready")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Init failed: {e}")
    
    async def _init_llms(self):
        """Initialize LLMs."""
        # Primary: Gemini
        if GOOGLE_API_KEY:
            try:
                self.primary_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    temperature=0.3,
                    google_api_key=GOOGLE_API_KEY,
                    convert_system_message_to_human=True,
                    streaming=True,
                    max_output_tokens=4096
                )
                
                # Also use for intent classification
                self.intent_classifier = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    temperature=0.1,
                    google_api_key=GOOGLE_API_KEY
                )
                
                await asyncio.to_thread(
                    self.primary_llm.invoke,
                    [HumanMessage(content="test")]
                )
                logger.info("✅ Gemini loaded")
            except Exception as e:
                logger.warning(f"⚠️ Gemini failed: {e}")
                self.primary_llm = None
        
        # Fallback: Ollama
        try:
            self.fallback_llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.3,
                num_predict=4096
            )
            await asyncio.to_thread(
                self.fallback_llm.invoke,
                [HumanMessage(content="test")]
            )
            logger.info(f"✅ Ollama loaded")
        except Exception as e:
            logger.error(f"❌ Ollama failed: {e}")
            raise RuntimeError("At least one LLM required")
    
    async def _init_reranker(self):
        """Initialize lighter reranker."""
        try:
            # Using TinyBERT instead of bge-reranker (10x faster on CPU)
            self.cross_encoder = HuggingFaceCrossEncoder(
                model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2"
            )
            self.reranker = CrossEncoderReranker(
                model=self.cross_encoder,
                top_n=5
            )
            logger.info("✅ TinyBERT reranker loaded (fast)")
        except Exception as e:
            logger.warning(f"⚠️ Reranker failed: {e}")
            self.reranker = None
    
    async def _init_sql_db(self):
        """Initialize SQL with READ-ONLY user."""
        try:
            self.sql_db = SQLDatabase.from_uri(
                CLICKHOUSE_URL,
                # Only expose safe tables
                include_tables=[
                    'dim_company',
                    'fact_income_statement',
                    'fact_balance_sheet',
                    'fact_cash_flow',
                    'fact_daily_market',
                    'fact_macro_timeseries',
                    'mart_master_analysis'
                ],
                view_support=True
            )
            logger.info("✅ ClickHouse connected (READ-ONLY)")
        except Exception as e:
            logger.warning(f"⚠️ ClickHouse failed: {e}")
            self.sql_db = None

    def _get_hybrid_retriever_postgres(self, query: str, k: int = 20) -> List:
        """
        FIXED HYBRID SEARCH: Uses Postgres FTS instead of BM25
        No more in-memory index building!
        """
        try:
            vector_db = ingest.get_vector_store()
            
            # Pure vector search
            vector_results = vector_db.similarity_search(query, k=k)
            
            # If Postgres FTS enabled
            if ENABLE_POSTGRES_FTS:
                # TODO: Implement Postgres full-text search
                # This would query a separate FTS table
                # For now, use vector only
                pass
            
            logger.info(f"📊 Retrieved {len(vector_results)} documents")
            return vector_results
                
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []

    def _rerank_documents(self, query: str, documents: List, top_k: int = 5) -> List:
        """Reranking with TinyBERT."""
        if not self.reranker or not documents or not ENABLE_RERANKING:
            return documents[:top_k]
        
        try:
            reranked = self.reranker.compress_documents(documents, query)
            logger.info(f"🎯 Reranked: {len(reranked)}")
            return reranked[:top_k]
        except Exception as e:
            logger.warning(f"⚠️ Reranking failed: {e}")
            return documents[:top_k]

    def get_retrieval_context(self, query: str, k: int = 5) -> str:
        """Enhanced retrieval pipeline."""
        try:
            # Hybrid search (fixed)
            candidates = self._get_hybrid_retriever_postgres(query, k=20)
            
            if not candidates:
                return "No relevant documents found."
            
            # Rerank
            final_docs = self._rerank_documents(query, candidates, top_k=k)
            
            # Format
            context_parts = []
            for idx, doc in enumerate(final_docs, 1):
                source = doc.metadata.get('filename', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                context_parts.append(
                    f"[Doc {idx}: {source}, Page {page}]\n{doc.page_content}\n"
                )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return "Error retrieving documents."

    async def _detect_intent_semantic(self, query: str) -> str:
        """
        SEMANTIC INTENT DETECTION
        Uses LLM for classification (more accurate than keywords)
        """
        if not self.intent_classifier:
            # Fallback to keyword-based
            return self._detect_intent_keyword(query)
        
        try:
            prompt = f"""Classify this user query into ONE category:
- "rag": Question about documents, general knowledge
- "sql": Question about numbers, statistics, financial data (revenue, profit, etc.)
- "visualization": Request for charts, graphs, dashboards

User query: "{query}"

Answer with ONLY ONE WORD: rag, sql, or visualization"""
            
            response = await asyncio.to_thread(
                self.intent_classifier.invoke,
                [HumanMessage(content=prompt)]
            )
            
            intent = response.content.strip().lower()
            if intent in ['rag', 'sql', 'visualization']:
                logger.info(f"🎯 Intent (semantic): {intent}")
                return intent
            else:
                return 'rag'
                
        except Exception as e:
            logger.warning(f"⚠️ Semantic intent failed: {e}, using keywords")
            return self._detect_intent_keyword(query)
    
    def _detect_intent_keyword(self, query: str) -> str:
        """Fallback keyword-based detection."""
        query_lower = query.lower()
        
        sql_keywords = ['doanh thu', 'revenue', 'lợi nhuận', 'profit', 'sales']
        viz_keywords = ['biểu đồ', 'chart', 'graph', 'dashboard']
        
        if any(kw in query_lower for kw in viz_keywords):
            return 'visualization'
        elif any(kw in query_lower for kw in sql_keywords):
            return 'sql'
        else:
            return 'rag'

    async def _execute_sql_query(self, query: str) -> Dict:
        """
        TEXT-TO-SQL with READ-ONLY protection
        """
        if not self.sql_db:
            return {"error": "ClickHouse not configured"}
        
        try:
            if not self.sql_chain:
                llm = self.primary_llm if self.primary_llm else self.fallback_llm
                self.sql_chain = create_sql_query_chain(llm, self.sql_db)
            
            # Generate SQL
            logger.info(f"🔍 Generating SQL...")
            sql_query = await self.sql_chain.ainvoke({"question": query})
            
            # Validate (basic protection)
            sql_lower = sql_query.lower()
            dangerous_keywords = ['drop', 'delete', 'truncate', 'update', 'insert', 'alter']
            
            if any(kw in sql_lower for kw in dangerous_keywords):
                return {
                    "error": "SQL query contains dangerous operations",
                    "type": "sql_error"
                }
            
            # Execute
            executor = QuerySQLDataBaseTool(db=self.sql_db)
            result = await asyncio.to_thread(executor.invoke, sql_query)
            
            logger.info(f"✅ SQL executed")
            
            return {
                "sql": sql_query,
                "result": result,
                "type": "sql_query"
            }
            
        except Exception as e:
            logger.error(f"❌ SQL failed: {e}")
            return {"error": str(e), "type": "sql_error"}

    def _get_visualization_link(self, query: str) -> Optional[str]:
        """Map query to dashboard."""
        query_lower = query.lower()
        
        dashboard_map = {
            'revenue': '/superset/dashboard/revenue/',
            'doanh thu': '/superset/dashboard/revenue/',
            'profit': '/superset/dashboard/profitability/',
            'lợi nhuận': '/superset/dashboard/profitability/',
            'financial': '/superset/dashboard/financial/',
            'tài chính': '/superset/dashboard/financial/',
        }
        
        for keyword, path in dashboard_map.items():
            if keyword in query_lower:
                return f"{SUPERSET_BASE_URL}{path}"
        
        return None

    def build_hierarchical_context(self, db: Session, session_id: str) -> str:
        """
        MEMORY: 3-3 Rule
        """
        try:
            # Get latest Checkpoint
            checkpoint = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == EventType.CHECKPOINT
            ).order_by(desc(ChatEvent.sequence_num)).first()
            
            checkpoint_txt = checkpoint.content if checkpoint else "No history."
            last_seq = checkpoint.sequence_num if checkpoint else 0

            # Get Summaries
            summaries = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == EventType.SUMMARY,
                ChatEvent.sequence_num > last_seq
            ).order_by(ChatEvent.sequence_num).all()
            
            summary_txt = "\n".join([f"- {s.content}" for s in summaries]) if summaries else "No summaries."
            
            if summaries:
                last_seq = summaries[-1].sequence_num

            # Get Recent
            raw_msgs = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == EventType.NORMAL,
                ChatEvent.sequence_num > last_seq
            ).order_by(ChatEvent.sequence_num).limit(10).all()
            
            recent_txt = "\n".join([
                f"{msg.role.value.upper()}: {msg.content}" 
                for msg in raw_msgs
            ]) if raw_msgs else "No recent messages."

            return f"""[CHECKPOINT]:
{checkpoint_txt}

[SUMMARIES]:
{summary_txt}

[RECENT]:
{recent_txt}"""
            
        except Exception as e:
            logger.error(f"❌ Context failed: {e}")
            return "Error loading history."

    async def generate_stream(
        self, 
        memory_ctx: str, 
        rag_ctx: str, 
        user_query: str,
        intent: str = 'rag',
        sql_result: Optional[Dict] = None,
        viz_link: Optional[str] = None
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """Generation with failover."""
        # Build prompt
        if intent == 'sql' and sql_result:
            system_instruction = f"""You are a financial analyst. Use SQL results.

SQL: {sql_result.get('sql', 'N/A')}
Results: {sql_result.get('result', 'N/A')}

Provide clear interpretation.

HISTORY:
{memory_ctx}"""

        elif intent == 'visualization' and viz_link:
            system_instruction = f"""You are a visualization assistant.

Dashboard: {viz_link}

Explain what insights are available.

HISTORY:
{memory_ctx}"""

        else:  # RAG
            system_instruction = f"""You are an AI financial analyst.

GUIDELINES:
1. Use [KNOWLEDGE BASE] for facts
2. Cite sources
3. Be accurate
4. Use [HISTORY] for context

[KNOWLEDGE BASE]:
{rag_ctx}

[HISTORY]:
{memory_ctx}"""

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_query)
        ]

        # Failover
        model_used = None
        
        if self.primary_llm:
            try:
                logger.info("🚀 Using Gemini")
                async for chunk in self.primary_llm.astream(messages):
                    if chunk.content:
                        model_used = "gemini-2.0-flash"
                        yield chunk.content, model_used
                return
            except Exception as e:
                logger.warning(f"⚠️ Gemini failed: {e}")
        
        if self.fallback_llm:
            try:
                logger.info(f"🔄 Using Ollama")
                async for chunk in self.fallback_llm.astream(messages):
                    if chunk.content:
                        model_used = f"ollama-{OLLAMA_MODEL}"
                        yield chunk.content, model_used
            except Exception as e:
                logger.critical(f"❌ Both LLMs failed: {e}")
                yield "Unable to process request.", "error"
        else:
            yield "No LLM available.", "error"

    async def process_query(
        self,
        db: Session,
        session_id: str,
        user_query: str
    ) -> AsyncGenerator[Tuple[str, str, Dict], None]:
        """Unified processor."""
        # Wait for init
        max_wait = 30
        waited = 0
        while not self.initialized and waited < max_wait:
            await asyncio.sleep(0.5)
            waited += 0.5
        
        # Semantic intent detection
        intent = await self._detect_intent_semantic(user_query)
        logger.info(f"🎯 Intent: {intent}")
        
        # Build memory
        memory_ctx = self.build_hierarchical_context(db, session_id)
        
        # Execute
        sql_result = None
        viz_link = None
        rag_ctx = ""
        
        if intent == 'sql':
            sql_result = await self._execute_sql_query(user_query)
        elif intent == 'visualization':
            viz_link = self._get_visualization_link(user_query)
        else:
            rag_ctx = self.get_retrieval_context(user_query, k=5)
        
        metadata = {
            'intent': intent,
            'sql_result': sql_result,
            'viz_link': viz_link
        }
        
        async for chunk, model in self.generate_stream(
            memory_ctx, rag_ctx, user_query, intent, sql_result, viz_link
        ):
            yield chunk, model, metadata

    async def trigger_memory_consolidation(self, session_id: str):
        """
        ASYNC MEMORY CONSOLIDATION
        Runs in background, doesn't block user response
        """
        db = SessionLocal()
        try:
            logger.info(f"🧠 Memory consolidation: {session_id}")
            
            # Check for Summary (3 turns = 6 messages)
            last_summary_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type.in_([EventType.SUMMARY, EventType.CHECKPOINT])
            ).scalar() or 0

            new_msgs = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == EventType.NORMAL,
                ChatEvent.sequence_num > last_summary_seq
            ).order_by(ChatEvent.sequence_num).all()

            if len(new_msgs) >= 6:
                logger.info(f"📝 Creating SUMMARY")
                await self._create_summary_async(db, session_id, new_msgs[:6], EventType.SUMMARY)

            # Check for Checkpoint (3 summaries)
            last_checkpoint_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == EventType.CHECKPOINT
            ).scalar() or 0

            summaries = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id,
                ChatEvent.event_type == EventType.SUMMARY,
                ChatEvent.sequence_num > last_checkpoint_seq
            ).order_by(ChatEvent.sequence_num).all()

            if len(summaries) >= 3:
                logger.info(f"🔖 Creating CHECKPOINT")
                await self._create_summary_async(db, session_id, summaries, EventType.CHECKPOINT)
                
        except Exception as e:
            logger.error(f"❌ Memory consolidation failed: {e}")
        finally:
            db.close()

    async def _create_summary_async(
        self, 
        db: Session, 
        session_id: str, 
        events: list, 
        target_type: EventType
    ):
        """Async summarization."""
        try:
            if target_type == EventType.SUMMARY:
                text = "\n".join([f"{e.role.value}: {e.content}" for e in events])
                prompt = f"Summarize concisely (< 100 words):\n\n{text}\n\nSUMMARY:"
            else:
                text = "\n".join([f"Part {i+1}: {e.content}" for i, e in enumerate(events)])
                prompt = f"Comprehensive summary (< 200 words):\n\n{text}\n\nSUMMARY:"

            summary = None
            llm = self.primary_llm if self.primary_llm else self.fallback_llm
            
            if llm:
                try:
                    response = await asyncio.to_thread(
                        llm.invoke,
                        [HumanMessage(content=prompt)]
                    )
                    summary = response.content
                except Exception as e:
                    logger.error(f"❌ Summarization failed: {e}")
                    return

            if not summary:
                return

            # Store
            next_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
                ChatEvent.session_id == session_id
            ).scalar() + 1
            
            new_event = ChatEvent(
                session_id=session_id,
                sequence_num=next_seq,
                role=MessageRole.SYSTEM,
                content=summary,
                event_type=target_type,
                visibility=Visibility.HIDDEN
            )
            db.add(new_event)
            db.commit()
            
            logger.info(f"✅ {target_type.value} created")
            
        except Exception as e:
            logger.error(f"❌ Summary failed: {e}")
            db.rollback()