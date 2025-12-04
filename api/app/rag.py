"""
RAG Query Module
Handles hybrid search, reranking, LLM generation, and 3-3 memory mechanism.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import uuid

import torch
import google.generativeai as genai
from FlagEmbedding import FlagReranker
from pymilvus import (
    Collection, CollectionSchema, FieldSchema, DataType,
    AnnSearchRequest, WeightedRanker, connections
)

from .database import (
    SessionLocal, ChatSession, ChatEvent,
    MessageRole, EventType, Visibility
)
from . import ingest

logger = logging.getLogger(__name__)

# Global reranker model
bge_reranker: Optional[FlagReranker] = None

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured")


def load_reranker_model(model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
    """Load BGE reranker model"""
    global bge_reranker
    logger.info(f"Loading reranker model: {model_name} on {device}")
    bge_reranker = FlagReranker(model_name, use_fp16=True, device=device)
    logger.info("Reranker model loaded successfully")


def create_or_get_collection(collection_name: str) -> Collection:
    """Create or retrieve Milvus collection with hybrid search schema"""
    from pymilvus import utility
    
    if utility.has_collection(collection_name):
        logger.info(f"Collection {collection_name} already exists")
        return Collection(collection_name)
    
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="page_number", dtype=DataType.INT32),
        FieldSchema(name="importance_score", dtype=DataType.FLOAT)
    ]
    
    schema = CollectionSchema(fields, description="RAG Hybrid Collection")
    collection = Collection(collection_name, schema)
    
    # Create indexes
    dense_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256}
    }
    
    sparse_index = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP"
    }
    
    collection.create_index("dense_vector", dense_index)
    collection.create_index("sparse_vector", sparse_index)
    
    logger.info(f"Created collection {collection_name} with hybrid indexes")
    return collection


async def hybrid_search_with_rerank(
    query_text: str,
    collection: Collection,
    filter_expr: str = None,
    top_k_dense: int = 20,
    top_k_sparse: int = 20,
    final_top_k: int = 7,
    alpha: float = 0.2,
    beta: float = 0.8,
    gamma: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Hybrid search with importance score and cross-encoder reranking.
    
    Args:
        query_text: User query
        collection: Milvus collection
        filter_expr: Optional filter expression
        top_k_dense: Top K for dense search
        top_k_sparse: Top K for sparse search
        final_top_k: Final number of chunks to return
        alpha: Weight for importance score in hybrid search
        beta: Weight for cross-encoder in final ranking
        gamma: Weight for importance score in final ranking
    """
    if ingest.bge_m3_model is None:
        raise RuntimeError("Embedding model not loaded")
    
    # Encode query
    query_emb = ingest.bge_m3_model.encode(
        queries=[query_text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )[0]
    
    dense_vec = query_emb['dense'].tolist()
    sparse_vec = query_emb['lexical_weights']
    
    # Create ANN requests
    dense_request = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"efSearch": 128}},
        limit=top_k_dense
    )
    
    sparse_request = AnnSearchRequest(
        data=[sparse_vec],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.1}},
        limit=top_k_sparse
    )
    
    # Hybrid search
    raw_results = collection.hybrid_search(
        reqs=[dense_request, sparse_request],
        rerank=WeightedRanker(0.7, 0.3),
        limit=top_k_dense + top_k_sparse,
        output_fields=["content", "file_id", "page_number", "importance_score"],
        expr=filter_expr
    )[0]
    
    # Aggregate and deduplicate
    candidate_chunks = []
    seen_ids = set()
    
    for hit in raw_results:
        if hit.id in seen_ids:
            continue
        seen_ids.add(hit.id)
        
        imp_score = hit.entity.get("importance_score", 0.0)
        combined_score = hit.distance + alpha * imp_score
        
        candidate_chunks.append({
            "id": hit.id,
            "content": hit.entity.get("content"),
            "file_id": hit.entity.get("file_id"),
            "page_number": hit.entity.get("page_number"),
            "distance": hit.distance,
            "importance_score": imp_score,
            "combined_score": combined_score
        })
    
    # Early return if few candidates
    if len(candidate_chunks) <= final_top_k:
        return candidate_chunks
    
    # Cross-encoder rerank
    if bge_reranker:
        top_candidates = candidate_chunks[:50]
        query_chunk_pairs = [(query_text, chunk["content"]) for chunk in top_candidates]
        
        with torch.no_grad():
            rerank_scores = bge_reranker.compute_score(
                query_chunk_pairs,
                batch_size=16,
                max_length=512,
                normalize=True
            )
        
        # Combine rerank score with importance
        for chunk, score in zip(top_candidates, rerank_scores):
            chunk["final_score"] = beta * score + gamma * chunk["importance_score"]
        
        # Sort and return top K
        final_chunks = sorted(top_candidates, key=lambda x: x["final_score"], reverse=True)[:final_top_k]
        return final_chunks
    
    return candidate_chunks[:final_top_k]


def build_rag_prompt(
    query: str,
    chunks: List[Dict[str, Any]],
    conversation_context: str = ""
) -> str:
    """Build structured RAG prompt with citations"""
    
    prompt = f"""You are an expert financial assistant. Use only the information provided below. Do not hallucinate or invent data.

=== CONVERSATION CONTEXT ===
{conversation_context}

=== DOCUMENTS ===
"""
    
    for i, chunk in enumerate(chunks, 1):
        prompt += f"\n[Document {i}]\n"
        prompt += f"Source: File ID {chunk['file_id']}, Page {chunk['page_number']}\n"
        prompt += f"Content: {chunk['content']}\n"
    
    prompt += f"""
=== USER QUESTION ===
{query}

=== INSTRUCTIONS ===
1. Provide a concise, accurate answer based on the documents.
2. Always cite sources using [Document N] notation.
3. If the documents don't contain sufficient information, say: "I don't have enough information to answer that accurately."
4. Do not guess or use external knowledge unless you explicitly indicate so.
5. Be professional, clear, and structured in your response.
"""
    
    return prompt


def get_conversation_context(session_id: str, max_messages: int = 6) -> str:
    """
    Retrieve conversation context using 3-3 memory mechanism.
    Returns: checkpoint + summaries + recent messages
    """
    db = SessionLocal()
    try:
        # Get checkpoint (most recent master summary)
        checkpoint = db.query(ChatEvent).filter(
            ChatEvent.session_id == uuid.UUID(session_id),
            ChatEvent.event_type == EventType.CHECKPOINT
        ).order_by(ChatEvent.sequence_num.desc()).first()
        
        # Get summaries after checkpoint
        summaries = db.query(ChatEvent).filter(
            ChatEvent.session_id == uuid.UUID(session_id),
            ChatEvent.event_type == EventType.SUMMARY
        )
        
        if checkpoint:
            summaries = summaries.filter(
                ChatEvent.sequence_num > checkpoint.sequence_num
            )
        
        summaries = summaries.order_by(ChatEvent.sequence_num).all()
        
        # Get recent normal messages
        recent_messages = db.query(ChatEvent).filter(
            ChatEvent.session_id == uuid.UUID(session_id),
            ChatEvent.event_type == EventType.NORMAL,
            ChatEvent.visibility == Visibility.VISIBLE
        ).order_by(ChatEvent.sequence_num.desc()).limit(max_messages).all()
        
        recent_messages = list(reversed(recent_messages))
        
        # Build context
        context_parts = []
        
        if checkpoint:
            context_parts.append(f"[Master Summary]\n{checkpoint.content}\n")
        
        if summaries:
            context_parts.append("[Recent Summaries]")
            for summary in summaries:
                context_parts.append(summary.content)
            context_parts.append("")
        
        if recent_messages:
            context_parts.append("[Recent Messages]")
            for msg in recent_messages:
                context_parts.append(f"{msg.role.value}: {msg.content}")
        
        return "\n".join(context_parts)
        
    finally:
        db.close()


async def generate_with_gemini(prompt: str) -> AsyncGenerator[str, None]:
    """Generate response using Gemini API with streaming"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        yield f"[Error: {str(e)}]"


async def generate_with_ollama(prompt: str) -> AsyncGenerator[str, None]:
    """Fallback: Generate response using local Ollama"""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": True
                },
                timeout=60.0
            )
            
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                        
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        yield f"[Error: {str(e)}]"


async def hybrid_search_and_generate(
    query_text: str,
    session_id: str,
    collection: Collection,
    top_k: int = 7,
    use_rag: bool = True
) -> AsyncGenerator[str, None]:
    """
    Main RAG pipeline: search → rerank → generate
    Yields streaming response chunks
    """
    db = SessionLocal()
    
    try:
        # Get next sequence number
        max_seq = db.query(ChatEvent).filter(
            ChatEvent.session_id == uuid.UUID(session_id)
        ).count()
        
        # Save user message
        user_event = ChatEvent(
            session_id=uuid.UUID(session_id),
            sequence_num=max_seq + 1,
            role=MessageRole.USER,
            content=query_text,
            event_type=EventType.NORMAL,
            visibility=Visibility.VISIBLE
        )
        db.add(user_event)
        db.commit()
        
        # Get conversation context
        context = get_conversation_context(session_id)
        
        # Perform RAG if enabled
        chunks = []
        if use_rag and collection:
            chunks = await hybrid_search_with_rerank(
                query_text=query_text,
                collection=collection,
                final_top_k=top_k
            )
        
        # Build prompt
        prompt = build_rag_prompt(query_text, chunks, context)
        
        # Generate response
        full_response = ""
        
        # Try Gemini first, fallback to Ollama
        if GEMINI_API_KEY:
            async for chunk in generate_with_gemini(prompt):
                full_response += chunk
                yield json.dumps({"type": "content", "content": chunk})
        else:
            async for chunk in generate_with_ollama(prompt):
                full_response += chunk
                yield json.dumps({"type": "content", "content": chunk})
        
        # Save assistant response
        assistant_event = ChatEvent(
            session_id=uuid.UUID(session_id),
            sequence_num=max_seq + 2,
            role=MessageRole.ASSISTANT,
            content=full_response,
            event_type=EventType.NORMAL,
            visibility=Visibility.VISIBLE,
            model_used="gemini-2.0-flash" if GEMINI_API_KEY else "llama3.2:3b"
        )
        db.add(assistant_event)
        
        # Update session timestamp
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id)
        ).first()
        if session:
            session.updated_at = datetime.utcnow()
        
        db.commit()
        
        # Send sources metadata
        if chunks:
            sources = [
                {
                    "file_id": chunk["file_id"],
                    "page_number": chunk["page_number"],
                    "score": chunk.get("final_score", chunk.get("combined_score", 0.0))
                }
                for chunk in chunks
            ]
            yield json.dumps({"type": "sources", "sources": sources})
        
        # Check if summary needed (every 3 message pairs = 6 messages)
        message_count = db.query(ChatEvent).filter(
            ChatEvent.session_id == uuid.UUID(session_id),
            ChatEvent.event_type == EventType.NORMAL
        ).count()
        
        if message_count % 6 == 0:
            # Generate summary (simplified - should use LLM)
            summary = f"Summary: Discussed {query_text[:50]}..."
            summary_event = ChatEvent(
                session_id=uuid.UUID(session_id),
                sequence_num=max_seq + 3,
                role=MessageRole.SYSTEM,
                content=summary,
                event_type=EventType.SUMMARY,
                visibility=Visibility.HIDDEN
            )
            db.add(summary_event)
            db.commit()
        
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        yield json.dumps({"type": "error", "error": str(e)})
    
    finally:
        db.close()