# file api/app/rag.py
"""
RAG Query Module
Handles hybrid search, reranking, LLM generation, and 3-3 memory mechanism.

FIXES:
1. Properly implemented 3-3 memory: SUMMARY every 3 message pairs, CHECKPOINT every 3 summaries
2. LLM-generated summaries and checkpoints using Gemini/Ollama
3. Automatic session title updates on SUMMARY/CHECKPOINT creation
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timezone
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
        [query_text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )
    
    dense_vec = query_emb['dense_vecs'][0].tolist()
    sparse_vec = query_emb['lexical_weights'][0]
    
    # Create ANN requests
    dense_request = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"efSearch": 128}},
        limit=top_k_dense,
        expr=filter_expr
    )
    
    sparse_request = AnnSearchRequest(
        data=[sparse_vec],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.1}},
        limit=top_k_sparse,
        expr=filter_expr
    )
    
    # Hybrid search
    raw_results = collection.hybrid_search(
        reqs=[dense_request, sparse_request],
        rerank=WeightedRanker(0.7, 0.3),
        limit=top_k_dense + top_k_sparse,
        output_fields=["content", "file_id", "page_number", "importance_score"]
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


async def generate_with_gemini(prompt: str, max_tokens: int = 2048) -> tuple[str, bool]:
    """
    Generate response using Gemini API (non-streaming for summaries)
    Returns: (response_text, success_flag)
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens)
        )
        return response.text, True
                
    except Exception as e:
        error_str = str(e)
        logger.error(f"Gemini API error: {e}")
        
        # Check if it's a quota/rate limit error
        if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
            logger.warning("Gemini API quota/rate limit exceeded, will fallback to Ollama")
        
        return None, False


async def generate_with_ollama(prompt: str, max_tokens: int = 2048) -> str:
    """Fallback: Generate response using local Ollama (non-streaming)"""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=200.0) as client:
            response = await client.post(
                "http://rag_ollama:11434/api/generate",
                json={
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens}
                },
                timeout=120.0
            )
            
            data = response.json()
            return data.get("response", "[No response generated]")
                        
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return f"[Error generating with Ollama: {str(e)}]"


async def generate_with_gemini_streaming(prompt: str) -> AsyncGenerator[tuple[str, bool], None]:
    """
    Generate response using Gemini API with streaming
    Yields: (chunk_text, success_flag)
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text, True
                
    except Exception as e:
        error_str = str(e)
        logger.error(f"Gemini API streaming error: {e}")
        
        # Check if it's a quota/rate limit error
        if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
            logger.warning("Gemini API quota/rate limit exceeded, signaling fallback")
        
        # Signal failure
        yield None, False


async def generate_with_ollama_streaming(prompt: str) -> AsyncGenerator[str, None]:
    """Fallback: Generate response using local Ollama with streaming"""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=200.0) as client:
            response = await client.post(
                "http://rag_ollama:11434/api/generate",
                json={
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": True
                },
                timeout=100.0
            )
            
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                        
    except Exception as e:
        logger.error(f"Ollama streaming error: {e}")
        yield f"[Error: {str(e)}]"


async def generate_summary(messages: List[ChatEvent]) -> str:
    """
    Generate a concise summary of recent message pairs using LLM.
    Auto-fallback from Gemini to Ollama on error.
    
    Args:
        messages: List of ChatEvent objects (should be 3 message pairs = 6 messages)
    """
    conversation_text = "\n".join([
        f"{msg.role.value}: {msg.content}" for msg in messages
    ])
    
    prompt = f"""Please create a concise summary (2-3 sentences) of the following conversation exchange:

{conversation_text}

Summary should capture:
- Main topics discussed
- Key questions asked
- Important information provided

Summary:"""
    
    summary = None
    
    # Try Gemini first if API key is available
    if GEMINI_API_KEY:
        summary, success = await generate_with_gemini(prompt, max_tokens=200)
        if not success or not summary:
            logger.warning("Gemini failed for summary generation, falling back to Ollama")
            summary = await generate_with_ollama(prompt, max_tokens=200)
    else:
        # No API key, use Ollama directly
        summary = await generate_with_ollama(prompt, max_tokens=200)
    
    return summary.strip() if summary else "Summary generation failed"


async def generate_checkpoint(summaries: List[ChatEvent], old_checkpoint: Optional[ChatEvent] = None) -> str:
    """
    Generate a master checkpoint by aggregating 3 summaries with optional old checkpoint.
    Auto-fallback from Gemini to Ollama on error.
    
    Args:
        summaries: List of 3 recent SUMMARY events
        old_checkpoint: Previous checkpoint to incorporate (if exists)
    """
    summaries_text = "\n".join([
        f"Summary {i+1}: {summary.content}" 
        for i, summary in enumerate(summaries)
    ])
    
    checkpoint_context = ""
    if old_checkpoint:
        checkpoint_context = f"\n=== Previous Master Summary ===\n{old_checkpoint.content}\n"
    
    prompt = f"""You are creating a master summary (checkpoint) of a conversation. 
{checkpoint_context}
=== Recent Summaries ===
{summaries_text}

Create a comprehensive master summary (3-5 sentences) that:
1. Integrates the recent summaries with the previous master summary (if provided)
2. Captures the overall conversation trajectory
3. Highlights key topics, decisions, and information discussed
4. Maintains important context for future reference

Master Summary:"""
    
    checkpoint = None
    
    # Try Gemini first if API key is available
    if GEMINI_API_KEY:
        checkpoint, success = await generate_with_gemini(prompt, max_tokens=300)
        if not success or not checkpoint:
            logger.warning("Gemini failed for checkpoint generation, falling back to Ollama")
            checkpoint = await generate_with_ollama(prompt, max_tokens=300)
    else:
        # No API key, use Ollama directly
        checkpoint = await generate_with_ollama(prompt, max_tokens=300)
    
    return checkpoint.strip() if checkpoint else "Checkpoint generation failed"


async def generate_session_title(content: str, is_checkpoint: bool = False) -> str:
    """
    Generate a concise session title based on summary or checkpoint content.
    Auto-fallback from Gemini to Ollama on error.
    
    Args:
        content: Summary or checkpoint content
        is_checkpoint: Whether this is from a checkpoint (more comprehensive) or summary
    """
    context_type = "master summary" if is_checkpoint else "conversation summary"
    
    prompt = f"""Based on this {context_type}, create a very concise title (3-7 words max) that captures the main topic:

{content}

Title should be:
- Clear and specific
- Professional
- No quotes or punctuation at the end
- Maximum 7 words

Title:"""
    
    title = None
    
    # Try Gemini first if API key is available
    if GEMINI_API_KEY:
        title, success = await generate_with_gemini(prompt, max_tokens=50)
        if not success or not title:
            logger.warning("Gemini failed for title generation, falling back to Ollama")
            title = await generate_with_ollama(prompt, max_tokens=50)
    else:
        # No API key, use Ollama directly
        title = await generate_with_ollama(prompt, max_tokens=50)
    
    if not title:
        return "Chat Session"
    
    # Clean up the title
    title = title.strip().strip('"').strip("'").strip()
    
    # Truncate if too long
    words = title.split()
    if len(words) > 7:
        title = " ".join(words[:7]) + "..."
    
    return title


async def check_and_create_memory_artifacts(session_id: uuid.UUID, db) -> None:
    """
    Implement 3-3 memory mechanism:
    - Create SUMMARY every 3 message pairs (6 messages)
    - Create CHECKPOINT every 3 summaries
    - Update session title on SUMMARY/CHECKPOINT creation
    """
    
    # Count normal messages since last summary
    last_summary = db.query(ChatEvent).filter(
        ChatEvent.session_id == session_id,
        ChatEvent.event_type.in_([EventType.SUMMARY, EventType.CHECKPOINT])
    ).order_by(ChatEvent.sequence_num.desc()).first()
    
    messages_query = db.query(ChatEvent).filter(
        ChatEvent.session_id == session_id,
        ChatEvent.event_type == EventType.NORMAL
    )
    
    if last_summary:
        messages_query = messages_query.filter(
            ChatEvent.sequence_num > last_summary.sequence_num
        )
    
    recent_messages = messages_query.order_by(ChatEvent.sequence_num).all()
    
    # Check if we need to create a SUMMARY (every 6 normal messages)
    if len(recent_messages) >= 6 and len(recent_messages) % 6 == 0:
        logger.info(f"Creating SUMMARY for session {session_id}")
        
        # Take the last 6 messages for summary
        messages_to_summarize = recent_messages[-6:]
        
        # Generate summary using LLM
        summary_content = await generate_summary(messages_to_summarize)
        
        # Get next sequence number
        max_seq = db.query(ChatEvent).filter(
            ChatEvent.session_id == session_id
        ).order_by(ChatEvent.sequence_num.desc()).first()
        next_seq = (max_seq.sequence_num if max_seq else 0) + 1
        
        # Create SUMMARY event
        summary_event = ChatEvent(
            session_id=session_id,
            sequence_num=next_seq,
            role=MessageRole.SYSTEM,
            content=summary_content,
            event_type=EventType.SUMMARY,
            visibility=Visibility.HIDDEN
        )
        db.add(summary_event)
        db.commit()
        
        logger.info(f"SUMMARY created: {summary_content[:100]}...")
        
        # Update session title based on summary
        new_title = await generate_session_title(summary_content, is_checkpoint=False)
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.title = new_title
            session.updated_at = datetime.now(timezone.utc)
            db.commit()
            logger.info(f"Session title updated to: {new_title}")
        
        # Check if we need to create a CHECKPOINT (every 3 summaries)
        summaries_since_checkpoint = db.query(ChatEvent).filter(
            ChatEvent.session_id == session_id,
            ChatEvent.event_type == EventType.SUMMARY
        )
        
        # Get last checkpoint
        last_checkpoint = db.query(ChatEvent).filter(
            ChatEvent.session_id == session_id,
            ChatEvent.event_type == EventType.CHECKPOINT
        ).order_by(ChatEvent.sequence_num.desc()).first()
        
        if last_checkpoint:
            summaries_since_checkpoint = summaries_since_checkpoint.filter(
                ChatEvent.sequence_num > last_checkpoint.sequence_num
            )
        
        summaries_list = summaries_since_checkpoint.order_by(ChatEvent.sequence_num).all()
        
        if len(summaries_list) >= 3:
            logger.info(f"Creating CHECKPOINT for session {session_id}")
            
            # Take the last 3 summaries
            summaries_to_aggregate = summaries_list[-3:]
            
            # Generate checkpoint using LLM
            checkpoint_content = await generate_checkpoint(
                summaries_to_aggregate,
                old_checkpoint=last_checkpoint
            )
            
            # Get next sequence number
            max_seq = db.query(ChatEvent).filter(
                ChatEvent.session_id == session_id
            ).order_by(ChatEvent.sequence_num.desc()).first()
            next_seq = (max_seq.sequence_num if max_seq else 0) + 1
            
            # Create CHECKPOINT event (append as new row, don't update old one)
            checkpoint_event = ChatEvent(
                session_id=session_id,
                sequence_num=next_seq,
                role=MessageRole.SYSTEM,
                content=checkpoint_content,
                event_type=EventType.CHECKPOINT,
                visibility=Visibility.HIDDEN
            )
            db.add(checkpoint_event)
            db.commit()
            
            logger.info(f"CHECKPOINT created: {checkpoint_content[:100]}...")
            
            # Update session title based on checkpoint (more comprehensive)
            new_title = await generate_session_title(checkpoint_content, is_checkpoint=True)
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                session.title = new_title
                session.updated_at = datetime.now(timezone.utc)
                db.commit()
                logger.info(f"Session title updated to: {new_title}")


async def hybrid_search_and_generate(
    query_text: str,
    session_id: str,
    collection: Collection,
    top_k: int = 7,
    use_rag: bool = True
) -> AsyncGenerator[str, None]:
    """
    Main RAG pipeline: search → rerank → generate → memory management
    Yields streaming response chunks
    Implements automatic fallback from Gemini to Ollama on API errors
    """
    db = SessionLocal()
    
    try:
        session_uuid = uuid.UUID(session_id)
        
        # Get next sequence number
        max_seq_result = db.query(ChatEvent).filter(
            ChatEvent.session_id == session_uuid
        ).order_by(ChatEvent.sequence_num.desc()).first()
        
        next_seq = (max_seq_result.sequence_num if max_seq_result else 0) + 1
        
        # Save user message
        user_event = ChatEvent(
            session_id=session_uuid,
            sequence_num=next_seq,
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
        
        # Generate response (streaming) with fallback logic
        full_response = ""
        model_used = "llama3.2:3b"  # Default fallback
        gemini_failed = False
        
        # Try Gemini first if API key is available
        if GEMINI_API_KEY:
            logger.info("Attempting to generate response with Gemini API")
            try:
                chunk_count = 0
                async for chunk, success in generate_with_gemini_streaming(prompt):
                    if not success:
                        # Gemini failed, need to fallback
                        logger.warning("Gemini streaming failed, falling back to Ollama")
                        gemini_failed = True
                        break
                    
                    if chunk:
                        full_response += chunk
                        chunk_count += 1
                        yield json.dumps({"type": "content", "content": chunk})
                
                # If we got at least some response, consider it successful
                if chunk_count > 0 and not gemini_failed:
                    model_used = "gemini-2.0-flash"
                    logger.info("Successfully generated response with Gemini")
                else:
                    gemini_failed = True
                    
            except Exception as e:
                logger.error(f"Exception during Gemini streaming: {e}")
                gemini_failed = True
        else:
            # No API key, use Ollama directly
            gemini_failed = True
        
        # Fallback to Ollama if Gemini failed or no API key
        if gemini_failed:
            logger.info("Using Ollama for response generation")
            # If we already have partial response from Gemini, clear it
            if full_response:
                logger.warning("Discarding partial Gemini response, regenerating with Ollama")
                full_response = ""
                # Send a signal to clear client-side display
                yield json.dumps({"type": "clear", "message": "Switching to local model..."})
            
            async for chunk in generate_with_ollama_streaming(prompt):
                full_response += chunk
                yield json.dumps({"type": "content", "content": chunk})
            
            model_used = "llama3.2:3b"
        
        # Ensure we have some response
        if not full_response:
            full_response = "I apologize, but I encountered an error generating a response. Please try again."
            yield json.dumps({"type": "content", "content": full_response})
        
        # Save assistant response
        assistant_event = ChatEvent(
            session_id=session_uuid,
            sequence_num=next_seq + 1,
            role=MessageRole.ASSISTANT,
            content=full_response,
            event_type=EventType.NORMAL,
            visibility=Visibility.VISIBLE,
            model_used=model_used
        )
        db.add(assistant_event)
        
        # Update session timestamp
        session = db.query(ChatSession).filter(
            ChatSession.id == session_uuid
        ).first()
        if session:
            session.updated_at = datetime.now(timezone.utc)
        
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
        
        # Check and create memory artifacts (SUMMARY/CHECKPOINT) with title updates
        await check_and_create_memory_artifacts(session_uuid, db)
        
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        yield json.dumps({"type": "error", "error": str(e)})
    
    finally:
        db.close()