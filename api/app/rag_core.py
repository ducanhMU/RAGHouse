"""
RAG V2 - Hybrid Search + Reranking + LLM Integration
"""

import logging
import os
import time
from typing import List, Dict, Optional
import numpy as np
from pymilvus import connections, Collection, utility
from sentence_transformers import CrossEncoder
import google.generativeai as genai
import requests

logger = logging.getLogger(__name__)

# =========================================
# CONFIGURATION
# =========================================

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_collection_v2_hnsw")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L-2-v2")

# =========================================
# MILVUS VECTOR STORE
# =========================================

class MilvusStore:
    """Milvus vector database manager"""
    
    def __init__(self, max_retries=10):
        self.collection_name = COLLECTION_NAME
        self.max_retries = max_retries
        self.collection: Optional[Collection] = None
        self._connect_with_retry()
    
    def _connect_with_retry(self):
        """Connect to Milvus with retry logic"""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"🔌 Connecting to Milvus (Attempt {attempt}/{self.max_retries})...")
                connections.connect(
                    alias="default",
                    host=MILVUS_HOST,
                    port=MILVUS_PORT
                )
                
                # Create collection if not exists
                if not utility.has_collection(self.collection_name):
                    self._create_collection()
                
                self.collection = Collection(self.collection_name)
                self.collection.load()
                logger.info(f"✅ Milvus connected. Collection: {self.collection_name}")
                return
                
            except Exception as e:
                logger.warning(f"❌ Milvus connection failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(3 * attempt)
                else:
                    raise ConnectionError("Failed to connect to Milvus")
    
    def _create_collection(self):
        """Create collection with HNSW index"""
        from pymilvus import CollectionSchema, FieldSchema, DataType
        
        logger.info(f"Creating collection: {self.collection_name}")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="page_number", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields=fields, description="RAG V2 HNSW Collection")
        collection = Collection(name=self.collection_name, schema=schema)
        
        # Create HNSW index for fast ANN search
        index_params = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        logger.info("✅ Collection created with HNSW index")
    
    def insert_vectors(self, vectors: List[List[float]], texts: List[str], 
                      file_ids: List[str], page_numbers: List[int]):
        """Batch insert vectors"""
        data = [
            vectors,
            texts,
            file_ids,
            page_numbers
        ]
        self.collection.insert(data)
        self.collection.flush()
    
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict]:
        """Vector similarity search"""
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["text", "file_id", "page_number"]
        )
        
        hits = []
        for hit in results[0]:
            hits.append({
                "id": str(hit.id),
                "text": hit.entity.get("text"),
                "file_id": hit.entity.get("file_id"),
                "page_number": hit.entity.get("page_number"),
                "distance": hit.distance,
                "score": 1 / (1 + hit.distance)  # Convert distance to similarity
            })
        return hits
    
    def delete_by_file(self, file_id: str):
        """Delete all vectors for a file"""
        expr = f'file_id == "{file_id}"'
        self.collection.delete(expr)
        self.collection.flush()


# =========================================
# EMBEDDING SERVICE (OLLAMA)
# =========================================

class EmbeddingService:
    """GPU-accelerated embedding via Ollama"""
    
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding for maximum GPU utilization"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": batch}
            )
            response.raise_for_status()
            
            embeddings = response.json().get("embedding") or response.json().get("embeddings")
            all_embeddings.extend(embeddings if isinstance(embeddings[0], list) else [embeddings])
        
        return all_embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """Single text embedding"""
        return self.embed_batch([text])[0]


# =========================================
# RERANKER (CROSS-ENCODER)
# =========================================

class Reranker:
    """GPU-accelerated reranking"""
    
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL, max_length=512, device='cuda')
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank documents using cross-encoder"""
        if not documents:
            return []
        
        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.model.predict(pairs)
        
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# =========================================
# HYBRID SEARCH (RRF FUSION)
# =========================================

def reciprocal_rank_fusion(results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Combine multiple ranking lists using RRF"""
    doc_scores = {}
    
    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get("id") or doc.get("text")[:50]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += 1 / (rank + k)
    
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs]


# =========================================
# LLM SERVICE (GEMINI + OLLAMA FALLBACK)
# =========================================

class LLMService:
    """Dual LLM with fallback"""
    
    def __init__(self):
        self.ollama_url = OLLAMA_BASE_URL
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        
        # Configure Gemini if API key exists
        self.use_gemini = bool(GOOGLE_API_KEY)
        if self.use_gemini:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.gemini = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def generate(self, prompt: str, temperature: float = 0.7) -> Dict:
        """Generate with Gemini, fallback to Ollama"""
        # Try Gemini first
        if self.use_gemini:
            try:
                response = self.gemini.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=2048
                    )
                )
                return {
                    "text": response.text,
                    "model": "gemini-2.0-flash",
                    "source": "cloud"
                }
            except Exception as e:
                logger.warning(f"Gemini failed: {e}. Falling back to Ollama...")
        
        # Fallback to Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                }
            )
            response.raise_for_status()
            return {
                "text": response.json()["response"],
                "model": self.ollama_model,
                "source": "local"
            }
        except Exception as e:
            logger.error(f"Ollama failed: {e}")
            raise


# =========================================
# MAIN RAG ENGINE
# =========================================

class RAGEngine:
    """Complete RAG pipeline"""
    
    def __init__(self, db_manager, keyword_search_fn):
        self.vector_store = MilvusStore()
        self.embedder = EmbeddingService()
        self.reranker = Reranker()
        self.llm = LLMService()
        self.keyword_search_fn = keyword_search_fn
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search with RRF fusion"""
        # 1. Semantic search
        query_vector = self.embedder.embed_single(query)
        semantic_results = self.vector_store.search(query_vector, top_k=10)
        
        # 2. Keyword search
        keyword_results = self.keyword_search_fn(query, limit=10)
        keyword_results = [
            {**r, "id": str(r["id"]), "text": r["content"]}
            for r in keyword_results
        ]
        
        # 3. RRF Fusion
        fused_results = reciprocal_rank_fusion([semantic_results, keyword_results])
        
        # 4. Reranking
        final_results = self.reranker.rerank(query, fused_results[:10], top_k=top_k)
        
        return final_results
    
    def generate_answer(self, query: str, context_docs: List[Dict], 
                       chat_history: List[Dict] = None) -> Dict:
        """Generate answer with RAG"""
        # Build context
        context_text = "\n\n".join([
            f"[Document {i+1}] (Page {d.get('page_number', 'N/A')})\n{d['text']}"
            for i, d in enumerate(context_docs)
        ])
        
        # Build chat history
        history_text = ""
        if chat_history:
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in chat_history[-5:]  # Last 5 messages
            ])
        
        # Build prompt
        prompt = f"""You are a helpful AI assistant with access to a knowledge base.

CHAT HISTORY:
{history_text if history_text else "No previous messages"}

RELEVANT DOCUMENTS:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer based on the documents provided
- Cite sources using [Document X] notation
- If information is not in the documents, say so clearly
- Be concise and precise

ANSWER:"""
        
        # Generate
        response = self.llm.generate(prompt)
        
        return {
            "answer": response["text"],
            "model_used": response["model"],
            "source_type": "knowledge_base" if context_docs else "llm_only",
            "sources": context_docs
        }