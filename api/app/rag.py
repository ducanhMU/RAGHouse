"""
RAG core components: BGE-M3 encoder, Milvus hybrid retrieval, reranker, and LLM router.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
    utility,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@dataclass
class HybridEmbedding:
    dense: List[float]
    sparse: Dict[int, float]
    importance: float


class BgeM3Encoder:
    """
    Minimal adapter for BAAI/bge-m3. The model produces dense + sparse vectors
    from the same forward pass. Sparse vectors are approximated when the native
    tokenizer is unavailable to keep development environments lightweight.
    """

    def __init__(self) -> None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
        self.model = SentenceTransformer(model_name, device=device)

    def encode_texts(self, texts: Sequence[str]) -> List[HybridEmbedding]:
        dense = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        sparse_vectors = [self._bag_of_words(text) for text in texts]
        embeddings: List[HybridEmbedding] = []
        for idx, row in enumerate(dense):
            importance = min(1.0, len(texts[idx]) / 1500.0)
            embeddings.append(
                HybridEmbedding(
                    dense=row.astype(np.float32).tolist(),
                    sparse=sparse_vectors[idx],
                    importance=float(importance),
                )
            )
        return embeddings

    def encode_query(self, query: str) -> HybridEmbedding:
        return self.encode_texts([query])[0]

    @staticmethod
    def _bag_of_words(text: str) -> Dict[int, float]:
        """
        Lightweight sparse approximation: hashed term-frequency vector.
        """
        vocab_size = 4096
        sparse: Dict[int, float] = {}
        tokens = [token.lower() for token in text.split()]
        if not tokens:
            return sparse
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        max_tf = max(tf.values())
        for token, freq in tf.items():
            index = hash(token) % vocab_size
            sparse[index] = freq / max_tf
        return sparse


# ---------------------------------------------------------------------------
# Milvus hybrid store
# ---------------------------------------------------------------------------


class MilvusHybridStore:
    """
    Milvus collection capable of storing dense + sparse vectors, chunk metadata,
    and importance scores. All schema definitions follow the design doc.
    """

    def __init__(self) -> None:
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv(
            "MILVUS_COLLECTION_NAME", "rag_hybrid_collection"
        )
        connections.connect(alias="default", host=host, port=port)
        if not utility.has_collection(self.collection_name):
            self._create_collection()
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _create_collection(self) -> None:
        fields = [
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
            ),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="importance_score", dtype=DataType.FLOAT),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="Hybrid dense+sparse store aligned with BGE-M3 embeddings",
        )
        collection = Collection(name=self.collection_name, schema=schema)
        collection.create_index(
            field_name="dense_vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        collection.create_index(
            field_name="sparse_vector",
            index_params={"index_type": "SPARSE_INVERTED_INDEX"},
        )

    def insert_chunks(
        self,
        *,
        file_id: str,
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[Dict[int, float]],
        texts: Sequence[str],
        pages: Sequence[int],
        importance: Sequence[float],
    ) -> None:
        payload = [
            list(dense_vectors),
            list(sparse_vectors),
            list(texts),
            [file_id] * len(texts),
            list(pages),
            list(importance),
        ]
        self.collection.insert(payload)
        self.collection.flush()

    def delete_file(self, file_id: str) -> None:
        self.collection.delete(expr=f'file_id == "{file_id}"')
        self.collection.flush()

    def stats(self) -> Dict[str, int]:
        return {
            "entities": self.collection.num_entities,
            "indexes": len(self.collection.indexes),
        }

    def hybrid_search(
        self,
        query_embedding: HybridEmbedding,
        *,
        filter_expr: Optional[str] = None,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        final_top_k: int = 7,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rerank_weight: float = 0.8,
        importance_weight: float = 0.2,
        reranker: Optional["Reranker"] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict]:
        dense_request = AnnSearchRequest(
            data=[query_embedding.dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"efSearch": 128}},
            limit=top_k_dense,
        )
        sparse_request = AnnSearchRequest(
            data=[query_embedding.sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.1}},
            limit=top_k_sparse,
        )
        raw_results = self.collection.hybrid_search(
            reqs=[dense_request, sparse_request],
            rerank=WeightedRanker(dense_weight, sparse_weight),
            limit=top_k_dense + top_k_sparse,
            output_fields=[
                "content",
                "file_id",
                "page_number",
                "importance_score",
            ],
            expr=filter_expr,
        )[0]

        deduped: Dict[int, Dict] = {}
        for hit in raw_results:
            if hit.id in deduped:
                continue
            importance = hit.entity.get("importance_score", 0.0)
            deduped[hit.id] = {
                "id": hit.id,
                "text": hit.entity.get("content"),
                "file_id": hit.entity.get("file_id"),
                "page_number": hit.entity.get("page_number"),
                "importance_score": importance,
                "score": 1 - hit.distance,
            }

        candidates = list(deduped.values())
        if not candidates:
            return []

        if reranker and query_text:
            reranked = reranker.rerank(
                query_text,
                candidates,
                final_top_k=final_top_k,
                rerank_weight=rerank_weight,
                importance_weight=importance_weight,
            )
            return reranked

        candidates.sort(key=lambda row: row["score"], reverse=True)
        return candidates[:final_top_k]


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    def __init__(self) -> None:
        model_name = os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-large"
        )
        device = os.getenv("RERANKER_DEVICE", "cpu")
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(
        self,
        query: str,
        documents: Sequence[Dict],
        *,
        final_top_k: int,
        rerank_weight: float,
        importance_weight: float,
    ) -> List[Dict]:
        if not documents:
            return []
        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.model.predict(pairs, convert_to_numpy=True)
        enriched: List[Dict] = []
        for doc, score in zip(documents, scores):
            final_score = (
                rerank_weight * float(score)
                + importance_weight * float(doc.get("importance_score", 0.0))
            )
            enriched.append({**doc, "rerank_score": float(final_score)})
        enriched.sort(key=lambda row: row["rerank_score"], reverse=True)
        return enriched[:final_top_k]


# ---------------------------------------------------------------------------
# LLM Router
# ---------------------------------------------------------------------------


class LLMRouter:
    """
    Route prompts to Gemini (primary) with Ollama fallback. Supports both
    streaming and blocking generations.
    """

    def __init__(self) -> None:
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        self.gemini_key = os.getenv("GOOGLE_API_KEY")

    def stream(self, prompt: str):
        if self.gemini_key:
            try:
                yield from self._stream_gemini(prompt)
                return
            except Exception:  # pragma: no cover - external dependency
                logger.warning("Gemini streaming failed, switching to Ollama")
        yield from self._stream_ollama(prompt)

    def complete(self, prompt: str) -> Dict[str, str]:
        text_parts: List[str] = []
        model = "unknown"
        for chunk in self.stream(prompt):
            text_parts.append(chunk["token"])
            model = chunk["model"]
        return {"text": "".join(text_parts), "model": model}

    def _stream_gemini(self, prompt: str):
        headers = {"Authorization": f"Bearer {self.gemini_key}"}
        payload = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
            "stream": True,
        }
        with requests.post(
            self.gemini_url, headers=headers, json=payload, stream=True, timeout=60
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                token = line.decode("utf-8")
                yield {"token": token, "model": "gemini-2.0-flash"}

    def _stream_ollama(self, prompt: str):
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": float(os.getenv("LLM_TEMPERATURE", "0.2"))},
        }
        with requests.post(
            f"{self.ollama_url}/api/generate", json=payload, stream=True, timeout=120
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                content = line.decode("utf-8")
                yield {"token": content, "model": self.ollama_model}


# ---------------------------------------------------------------------------
# Prompt builder & RAG engine
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Prompt templates for the financial assistant style guide."""

    @staticmethod
    def build(
        *,
        query: str,
        context_chunks: Sequence[Dict],
        conversation: Sequence[Dict],
    ) -> str:
        history_lines = [
            f"{item['role'].upper()}: {item['content']}" for item in conversation
        ]
        history = "\n".join(history_lines) if history_lines else "No previous messages"
        documents = "\n".join(
            [
                f"- Source: {chunk.get('file_id')} page {chunk.get('page_number')}"
                f"\n  Content: {chunk['text']}"
                for chunk in context_chunks
            ]
        )
        return f"""
You are an expert financial assistant. Use only the evidence below.

=== CONVERSATION MEMORY ===
{history}

=== DOCUMENTS ===
{documents}

=== QUESTION ===
{query}

=== INSTRUCTIONS ===
1. Provide concise, structured answers with bullet lists where possible.
2. Reference each fact using (source, page).
3. If information is missing, respond with "I do not have enough information."
"""


class RAGEngine:
    def __init__(self) -> None:
        self.encoder = BgeM3Encoder()
        self.vector_store = MilvusHybridStore()
        self.reranker = Reranker() if os.getenv("ENABLE_RERANKING", "true").lower() == "true" else None
        self.llm = LLMRouter()

    def retrieve(
        self,
        query: str,
        *,
        filter_expr: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        embedding = self.encoder.encode_query(query)
        return self.vector_store.hybrid_search(
            embedding,
            filter_expr=filter_expr,
            final_top_k=top_k,
            reranker=self.reranker,
            query_text=query,
        )

    def generate(
        self,
        *,
        query: str,
        retrieved: Sequence[Dict],
        conversation: Sequence[Dict],
    ) -> Dict:
        prompt = PromptBuilder.build(
            query=query, context_chunks=retrieved, conversation=conversation
        )
        text = ""
        model = "unknown"
        for chunk in self.llm.stream(prompt):
            text += chunk["token"]
            model = chunk["model"]
        return {"answer": text, "model_used": model}

    def stream(
        self,
        *,
        query: str,
        retrieved: Sequence[Dict],
        conversation: Sequence[Dict],
    ):
        prompt = PromptBuilder.build(
            query=query, context_chunks=retrieved, conversation=conversation
        )
        for chunk in self.llm.stream(prompt):
            yield chunk

