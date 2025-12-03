"""
Deprecated shim retained so older imports keep working.  The full RAG stack
is implemented in `rag.py`; this module simply re-exports the relevant
classes for backwards compatibility.
"""

from .rag import (  # noqa: F401
    BgeM3Encoder as EmbeddingService,
    LLMRouter as LLMService,
    MilvusHybridStore as MilvusStore,
    RAGEngine,
    Reranker,
)

