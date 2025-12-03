"""
Document ingestion pipeline with background worker and GPU embeddings.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pypdf

from .database import DocumentChunkDAO, FileRegistryDAO
from .rag import BgeM3Encoder, MilvusHybridStore

logger = logging.getLogger(__name__)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 512), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_text(text: str, chunk_tokens: int = 512, overlap: int = 128) -> Iterable[str]:
    tokens = text.split()
    stride = max(chunk_tokens - overlap, 1)
    for start in range(0, len(tokens), stride):
        chunk = " ".join(tokens[start : start + chunk_tokens])
        if chunk.strip():
            yield chunk


def _extract_pdf(path: Path) -> List[Dict]:
    pages: List[Dict] = []
    with path.open("rb") as pdf_file:
        reader = pypdf.PdfReader(pdf_file)
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": idx, "text": text})
    return pages


@dataclass
class IngestionJob:
    file_path: Path
    display_name: str
    metadata: Dict = field(default_factory=dict)
    force: bool = False


class IngestionWorker:
    """
    Asynchronous ingestion worker that keeps the FastAPI thread non-blocking.
    """

    def __init__(
        self,
        *,
        files: FileRegistryDAO,
        chunks: DocumentChunkDAO,
        encoder: BgeM3Encoder,
        vector_store: MilvusHybridStore,
        data_dir: Path,
    ) -> None:
        self.files = files
        self.chunks = chunks
        self.encoder = encoder
        self.vector_store = vector_store
        self.data_dir = data_dir
        self.queue: asyncio.Queue[IngestionJob] = asyncio.Queue(maxsize=100)
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if not self._task:
            self._task = asyncio.create_task(self._run(), name="ingestion-worker")

    async def enqueue(self, job: IngestionJob) -> None:
        await self.queue.put(job)

    async def bootstrap_from_directory(self) -> None:
        for file_path in sorted(self.data_dir.glob("*")):
            if file_path.is_file():
                await self.enqueue(
                    IngestionJob(
                        file_path=file_path,
                        display_name=file_path.name,
                        metadata={"bootstrap": True},
                    )
                )

    async def _run(self) -> None:
        logger.info("Ingestion worker started")
        while True:
            job = await self.queue.get()
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, self._process_job, job
                )
            except Exception as exc:  # pragma: no cover - resilience over strictness
                logger.exception("Ingestion failed for %s (%s)", job.display_name, exc)
            finally:
                self.queue.task_done()

    # ------------------------------------------------------------------ #
    # CPU intensive processing (runs in thread pool)
    # ------------------------------------------------------------------ #

    def _process_job(self, job: IngestionJob) -> None:
        file_path = job.file_path
        if not file_path.exists():
            logger.warning("File missing, skipping ingestion: %s", file_path)
            return

        logger.info("Starting ingestion for %s", job.display_name)
        file_hash = _md5(file_path)

        existing = self.files.lookup_by_hash(file_hash)
        if existing and not job.force:
            logger.info("File already ingested (%s), skipping", job.display_name)
            return

        file_id = existing["id"] if existing else self.files.register(job.display_name, file_hash)
        self.files.update_status(file_id, "PROCESSING")

        pages = _extract_pdf(file_path)
        chunks = []
        chunk_index = 0
        for page in pages:
            for chunk_text in _chunk_text(page["text"]):
                chunks.append(
                    {
                        "index": chunk_index,
                        "text": chunk_text,
                        "page": page["page"],
                    }
                )
                chunk_index += 1

        if not chunks:
            raise RuntimeError("No textual content detected")

        self.chunks.insert_batch(file_id, chunks)

        embeddings = self.encoder.encode_texts([c["text"] for c in chunks])
        self.vector_store.insert_chunks(
            file_id=file_id,
            dense_vectors=[emb.dense for emb in embeddings],
            sparse_vectors=[emb.sparse for emb in embeddings],
            texts=[c["text"] for c in chunks],
            pages=[c["page"] for c in chunks],
            importance=[emb.importance for emb in embeddings],
        )

        self.files.upsert_metadata(
            file_id,
            {
                "pages": len(pages),
                "chunks": len(chunks),
                "bootstrap": job.metadata.get("bootstrap", False),
            },
        )
        self.files.update_status(file_id, "COMPLETED")
        logger.info("Ingestion completed for %s (%s chunks)", job.display_name, len(chunks))