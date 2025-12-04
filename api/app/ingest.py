"""
File Ingestion Module
Handles document extraction, chunking, embedding generation, and vector insertion into Milvus.
"""

import os
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

import fitz  # PyMuPDF
from docx import Document
from FlagEmbedding import BGEM3FlagModel
from pymilvus import Collection, utility
import numpy as np

from .database import SessionLocal, FileRegistry, FileStatus

logger = logging.getLogger(__name__)

# Global model instances (loaded at startup)
bge_m3_model: Optional[BGEM3FlagModel] = None


def load_embedding_model(model_name: str = "BAAI/bge-m3", device: str = "cuda"):
    """Load BGE-M3 embedding model on GPU"""
    global bge_m3_model
    logger.info(f"Loading embedding model: {model_name} on {device}")
    bge_m3_model = BGEM3FlagModel(model_name, use_fp16=True, device=device)
    logger.info("Embedding model loaded successfully")


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of file for deduplication"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with page metadata"""
    pages = []
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append({
                    "page_number": page_num,
                    "content": text.strip()
                })
        doc.close()
        logger.info(f"Extracted {len(pages)} pages from {file_path}")
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {e}")
        raise
    return pages


def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from DOCX (single page since Word doesn't have page concept)"""
    try:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        content = "\n".join(paragraphs)
        logger.info(f"Extracted content from {file_path}")
        return [{"page_number": 1, "content": content}]
    except Exception as e:
        logger.error(f"Error extracting DOCX {file_path}: {e}")
        raise


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by token count (simplified word-based)"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def generate_embeddings(texts: List[str]) -> Dict[str, Any]:
    """Generate dense + sparse embeddings using BGE-M3"""
    if bge_m3_model is None:
        raise RuntimeError("Embedding model not loaded")
    
    logger.info(f"Generating embeddings for {len(texts)} chunks")
    embeddings = bge_m3_model.encode(
        texts,
        batch_size=8,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )
    
    return {
        "dense": embeddings['dense'],
        "sparse": embeddings['lexical_weights']
    }


async def insert_vectors_to_milvus(
    collection: Collection,
    chunks: List[Dict[str, Any]],
    file_id: str
):
    """Insert dense + sparse vectors into Milvus collection"""
    if not chunks:
        logger.warning(f"No chunks to insert for file {file_id}")
        return
    
    texts = [chunk["content"] for chunk in chunks]
    embeddings = generate_embeddings(texts)
    
    # Prepare data for insertion
    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            "dense_vector": embeddings["dense"][i].tolist(),
            "sparse_vector": embeddings["sparse"][i],
            "content": chunk["content"],
            "file_id": file_id,
            "page_number": chunk["page_number"],
            "importance_score": chunk.get("importance_score", 0.0)
        })
    
    # Insert in batches
    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        collection.insert(batch)
        logger.info(f"Inserted batch {i//batch_size + 1} for file {file_id}")
    
    collection.flush()
    logger.info(f"Successfully inserted {len(data)} vectors for file {file_id}")


async def process_file(
    file_path: str,
    file_id: str,
    collection: Collection
) -> Dict[str, Any]:
    """Main ingestion pipeline: extract → chunk → embed → insert"""
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Extract text based on file type
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.pdf':
            pages = extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            pages = extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Chunk each page
        all_chunks = []
        for page_data in pages:
            page_chunks = chunk_text(page_data["content"])
            for chunk_str in page_chunks:
                all_chunks.append({
                    "content": chunk_str,
                    "page_number": page_data["page_number"],
                    "importance_score": 0.0  # Can be enhanced with importance scoring
                })
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(pages)} pages")
        
        # Insert into Milvus
        await insert_vectors_to_milvus(collection, all_chunks, file_id)
        
        return {
            "success": True,
            "pages": len(pages),
            "chunks": len(all_chunks)
        }
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


async def ingest_file_task(
    file_id: str,
    file_path: str,
    collection: Collection
):
    """Background task for file ingestion with DB status updates"""
    db = SessionLocal()
    try:
        # Update status to PROCESSING
        file_record = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
        if not file_record:
            logger.error(f"File record not found: {file_id}")
            return
        
        file_record.status = FileStatus.PROCESSING
        db.commit()
        
        # Process file
        result = await process_file(file_path, file_id, collection)
        
        # Update status based on result
        if result["success"]:
            file_record.status = FileStatus.COMPLETED
            file_record.meta_info = {
                **file_record.meta_info,
                "pages": result.get("pages", 0),
                "chunks": result.get("chunks", 0),
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
        else:
            file_record.status = FileStatus.FAILED
            file_record.meta_info = {
                **file_record.meta_info,
                "error": result.get("error", "Unknown error")
            }
        
        db.commit()
        logger.info(f"File {file_id} ingestion completed with status: {file_record.status}")
        
    except Exception as e:
        logger.error(f"Critical error in ingestion task for {file_id}: {e}", exc_info=True)
        file_record = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
        if file_record:
            file_record.status = FileStatus.FAILED
            file_record.meta_info = {**file_record.meta_info, "error": str(e)}
            db.commit()
    finally:
        db.close()


async def auto_ingest_directory(
    directory: str,
    collection: Collection
):
    """Auto-ingest all files from a directory (e.g., api/data/) on startup"""
    db = SessionLocal()
    try:
        data_dir = Path(directory)
        if not data_dir.exists():
            logger.warning(f"Data directory does not exist: {directory}")
            return
        
        files = list(data_dir.glob("**/*.pdf")) + list(data_dir.glob("**/*.docx"))
        logger.info(f"Found {len(files)} files in {directory}")
        
        for file_path in files:
            file_hash = compute_file_hash(str(file_path))
            
            # Check if already processed
            existing = db.query(FileRegistry).filter(
                FileRegistry.file_hash == file_hash
            ).first()
            
            if existing and existing.status == FileStatus.COMPLETED:
                logger.info(f"File already processed: {file_path.name}")
                continue
            
            # Register or update file
            if not existing:
                file_record = FileRegistry(
                    file_hash=file_hash,
                    filename=file_path.name,
                    status=FileStatus.PENDING,
                    meta_info={"source": "auto_ingest"}
                )
                db.add(file_record)
                db.commit()
                db.refresh(file_record)
                file_id = str(file_record.id)
            else:
                file_id = str(existing.id)
            
            # Start ingestion
            logger.info(f"Starting ingestion for: {file_path.name}")
            await ingest_file_task(file_id, str(file_path), collection)
        
    finally:
        db.close()