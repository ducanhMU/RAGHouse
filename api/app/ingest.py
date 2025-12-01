"""
RAG V2 - Document Ingestion Pipeline
"""

import logging
import hashlib
from typing import List, Dict
from pathlib import Path
import pypdf
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# =========================================
# PDF PROCESSING
# =========================================

def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """Extract text from PDF with page numbers"""
    chunks = []
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    chunks.append({
                        "text": text,
                        "page": page_num
                    })
        logger.info(f"✅ Extracted {len(chunks)} pages from PDF")
        return chunks
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return []


# =========================================
# TEXT CHUNKING
# =========================================

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def process_document_chunks(pages: List[Dict]) -> List[Dict]:
    """Convert pages into semantic chunks"""
    all_chunks = []
    chunk_index = 0
    
    for page_data in pages:
        text_chunks = chunk_text(page_data["text"])
        for chunk_text in text_chunks:
            all_chunks.append({
                "index": chunk_index,
                "text": chunk_text,
                "page": page_data["page"]
            })
            chunk_index += 1
    
    logger.info(f"✅ Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


# =========================================
# FILE HASH
# =========================================

def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash for deduplication"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


# =========================================
# MAIN INGESTION PIPELINE
# =========================================

class IngestionPipeline:
    """Orchestrates dual ingestion (Postgres + Milvus)"""
    
    def __init__(self, file_registry, doc_chunks, embedder, vector_store):
        self.file_registry = file_registry
        self.doc_chunks = doc_chunks
        self.embedder = embedder
        self.vector_store = vector_store
    
    def ingest_file(self, file_path: str, filename: str) -> Dict:
        """Complete ingestion workflow"""
        logger.info(f"🚀 Starting ingestion: {filename}")
        
        try:
            # 1. Calculate hash
            file_hash = calculate_file_hash(file_path)
            
            # 2. Check duplicate
            existing = self.file_registry.check_duplicate(file_hash)
            if existing:
                logger.warning(f"⚠️ File already exists: {existing['filename']}")
                return {"status": "duplicate", "file_id": existing["id"]}
            
            # 3. Register file
            file_id = self.file_registry.register_file(filename, file_hash)
            self.file_registry.update_status(file_id, "PROCESSING")
            
            # 4. Extract text
            pages = extract_text_from_pdf(file_path)
            if not pages:
                raise ValueError("No text extracted from PDF")
            
            # 5. Chunk documents
            chunks = process_document_chunks(pages)
            
            # 6. DUAL INGESTION
            # 6a. Insert to Postgres (triggers FTS indexing)
            self.doc_chunks.insert_chunks(file_id, chunks)
            
            # 6b. Generate embeddings (GPU batch processing)
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed_batch(texts)
            
            # 6c. Insert to Milvus
            self.vector_store.insert_vectors(
                vectors=embeddings,
                texts=texts,
                file_ids=[file_id] * len(chunks),
                page_numbers=[c["page"] for c in chunks]
            )
            
            # 7. Update metadata
            self.file_registry.update_status(file_id, "COMPLETED")
            
            logger.info(f"✅ Ingestion completed: {filename} ({len(chunks)} chunks)")
            return {
                "status": "success",
                "file_id": file_id,
                "chunks": len(chunks),
                "pages": len(pages)
            }
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            if 'file_id' in locals():
                self.file_registry.update_status(file_id, "FAILED")
            return {"status": "failed", "error": str(e)}
    
    def process_pending_files(self, data_path: str):
        """Auto-process pending files on startup"""
        pending_files = self.file_registry.list_files()
        pending = [f for f in pending_files if f['status'] in ('PENDING', 'PROCESSING')]
        
        if not pending:
            logger.info("No pending files to process")
            return
        
        logger.info(f"🔄 Found {len(pending)} pending files. Starting ingestion...")
        
        for file_record in pending:
            file_path = Path(data_path) / file_record['filename']
            if file_path.exists():
                self.ingest_file(str(file_path), file_record['filename'])
            else:
                logger.warning(f"File not found: {file_path}")
                self.file_registry.update_status(file_record['id'], "FAILED")