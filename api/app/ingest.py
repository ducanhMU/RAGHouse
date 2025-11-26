# file: api/app/ingest.py

import os
import logging
import hashlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_milvus import Milvus
from sqlalchemy.orm import Session
from app.database import SessionLocal, FileRegistry

# --- CONFIG ---
MILVUS_URI = f"http://{os.getenv('MILVUS_HOST', 'milvus')}:{os.getenv('MILVUS_PORT', '19530')}"
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_collection")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBEDDING_MODEL = "nomic-embed-text" # Using a dedicated embedding model

logger = logging.getLogger(__name__)

def calculate_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_vector_store():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    return Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        drop_old=False,
        auto_id=True
    )

def process_file_task(file_id: str):
    """
    Background Task: 
    1. Fetch file info from DB.
    2. Parse -> Chunk -> Embed -> Ingest to Milvus.
    3. Update Status.
    """
    db = SessionLocal()
    file_record = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
    
    if not file_record:
        logger.error(f"File ID {file_id} not found.")
        db.close()
        return

    try:
        logger.info(f"Processing file: {file_record.filename}")
        file_record.status = "PROCESSING"
        db.commit()

        # 1. Load
        if file_record.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_record.file_path)
        else:
            loader = TextLoader(file_record.file_path)
        docs = loader.load()

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # 3. Add Metadata
        for chunk in chunks:
            chunk.metadata["file_id"] = str(file_record.id)
            chunk.metadata["filename"] = file_record.filename
            chunk.metadata["source_hash"] = file_record.file_hash

        # 4. Ingest
        if chunks:
            vector_db = get_vector_store()
            vector_db.add_documents(chunks)

        file_record.status = "COMPLETED"
        logger.info(f"Finished processing {file_record.filename}. Ingested {len(chunks)} chunks.")

    except Exception as e:
        logger.error(f"Failed to process {file_record.filename}: {e}")
        file_record.status = "FAILED"
        file_record.error_log = str(e)
    
    finally:
        db.commit()
        db.close()

def resume_stuck_files():
    """Run on startup to reset any files stuck in PROCESSING state due to crash."""
    db = SessionLocal()
    stuck_files = db.query(FileRegistry).filter(FileRegistry.status == "PROCESSING").all()
    if stuck_files:
        logger.info(f"Found {len(stuck_files)} stuck files. Resetting to PENDING.")
        for f in stuck_files:
            f.status = "PENDING"
            # In a real production system, we would re-queue these tasks here.
        db.commit()
    db.close()