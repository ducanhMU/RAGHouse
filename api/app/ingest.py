# file: api/app/ingest.py

import os
import logging
import hashlib
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_milvus import Milvus
from pymilvus import (
    Collection, CollectionSchema, FieldSchema, DataType, 
    connections, utility, MilvusClient
)
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.database import SessionLocal, FileRegistry, FileStatus

# --- CONFIG ---
MILVUS_HOST = os.getenv('MILVUS_HOST', 'milvus')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_collection_v2_hnsw")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
DATA_PATH = os.getenv("DATA_PATH", "./data")

# GPU OPTIMIZATION
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))  # GPU batch
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

logger = logging.getLogger(__name__)

# Global caches
_vector_store_cache = None
_embeddings_cache = None
_milvus_client = None

def calculate_md5(file_path: str) -> str:
    """Calculate MD5 hash efficiently."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def create_collection_with_hnsw():
    """
    OPTIMIZED: Create Milvus collection with HNSW index
    HNSW is 10x faster than IVF_FLAT for retrieval
    """
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                timeout=10
            )
            
            if utility.has_collection(COLLECTION_NAME):
                logger.info(f"✅ Collection '{COLLECTION_NAME}' exists")
                connections.disconnect("default")
                return True
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # nomic-embed-text
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="page", dtype=DataType.INT64),
                FieldSchema(name="source_hash", dtype=DataType.VARCHAR, max_length=32),
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="RAG V2 with HNSW Index",
                enable_dynamic_field=True
            )
            
            collection = Collection(
                name=COLLECTION_NAME,
                schema=schema,
                using='default'
            )
            
            # === HNSW INDEX (PRODUCTION GRADE) ===
            index_params = {
                "metric_type": "L2",
                "index_type": "HNSW",  # Changed from IVF_FLAT
                "params": {
                    "M": 16,              # Number of connections (8-64, 16 is balanced)
                    "efConstruction": 200 # Build quality (100-500, higher = better quality)
                }
            }
            
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"✅ Collection created with HNSW index (Production)")
            connections.disconnect("default")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_delay)
            else:
                logger.error(f"❌ Collection creation failed")
                return False
        finally:
            try:
                connections.disconnect("default")
            except:
                pass
    
    return False

def get_embeddings():
    """Get cached embeddings model."""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL
        )
    return _embeddings_cache

def get_vector_store():
    """Get cached vector store with LangChain compatibility."""
    global _vector_store_cache
    if _vector_store_cache is None:
        _vector_store_cache = Milvus(
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
            drop_old=False,
            auto_id=True,
            text_field="text"  # CRITICAL: Map to our schema field
        )
    return _vector_store_cache

def get_milvus_client():
    """Get native PyMilvus client for direct operations."""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=MILVUS_URI)
    return _milvus_client

def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metadata for dynamic schema."""
    standard_fields = {'file_id', 'filename', 'page', 'source_hash'}
    
    normalized = {}
    dynamic_fields = {}
    
    for key, value in metadata.items():
        if key in standard_fields:
            normalized[key] = value
        else:
            dynamic_fields[key] = value
    
    if dynamic_fields:
        normalized['$meta'] = dynamic_fields
    
    return normalized

def embed_documents_gpu_optimized(texts: List[str], embeddings) -> List[List[float]]:
    """
    GPU-OPTIMIZED: Batch embedding with size 32
    Leverages RTX A4000's 16GB VRAM and parallel processing
    """
    try:
        all_embeddings = []
        
        # Process in large batches for GPU efficiency
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            logger.info(f"🚀 GPU Batch {i//EMBEDDING_BATCH_SIZE + 1}/{(len(texts)-1)//EMBEDDING_BATCH_SIZE + 1} (size={len(batch)})")
            
            # GPU processes this batch in parallel
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        
        logger.info(f"✅ Embedded {len(texts)} chunks on GPU")
        return all_embeddings
        
    except Exception as e:
        logger.error(f"❌ GPU embedding failed: {e}")
        raise

def insert_to_milvus_native(
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict]
):
    """
    NATIVE PYMILVUS: Direct insert for maximum control
    Bypasses LangChain wrapper for precise schema mapping
    """
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        
        collection = Collection(COLLECTION_NAME)
        
        # Prepare data for insertion
        entities = [
            embeddings,  # embedding field
            texts,       # text field
            [m.get('file_id', '') for m in metadatas],
            [m.get('filename', '') for m in metadatas],
            [m.get('page', 0) for m in metadatas],
            [m.get('source_hash', '') for m in metadatas],
        ]
        
        # Insert
        result = collection.insert(entities)
        
        logger.info(f"✅ Inserted {len(texts)} entities (IDs: {result.primary_keys[:3]}...)")
        
        # Flush to ensure data is persisted
        collection.flush()
        
        connections.disconnect("default")
        return result
        
    except Exception as e:
        logger.error(f"❌ Native insert failed: {e}")
        raise
    finally:
        try:
            connections.disconnect("default")
        except:
            pass

def process_file_task(file_id: str):
    """
    ULTIMATE OPTIMIZED: GPU + Native PyMilvus + HNSW
    """
    db = SessionLocal()
    file_record = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
    
    if not file_record:
        logger.error(f"❌ File ID {file_id} not found")
        db.close()
        return
    
    try:
        logger.info(f"🚀 Processing: {file_record.filename}")
        file_record.status = FileStatus.PROCESSING
        db.commit()
        
        # === STEP 1: Load ===
        if file_record.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_record.file_path)
        else:
            loader = TextLoader(file_record.file_path, encoding='utf-8')
        
        docs = loader.load()
        logger.info(f"📄 Loaded {len(docs)} pages")
        
        # === STEP 2: Chunk ===
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"✂️ Split into {len(chunks)} chunks")
        
        if not chunks:
            file_record.status = FileStatus.COMPLETED
            file_record.error_log = "No content"
            file_record.meta_info = {"pages": 0, "chunks": 0}
            db.commit()
            db.close()
            return
        
        # === STEP 3: Normalize Metadata ===
        for chunk in chunks:
            chunk.metadata["file_id"] = str(file_record.id)
            chunk.metadata["filename"] = file_record.filename
            chunk.metadata["source_hash"] = file_record.file_hash
            
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = 0
            
            chunk.metadata = normalize_metadata(chunk.metadata)
        
        # === STEP 4: GPU-OPTIMIZED EMBEDDING ===
        embeddings_model = get_embeddings()
        texts = [chunk.page_content for chunk in chunks]
        
        logger.info(f"🚀 Starting GPU-optimized embedding (batch={EMBEDDING_BATCH_SIZE})...")
        embedded_vectors = embed_documents_gpu_optimized(texts, embeddings_model)
        
        # === STEP 5: NATIVE PYMILVUS INSERT ===
        metadatas = [chunk.metadata for chunk in chunks]
        
        logger.info(f"📥 Inserting to Milvus using native API...")
        insert_to_milvus_native(texts, embedded_vectors, metadatas)
        
        # === STEP 6: Update Metadata ===
        file_record.status = FileStatus.COMPLETED
        file_record.meta_info = {
            "pages": len(docs),
            "chunks": len(chunks),
            "model": EMBEDDING_MODEL,
            "index_type": "HNSW",
            "chunk_size": 1000,
            "batch_size": EMBEDDING_BATCH_SIZE
        }
        logger.info(f"🎉 Completed: {file_record.filename}")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        file_record.status = FileStatus.FAILED
        file_record.error_log = str(e)
        
    finally:
        db.commit()
        db.close()

def scan_and_ingest_directory_parallel():
    """
    PARALLEL DIRECTORY SCAN
    """
    if not os.path.exists(DATA_PATH):
        logger.warning(f"⚠️ Data path {DATA_PATH} does not exist")
        return
    
    db = SessionLocal()
    try:
        files = list(Path(DATA_PATH).glob("*.*"))
        logger.info(f"📁 Found {len(files)} files")
        
        valid_files = [f for f in files if f.suffix.lower() in ['.pdf', '.txt', '.md']]
        
        if not valid_files:
            logger.info("✅ No files to process")
            return
        
        # Register all files
        file_ids_to_process = []
        
        for file_path in valid_files:
            try:
                file_hash = calculate_md5(str(file_path))
                
                existing = db.query(FileRegistry).filter(
                    FileRegistry.file_hash == file_hash
                ).first()
                
                if existing:
                    continue
                
                new_file = FileRegistry(
                    file_hash=file_hash,
                    filename=file_path.name,
                    file_path=str(file_path),
                    file_size=file_path.stat().st_size,
                    status=FileStatus.PENDING
                )
                
                try:
                    db.add(new_file)
                    db.commit()
                    db.refresh(new_file)
                    
                    file_ids_to_process.append(str(new_file.id))
                    logger.info(f"📝 Registered: {file_path.name}")
                    
                except IntegrityError:
                    db.rollback()
                    
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                db.rollback()
        
        # Process in parallel
        if file_ids_to_process:
            logger.info(f"🚀 Processing {len(file_ids_to_process)} files in parallel (workers={MAX_WORKERS})...")
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_file_task, file_id): file_id 
                    for file_id in file_ids_to_process
                }
                
                for future in as_completed(futures):
                    file_id = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"❌ Failed for {file_id}: {e}")
            
            logger.info(f"✅ Parallel processing complete")
        else:
            logger.info("✅ No new files")
            
    finally:
        db.close()

def resume_stuck_files():
    """Reset and retry stuck files."""
    db = SessionLocal()
    try:
        stuck_files = db.query(FileRegistry).filter(
            FileRegistry.status == FileStatus.PROCESSING
        ).all()
        
        if stuck_files:
            logger.info(f"🔄 Found {len(stuck_files)} stuck files")
            
            for f in stuck_files:
                f.status = FileStatus.PENDING
            db.commit()
            
            file_ids = [str(f.id) for f in stuck_files]
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_file_task, file_id): file_id 
                    for file_id in file_ids
                }
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"❌ Retry failed: {e}")
        else:
            logger.info("✅ No stuck files")
            
    finally:
        db.close()

def get_collection_stats() -> Dict:
    """Get Milvus statistics."""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        
        if not utility.has_collection(COLLECTION_NAME):
            return {"error": "Collection does not exist"}
        
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        stats = {
            "name": COLLECTION_NAME,
            "num_entities": collection.num_entities,
            "index_type": "HNSW",
            "schema": {
                "fields": [f.name for f in collection.schema.fields],
                "dynamic_enabled": collection.schema.enable_dynamic_field
            }
        }
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            connections.disconnect("default")
        except:
            pass

def initialize_ingest_system():
    """
    MAIN STARTUP
    """
    logger.info("🚀 Initializing ULTIMATE ingest system...")
    logger.info(f"📊 Config: GPU batch={EMBEDDING_BATCH_SIZE}, workers={MAX_WORKERS}, model={EMBEDDING_MODEL}")
    
    # Create HNSW collection
    if not create_collection_with_hnsw():
        return False
    
    # Resume stuck files
    resume_stuck_files()
    
    # Scan directory
    scan_and_ingest_directory_parallel()
    
    logger.info("✅ ULTIMATE ingest system initialized")
    return True