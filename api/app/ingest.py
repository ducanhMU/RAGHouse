import os
import logging
import sys
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_milvus import Milvus

# --- CONFIGURATION FROM ENV ---
DATA_PATH = os.getenv("DATA_PATH", "./data")
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_demo")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")

# Logging Setup
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ingest_docs():
    """
    Loads documents from the data directory, splits them, and ingests them into Milvus.
    This function appends data to the existing collection; it does not overwrite.
    """
    
    # 1. Load Data
    logger.info(f"Checking for documents in: {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True
    )
    docs = loader.load()
    
    if not docs:
        logger.warning("No documents found in the data directory.")
        return
    logger.info(f"Loaded {len(docs)} documents.")

    # 2. Chunking
    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Total chunks created: {len(chunks)}")

    # 3. Embeddings Setup
    logger.info(f"Initializing Embedding Model: {EMBEDDING_MODEL_NAME}")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=OLLAMA_BASE_URL
    )
    
    # 4. Initialize Milvus Connection
    milvus_uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    logger.info(f"Connecting to Milvus at {milvus_uri}...")

    # We initialize the Milvus object with drop_old=False to ensure we APPEND data.
    # If the collection does not exist, it will be created with the specified index_params.
    vector_db = Milvus(
        embedding_function=embeddings,
        collection_name=MILVUS_COLLECTION_NAME,
        connection_args={"uri": milvus_uri},
        drop_old=False,  # CRITICAL: Ensures we do not delete existing data
        auto_id=True,
        consistency_level="Strong",
        index_params={
            "index_type": "HNSW", 
            "metric_type": "L2", 
            "params": {"M": 8, "efConstruction": 64}
        }
    )

    # 5. Ingest with Batching
    BATCH_SIZE = 64
    total_chunks = len(chunks)
    
    try:
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            current_batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"Ingesting Batch [{current_batch_num}/{total_batches}] - {len(batch)} chunks...")
            
            # Since vector_db is already initialized, we just add documents
            vector_db.add_documents(batch)

        logger.info("Ingestion process completed successfully.")
        
    except Exception as e:
        logger.error(f"Failed to ingest into Milvus: {e}")

if __name__ == "__main__":
    ingest_docs()