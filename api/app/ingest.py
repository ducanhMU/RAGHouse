import os
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_milvus import Milvus

# --- CONFIGURATION FROM ENV ---
# Tự động lấy từ biến môi trường, nếu không có thì dùng giá trị mặc định bên phải
DATA_PATH = os.getenv("DATA_PATH", "./data")
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_demo")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # 1. Load Data
    logger.info(f"Loading data from: {DATA_PATH}...")
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
        logger.warning("No documents found!")
        return
    logger.info(f"Loaded {len(docs)} documents.")

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Total chunks created: {len(chunks)}")

    # 3. Embeddings
    logger.info(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=OLLAMA_BASE_URL
    )
    
    # 4. Ingest to Milvus with Batching
    milvus_uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    logger.info(f"Ingesting to Milvus at {milvus_uri}...")

    BATCH_SIZE = 64
    total_chunks = len(chunks)
    vector_db = None
    
    try:
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            current_batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"Processing Batch [{current_batch_num}/{total_batches}] - {len(batch)} chunks...")
            
            if i == 0:
                # First batch: Drop old collection and create new HNSW Index
                vector_db = Milvus.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=MILVUS_COLLECTION_NAME,
                    connection_args={"uri": milvus_uri},
                    drop_old=True,
                    consistency_level="Strong", 
                    index_params={
                        "index_type": "HNSW", 
                        "metric_type": "L2", 
                        "params": {"M": 8, "efConstruction": 64}
                    }
                )
            else:
                if vector_db:
                    vector_db.add_documents(batch)

        logger.info("Successfully ingested ALL chunks into Milvus!")
        
    except Exception as e:
        logger.error(f"Failed to ingest into Milvus: {e}")

if __name__ == "__main__":
    main()