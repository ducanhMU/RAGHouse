from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(model="embeddinggemma", base_url="http://localhost:11434")

vector = emb.embed_query("hello world")
print(len(vector))
