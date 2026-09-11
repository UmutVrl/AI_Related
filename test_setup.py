import chromadb
import langchain
import pydantic
from sentence_transformers import SentenceTransformer

print(f"LangChain: {langchain.__version__}")
print(f"Pydantic: {pydantic.__version__}")
print(f"ChromaDB: {chromadb.__version__}")

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("RAG retrieves relevant context before an LLM answers.")
print(f"Embedding dimensions: {len(embedding)}")
print("Environment is ready.")