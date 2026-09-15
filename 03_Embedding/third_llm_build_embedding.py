import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_ollama import OllamaEmbeddings


def find_env_file(start_path: Path | None = None) -> Path | None:
    """Search upward from start_path for a .env file."""
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current = start_path
    while current != current.parent:
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        current = current.parent
    return None

project_root = Path(__file__).resolve().parent
env_path = find_env_file(project_root)

if env_path:
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

#---HF Embeddings---
# https://docs.langchain.com/oss/python/integrations/embeddings
hf_embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5", # 512 tokens max
    encode_kwargs={"normalize_embeddings": True},
    query_encode_kwargs={
        "prompt": "Represent this sentence for searching relevant passages: ",
        "normalize_embeddings": True,
    },
)
#print(embeddings)

#---text---

text = "this is a sample text line."
query_text = hf_embedder.embed_query(text)
#print(query_text)
v = np.array(query_text)
#print(len(v))
#print(np.linalg.norm(v))

with open("sample.txt", "r", encoding="utf-8") as fh:
    sample_text = fh.read()

splitter_text = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
text_chunks = splitter_text.create_documents([sample_text])

#print("Total chunks from docs:", len(text_chunks))
#print("Sample chunk preview: \n", text_chunks[3].page_content[:400])
#print("Sample chunk metadata: \n", getattr(text_chunks[0], "metadata", {}))

doc_texts =[doc.page_content for doc in text_chunks]
doc_result = hf_embedder.embed_documents(doc_texts)
#print(len(doc_result), "embeddings created")
#print(doc_result[0][:10])

#---pdf---

pdf_documents = PyMuPDF4LLMLoader("Schitzer+24_ACM_SocialRobotCompanion.pdf").load()
#print(pdf_documents[0].metadata)

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""], # paragraphs > lines > sentences > words > characters
)
doc_chunks = char_splitter.split_documents(pdf_documents)

#print("Total chunks from docs:", len(doc_chunks))
#print("Sample chunk preview: \n", doc_chunks[3].page_content[:400])
#print("Sample chunk metadata: \n", getattr(doc_chunks[0], "metadata", {}))

doc_pdf = [doc.page_content for doc in doc_chunks]
doc_result_pdf = hf_embedder.embed_documents(doc_pdf)
#print(len(doc_result_pdf), "embeddings created")
#print(doc_result_pdf[0][:10])

#---Ollama Embeddings---
#https://ollama.com/download

#ollama_embedder = OllamaEmbeddings(
#    model="llama3.2:3b",
#)

ollama_embedder = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

sample_texts = [
    "Ollama lets you use open models with your coding agents so you can spend less while keeping your data private.",
    "You can use open models in your desktop apps and coding agents, or build them into your application."
]
doc_vectors = ollama_embedder.embed_documents(sample_texts)

print("Number of document vectors:", len(doc_vectors))
print("Sample chunk preview: \n", doc_vectors[1][:10])
print("First vector dimensionality length:", len(doc_vectors[1]))

query_text = "Ollama lets you use open models with your coding agents"
query_vector = ollama_embedder.embed_query(query_text)
print("Query vector length:", len(query_vector))
print("Query vector preview: \n", query_vector[:10])

