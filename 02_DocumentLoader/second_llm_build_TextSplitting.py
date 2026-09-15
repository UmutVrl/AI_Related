import os
import wikipediaapi
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader


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

#---pdf---
pdf_documents = PyMuPDF4LLMLoader("Bandhu+24_Elsevier_MotivationTheories.pdf").load()
print(pdf_documents[0].metadata)

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks_from_docs = splitter.split_documents(pdf_documents) # list of document objects

print("Total chunks from docs:", len(chunks_from_docs))
print("Sample chunk preview: \n", chunks_from_docs[3].page_content[:400])
print("Sample chunk metadata: \n", getattr(chunks_from_docs[0], "metadata", {}))

char_splitter =CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=50)
doc_chunks = char_splitter.split_documents(pdf_documents)

print("Total chunks from docs:", len(doc_chunks))
print("Sample chunk preview: \n", doc_chunks[3].page_content[:400])
print("Sample chunk metadata: \n", getattr(doc_chunks[0], "metadata", {}))

