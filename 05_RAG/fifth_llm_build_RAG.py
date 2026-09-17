import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader, PyMuPDF4LLMParser
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#---Environment---
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


INDEX_NAME = "rag-learning-september2026"
EMBEDDING_MODEL = "nomic-embed-text:latest"
REBUILD_INDEX = True
NAMESPACE = "academic-pdf-test"
pinecone_key = os.environ["PINECONE_API_KEY"]

#---DocumentLoading---
PDF_DIR = "../99_AuxiliarySource"

documents = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path=PDF_DIR,
        glob="*.pdf",
    ),
    blob_parser=PyMuPDF4LLMParser(),
).load()

#---SmartChunking---

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    add_start_index=True,
    separators=["\n\n", "\n", ". ", " ", ""], # paragraphs > lines > sentences > words > characters
)
doc_chunks = char_splitter.split_documents(documents)

if not documents:
    raise RuntimeError(
        f"No PDF text was loaded from: {Path(PDF_DIR).resolve()}"
    )

if not doc_chunks:
    raise RuntimeError(
        "No chunks were created. Check PDF parsing and splitter settings."
    )

'''
#---Quality Check---
chunk_lengths = [len(chunk.page_content) for chunk in doc_chunks]

print(f"Loaded page Documents: {len(documents)}")
print(f"Total chunks: {len(doc_chunks)}")
print(f"Shortest chunk: {min(chunk_lengths)} characters")
print(f"Longest chunk: {max(chunk_lengths)} characters")
print(f"Average chunk: {sum(chunk_lengths) / len(chunk_lengths):.1f} characters")

short_chunks = [
    chunk for chunk in doc_chunks
    if len(chunk.page_content.strip()) < 100
]

print(f"Very short chunks (<100 characters): {len(short_chunks)}")

print("\n--- FIRST CHUNK METADATA, CONTENT, LENGTH ---")
print(doc_chunks[0].metadata)
#print(doc_chunks[0].page_content[:800])
#print(len(doc_chunks[0].page_content))
'''

#---Create Embeddings---

if not pinecone_key:
    raise RuntimeError(
        "PINECONE_API_KEY was not found. "
    )

try:
    pc = Pinecone(api_key=pinecone_key)
    indexes = pc.list_indexes()  # call the method
    # In the new SDK, .names is an attribute (list-like), not a method
    names = list(indexes.names())
    # print(f"✅ Pinecone is working. Found {len(names)} index(es): {names}")
    #print(pc.has_index(names[0]))
    #print(names)
    #print(list(indexes.names())#)

    if INDEX_NAME not in names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
            deletion_protection="disabled",
        )

        print(f"✅ Index '{INDEX_NAME}' created.")
        print("Waiting for Pinecone index to become ready...")

        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

        print("✅ Pinecone index is ready.")

    else:
        print(f"ℹ️ Index '{INDEX_NAME}' already exists.")

    ollama_embedder = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://127.0.0.1:11434",
    )

    print("Testing Ollama embedding connection...")

    test_vector = ollama_embedder.embed_query(
        "This is a local embedding connection test."
    )

    print(f"✅ Ollama embedding works: {len(test_vector)} dimensions")

    if len(test_vector) != 768:
        raise ValueError(
            f"Expected a 768-dimensional embedding, "
            f"but got {len(test_vector)}."
        )

    index_description = pc.describe_index(INDEX_NAME)

    print("\n--- Pinecone index description ---")
    print(index_description)

    print("\n--- Pinecone index host ---")
    print(index_description.host)

    index_host = pc.describe_index(INDEX_NAME).host # !!FULL HOSTNAME

    #vector_index = pc.Index(INDEX_NAME)
    vector_index = pc.Index(host=index_host)
    vector_store = PineconeVectorStore(
        index=vector_index,
        embedding=ollama_embedder,
        namespace=NAMESPACE,
    )

    if REBUILD_INDEX:
        stats = vector_index.describe_index_stats()
        existing_namespaces = stats.get("namespaces", {})

        if NAMESPACE in existing_namespaces:
            print(f"Clearing namespace '{NAMESPACE}'...")

            vector_index.delete(
                delete_all=True,
                namespace=NAMESPACE,
            )

            print("Namespace cleared.")

        else:
            print(
                f"Namespace '{NAMESPACE}' does not exist yet. "
                "First upload will create it."
            )

    vector_store.add_documents(
        documents=doc_chunks,
    )

    print(f"Indexed {len(doc_chunks)} chunks into '{NAMESPACE}'.")

    # --- Similarity Search ---
    results_simple = vector_store.similarity_search(
        "What are main problems of Human Technology Interaction?",
        k=1
    )
    print(results_simple)


except Exception as error:
    print(f"❌ Pinecone/Ollama run failed: {error}")




