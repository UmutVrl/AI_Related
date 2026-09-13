import os
from pathlib import Path

import chromadb
import langchain
import pydantic
from sentence_transformers import SentenceTransformer
from huggingface_hub import whoami
from dotenv import load_dotenv

# Ensure .env is loaded from the project root
project_root = Path(__file__).resolve().parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# ---------- LangSmith ----------
langsmith_key = os.getenv("LANGCHAIN_API_KEY")
print(f"LANGCHAIN_API_KEY present: {bool(langsmith_key)}")

# ---------- Hugging Face ----------
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    try:
        info = whoami()
        print(f"✅ Hugging Face token is working. Logged in as: {info['name']}")
    except Exception as e:
        print(f"❌ Hugging Face token check failed: {e}")
else:
    print("ℹ️  No HF_TOKEN in environment; skipping Hugging Face authentication check.")

print(f"LangChain: {langchain.__version__}")
print(f"Pydantic: {pydantic.__version__}")
print(f"ChromaDB: {chromadb.__version__}")

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("RAG retrieves relevant context before an LLM answers.")
print(f"Embedding dimensions: {len(embedding)}")

# ---------- Groq ----------
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    print("\n--- Testing Groq ---")
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1"
        )

        candidate_models = [
            "openai/gpt-oss-120b", # https://console.groq.com/playground
            "llama-3.1-70b-versatile", # no longer supported by Groq
            "llama-3.1-8b-instant", # no longer supported by Groq
            "llama3-70b-8192", # no longer supported by Groq
            "llama3-8b-8192", # no longer supported by Groq
        ]

        for model in candidate_models:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a concise assistant."},
                        {"role": "user", "content": "Say hello in one sentence."}
                    ],
                    temperature=0.0,
                    max_tokens=40
                )
                text = response.choices[0].message.content.strip()
                print(f"✅ Groq is working with model '{model}'. Response: {text}")
                break
            except Exception:
                continue
        else:
            print("❌ Groq test failed: none of the candidate models worked.")
    except Exception as e:
        print(f"❌ Groq test failed: {e}")
else:
    print("\nℹ️  No GROQ_API_KEY in environment; skipping Groq test.")

# ---------- Tavily ----------
tavily_key = os.getenv("TAVILY_API_KEY")
if tavily_key:
    print("\n--- Testing Tavily ---")
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=tavily_key)
        result = client.search("What is Tavily?", max_results=1)
        if result and "results" in result and len(result["results"]) > 0:
            snippet = result["results"][0].get("content", "")[:120]
            print(f"✅ Tavily is working. Sample result: {snippet}...")
        else:
            print("✅ Tavily is working (no results, but no error).")
    except Exception as e:
        print(f"❌ Tavily test failed: {e}")
else:
    print("\nℹ️  No TAVILY_API_KEY in environment; skipping Tavily test.")

# ---------- Pinecone ----------
pinecone_key = os.getenv("PINECONE_API_KEY")
if pinecone_key:
    print("\n--- Testing Pinecone ---")
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=pinecone_key)

        indexes = pc.list_indexes() # call the method

        # In the new SDK, .names is an attribute (list-like), not a method
        names = list(indexes.names())

        print(f"✅ Pinecone is working. Found {len(names)} index(es): {names}")
    except Exception as e:
        print(f"❌ Pinecone test failed: {e}")
else:
    print("\nℹ️  No PINECONE_API_KEY in environment; skipping Pinecone test.")

# ---------- Weaviate ----------
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
if weaviate_url:
    print("\n--- Testing Weaviate ---")
    try:
        import weaviate
        from weaviate.classes.init import Auth

        auth = None
        if weaviate_api_key:
            auth = Auth.api_key(weaviate_api_key)

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=auth,
        )

        meta = client.get_meta()
        version = meta.get("version", "unknown")
        print(f"✅ Weaviate is working. Version: {version}")

        client.close()
    except Exception as e:
        print(f"❌ Weaviate test failed: {e}")
else:
    print("\nℹ️  No WEAVIATE_URL in environment; skipping Weaviate test.")

print("\nEnvironment is ready.")
