import os

from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq # langchain wrapper of groq
from langchain_core.prompts import ChatPromptTemplate # https://reference.langchain.com/python/langchain-core

def find_env_file(start_path: Path | None = None) -> Path | None:
    """Search upward from start_path for a .env file."""
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current = start_path
    while current != current.parent:  # stop at filesystem root
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
    # Fallback: let dotenv search default locations
    load_dotenv()

#print(os.getenv("LANGCHAIN_PROJECT"))

# to automatic retrieval later
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT") # organizing logs
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") # logging

# https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Assume that your age is 100. Provide the answer based on the question in 10  words"),
        ("user", "{input}")
    ]
)
print(prompt)

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.5, # 1.5 near max creativity
)

#response = llm.invoke("Explain what is maximum temperature value for ChatGroq llm models in 100 words")
response = llm.invoke("What are you?")
print(response.content)
#print(response.id)

chain = prompt | llm

response = chain.invoke({"input":"What are you?"})
print(response.content)




