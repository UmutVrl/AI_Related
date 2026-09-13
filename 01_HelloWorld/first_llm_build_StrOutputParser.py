import os
from pathlib import Path

from IPython.terminal.shortcuts.auto_suggest import llm_autosuggestion
from dotenv import load_dotenv
from langchain_groq import ChatGroq # langchain wrapper of groq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate # https://reference.langchain.com/python/langchain-core
from langchain_core.output_parsers import StrOutputParser #https://reference.langchain.com/python/langchain-core/output_parsers/string/StrOutputParser
from transformers.models.llama4.processing_llama4 import chat_template


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

# Build a chat template with system + user roles
sys_msg = SystemMessagePromptTemplate.from_template("You are a helpful AI bot. Assume that you are female. Provide the answer based on the question. Answer should less than 100 word")
user_msg = HumanMessagePromptTemplate.from_template("{question_text}")

chat_template = ChatPromptTemplate.from_messages([sys_msg, user_msg])

# Initialize Groq model
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.0, # sampling parameter to adjust randomness
)

# Create a string parser and wire the pipeline
text_only_parser = StrOutputParser()
string_pipeline = chat_template | llm | text_only_parser

# Call the chain
result = string_pipeline.invoke({"question_text": "Do you have a gender?"})
print("RAW parser result:", result)







