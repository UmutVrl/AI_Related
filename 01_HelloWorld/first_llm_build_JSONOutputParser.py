import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq # langchain wrapper of groq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


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


# Initialisation JSON parser
json_parser = JsonOutputParser()

# Create Prompt Template
query_template = PromptTemplate(
    template="You are a helpful AI bot. Provide the answer in JSON format based on the question.\n"
        "Answer should be less than 100 words total.\n"
        "Output a JSON object with these keys:\n"
        "- name: string – name of the tool\n"
        "- description: string – short description of what it is used for\n"
        "- main_features: array of strings – list of main features\n"
        "{format_instructions}\n"
        "Question: {user_query}\n",
    input_variables=["user_query"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()}
)

# Initialize Groq model
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.0, # sampling parameter to adjust randomness
)

# Build Chain
chain = query_template | llm | json_parser

# Run Chain
response = chain.invoke({"user_query": "Explain what is LangGraph used for?"})
print(response)




