import os
import json

from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq # langchain wrapper of groq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, ValidationError # Data validation & Serialization
from torch.fx.experimental.unification.multipledispatch.conflict import consistent


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


creative_llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=1.8, # sampling parameter to adjust randomness
)

consistent_llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.0, # sampling parameter to adjust randomness
)


class AnswerSchema(BaseModel):
    setup: str = Field(..., description="The Start Line of the sentence")
    punchline: str = Field(..., description="The End of the sentence")

pydantic_parser = JsonOutputParser(pydantic_object=AnswerSchema)

# Prompt template

pydantic_query_template = PromptTemplate(
    template="Assume that you are a comedian. Follow the JSON instructions.\n"
        "Answers should be maximum 100 words total.\n"
        "Output a JSON object with these keys:\n"
        "{format_instructions}\n"
        "Question: {user_query}\n",
    input_variables=["user_query"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)

#pydantic_chain = pydantic_query_template | creative_llm | pydantic_parser
pydantic_chain = pydantic_query_template | consistent_llm | pydantic_parser

try:
    pydantic_query_result = pydantic_chain.invoke({"user_query": "Tell me a joke"})
    # convert dict -Y AnswerSchema instance
    answer_obj = AnswerSchema(**pydantic_query_result)
    # AnswerSchema(setup=pydantic_query_result["setup"], pydantic_query_result["punchline"])

    print("Pydantic parsed result (typed):", answer_obj)
    print("setup:", answer_obj.setup)
    print("punchline:", answer_obj.punchline)

except ValidationError as e:
    print("Validation failed: ", e)
    raw = pydantic_chain.invoke({"user_query": "Tell me a joke"})
    print("Raw output for debugging:", raw)


