import os
import json

from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq # langchain wrapper of groq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
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

# to automatic retrieval later
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT") # organizing logs
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") # logging

creative_llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=1.8, # sampling parameter to adjust randomness
)

consistent_llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.0, # sampling parameter to adjust randomness
)


# Define the schema
class TravelSuggestion(BaseModel):
    destination: str = Field(description="Name of the travel destination in Germany")
    price_range: str = Field(description="Approximate price range, e.g. '€€', '€€€', or 'budget/mid-range/luxury'")
    things_to_do: list[str] = Field(description="List of 3–5 recommended activities or sights")
    best_time_to_visit: str = Field(description="Best season or months to visit, e.g. 'May–September'")
    short_description: str = Field(description="2–3 sentence description of why this place is worth visiting")


# Create parser
parser = PydanticOutputParser(pydantic_object=TravelSuggestion)

# Prompt template
pydantic_query_template = PromptTemplate(
    template="You are a helpful travel assistant for Germany.\n"
             "You have a good knowledge over camping, especially with Bike and Tent.\n"
             "Suggest a travel destination in Germany based on the user's request.\n"
        "Answers should be concise (maximum 200 words total).\n"
        "Output a JSON object with these keys:\n"
        "{format_instructions}\n"
        "Question: {user_query}\n",
    input_variables=["user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

#pydantic_chain = pydantic_query_template | creative_llm | pydantic_parser
pydantic_chain = pydantic_query_template | consistent_llm | parser

try:
    pydantic_query_result = pydantic_chain.invoke({
        "user_query": "Suggest me a travel destination in Germany for a 3-day trip in spring."
    })

    # pydantic_query_result is already a TravelSuggestion instance
    answer_obj = pydantic_query_result

    print("Pydantic parsed result (typed):", answer_obj)
    print("Destination:", answer_obj.destination)
    print("Price range:", answer_obj.price_range)
    print("Things to do:", answer_obj.things_to_do)
    print("Best time to visit:", answer_obj.best_time_to_visit)
    print("Description:", answer_obj.short_description)

except ValidationError as e:
    print("Validation failed:", e)
    # If you want to debug, re-run without the parser in the chain:
    raw_chain = pydantic_query_template | consistent_llm
    raw = raw_chain.invoke({
        "user_query": "Suggest me a travel destination in Germany for a 3-day trip in spring."
    })
    print("Raw output for debugging:", raw)


