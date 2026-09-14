import os
import wikipediaapi
import arxiv

from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, TextLoader, WebBaseLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.documents import Document


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

#---text---
documents =  TextLoader("sample.txt", encoding="utf-8").load()
#print(documents[0].page_content)
print(documents[0].metadata)

#---pdf---
pdf_documents = PyMuPDF4LLMLoader("sample.pdf").load()
print(pdf_documents[0].metadata)

#---csv---
csv_docs = CSVLoader("sample_employees.csv", encoding="utf-8").load()
print(csv_docs[0].metadata)

#---web---
web_docs = WebBaseLoader(web_paths=("https://en.wikipedia.org/wiki/Cyrus_the_Great",)).load()
print(web_docs[0].metadata)

#---wiki---
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent="MyLangchainApp/1.0 (myemail@example.com)",
    language="en"
)

page = wiki_wiki.page("Alexander the Great")

wiki_doc = Document(
    page_content=page.summary,
    metadata={
        "title": page.title,
        "url": page.fullurl
    }
)

print(wiki_doc.metadata)

"""
#---arxiv---
client = arxiv.Client(
    page_size=2,
    delay_seconds=10,
    num_retries=10
)

search = arxiv.Search(
    query="uncanny valley",
    max_results=1,
    sort_by=arxiv.SortCriterion.Relevance
)

documents = []

for result in client.results(search):
    documents.append(
        Document(
            page_content=result.summary,
            metadata={
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "published": result.published.isoformat(),
                "arxiv_url": result.entry_id,
                "pdf_url": result.pdf_url,
                "doi": result.doi,
                "categories": result.categories,
            }
        )
    )

for document in documents:
    print(document.metadata["title"])
"""


