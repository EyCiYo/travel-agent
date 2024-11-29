from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel, Field
import os
from dotenv import load_dotenv
from typing import Type

load_dotenv()


class SimpleSearchInput(BaseModel):
    query: str = Field(description="should be a search query")


class AttractionSearchTool(BaseTool):
    name: str = "attarctions_search"
    description: str = "to search attractions in a provided area"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str) -> str:
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)
        return f"Search results for {query} are\n\n {results}"
