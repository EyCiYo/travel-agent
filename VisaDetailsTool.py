from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel, Field
import os
from dotenv import load_dotenv
from typing import Type
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
apikey = os.getenv("openai_key")


class RAGSearchInput(BaseModel):
    query: str = Field(
        description="query about visa information from the user")


class RagSearchTool(BaseTool):
    name: str = "rag_search_tool"
    description: str = "to search te vector databse for visa related enquiries. If no details are found, please ask the agent to use web search to find relevant info"
    args_schema: Type[BaseTool] = RAGSearchInput

    def _run(self, query: str):
        current_script_path = os.path.abspath(__file__)
        script_directory = os.path.dirname(current_script_path)
        db_folder_path = os.path.join(script_directory, 'db', 'vector_store')
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=apikey)
        db = Chroma(persist_directory=db_folder_path,
                    embedding_function=embedding)

        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3, "score_threshold": 0.4
            }
        )
        relevant_docs = retriever.invoke(query)
        return relevant_docs


class VisaOnlineSearch(BaseTool):
    name: str = "online_search_tool"
    description: str = "to search for visa related information online if RAG fails to find info"
    args_schema: Type[BaseModel] = RAGSearchInput

    def _run(self, query: str):
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)
        return f"Search results for {query} are\n\n {results}"
