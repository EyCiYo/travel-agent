import os
from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel, Field
from dotenv import load_dotenv
from typing import Type
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("openai_key")


class SimpleSearchInput(BaseModel):
    query: str = Field(description="should be a search query")


class TripData(BaseModel):
    destination: str = Field("NA", description="destination of user visit")
    duration: str = Field("NA", description="duration of visit")
    start_date: str = Field("NA", description="start date of visit")
    end_date: str = Field("NA", description="end date of visit")
    budget: str = Field("NA", description="budget requirement of the user")


class InfoExtractorTool(BaseTool):
    name: str = "info_extractor"
    description: str = "extract info from user query and return it in specified format"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str) -> TripData:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                           api_key=api_key)
        struct_llm = model.with_structured_output(TripData)
        prompt_template = ChatPromptTemplate([
            ("system", "You just need to extract info and provide it in structred form. The details to be extracted are DESTINATION, DURATION , START_DATE and END_DATE, BUDGET and output them in JSON format. Donot answer the user question. If any details are missing or ambiguos, indiate them as NA"),
            ("user", "{input}")
        ])
        extraction_chain = prompt_template | struct_llm
        result = extraction_chain.invoke({"input": query})
        return result
