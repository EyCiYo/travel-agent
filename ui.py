import os
from dotenv import load_dotenv
from langchain import hub
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai.chat_models import ChatOpenAI
from SearchTool import AttractionSearchTool
from InfoExtractTool import InfoExtractorTool
from LocationTool import LocationTool
from FlightSearchTool import AirportFindTool, FlightSearchTool
from VisaDetailsTool import RagSearchTool, VisaOnlineSearch
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from datetime import datetime
from queue import Queue, Empty


load_dotenv()
api_key = os.getenv("openai_key")
streaming_handler = StreamingStdOutCallbackHandler()

model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key,
                   streaming=True)
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="output"
)


def date_time(inp):
    return datetime.now()


def stream_agent_response(agent_executor, query):

    memory.chat_memory.add_message(HumanMessage(content=query))

    print("AI: ", flush=True)
    response_stream = agent_executor.stream({"input": query})
    full_stream = ""
    for response in response_stream:
        if isinstance(response, dict) and "output" in response:
            print("@", response)
            chunk = response["output"]
            full_stream += chunk
            yield chunk
        else:
            print("bad")
    memory.chat_memory.add_message(
        AIMessage(content=full_stream))


tools = [
    InfoExtractorTool(),
    AttractionSearchTool(),
    LocationTool(),
    FlightSearchTool(),
    AirportFindTool(),
    RagSearchTool(),
    VisaOnlineSearch(),
    Tool(
        name="datetime_tool",
        func=date_time,
        description="to get the date and time of the current moment"
    )
]


def travel_agent():
    initial_msg = "You are a travel who helps users plan trips by providing different options according to user input. If you dont know any answers, please ask for clarifications."
    memory.chat_memory.add_message(SystemMessage(content=initial_msg))
    prompt = hub.pull("hwchase17/openai-tools-agent")

    agent = create_tool_calling_agent(
        llm=model,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        handle_parsing_errors=True,
        verbose=True
    )
    return agent_executor
