from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.tools.render import render_text_description
from langchain import hub
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from typing import Union
from langchain.agents.agent import AgentAction, AgentFinish


load_dotenv()

# Set your Azure OpenAI credentials
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
    temperature=0,
    stop=["\nObservation", "Observation"],
)


@tool
def get_text_length(text: str) -> int:
    """
    Get the length of the text in tokens.
    """
    text = text.strip("\n").strip('"')
    return len(text)


tools = [get_text_length]
template = """
        Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:

"""

prompt = PromptTemplate.from_template(template=template).partial(
    tools=render_text_description(tools), tool_names=",".join(t.name for t in tools)
)

agent = {"input": lambda input_dict: input_dict["input"]} | prompt | llm | ReActSingleInputOutputParser()

agent_step:Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of the text 'Hello, World!'?"})

print(agent_step)

def find_tool_by_name(tools, tool_name):
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")

if isinstance(agent_step, AgentAction):
    tool_name = agent_step.tool
    tool_input = agent_step.tool_input
    tool_to_use = find_tool_by_name(tools, tool_name)
    observation = tool_to_use.func(str(tool_input))
    print(f"Observation: {observation}")