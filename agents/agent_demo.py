# Import os module to access environment variables
import os

# Import Azure OpenAI chat model for LLM interactions
from langchain_openai import AzureChatOpenAI

# Import hub to pull pre-built prompts from LangChain Hub
from langchain import hub

# Import agent creation and execution classes for ReAct pattern
from langchain.agents import create_react_agent, AgentExecutor

# Import function to load predefined tools (Wikipedia, DuckDuckGo search)
from langchain_community.agent_toolkits.load_tools import load_tools

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables from .env file into the current environment
load_dotenv()
# Set your Azure OpenAI credentials (single resource for both models)
# Get the API key from environment variable for authentication
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
# Get the Azure OpenAI endpoint URL from environment variable
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
# Get the API version to use for Azure OpenAI service
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
# Get the GPT model deployment name for text generation
gpt_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]  # Your GPT model deployment

# Initialize the Azure OpenAI Chat model for text generation
# Create a chat model instance with Azure OpenAI configuration
llm = AzureChatOpenAI(
    azure_deployment=gpt_deployment,  # Specify which deployment to use
    api_version=api_version,  # Set the API version
    azure_endpoint=endpoint,  # Set the Azure endpoint URL    api_key=subscription_key,         # Provide authentication key
)


# Pull a pre-built ReAct (Reasoning + Acting) prompt template from LangChain Hub
# ReAct prompts help the agent reason through problems step-by-step and take actions
prompt = hub.pull("hwchase17/react")

# Load predefined tools for the agent to use
# "wikipedia": Allows searching Wikipedia for factual information
# "ddg-search": Allows searching DuckDuckGo for current web information
tools = load_tools(["wikipedia", "ddg-search"])
# Create a ReAct agent that can reason and act using the provided tools
agent = create_react_agent(llm, tools, prompt)
# Create an executor to run the agent with verbose output to see reasoning steps
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Get user input for the task they want the agent to perform
task = input("Assign me Task")

# Check if the user provided a task
if task:
    # Execute the agent with the given task
    # The agent will reason through the problem and use tools as needed
    response = agent_executor.invoke({"input": task})
    # Print the final output/answer from the agent
    print(response["output"])
