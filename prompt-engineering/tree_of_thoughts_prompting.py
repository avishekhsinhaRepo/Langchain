from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

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
)

prompt = ChatPromptTemplate.from_messages([{"role": "user", "content": "{query}"}])

chain = prompt | llm
query = """ You are a health insurance advisor evaluating the best plan for a patient.
            Question: Should a patient with chronic diabetes and hypertension be offered a standard health plan or a specialized chronic care plan?

            Think in multiple ways (Tree of Thought):
            1. Think based on cost-effectiveness.
            2. Think based on patient health outcomes and care coordination.
            3. Think based on long-term insurance risk and sustainability.

            Evaluate each path and provide a final recommendation with reasoning.
            Display a tree-structure with proper blocks to show the paths.
        """
response = chain.invoke({"query": query})
print(response.content)
