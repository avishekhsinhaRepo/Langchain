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
# without CoT
# query = """ Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
#             Each can has 3 tennis balls. How many tennis balls does he have now?

#         """
# with CoT
query = """ Let's think through this step by step.

            Scenario:
            - Patient has a PPO plan.
            - Procedure: MRI (advanced imaging).
            - Location: Out-of-network imaging center.
            - The plan requires pre-authorization for advanced imaging.

            Step-by-step:
            1. Does the PPO plan require pre-authorization for advanced imaging? 
            2. Is the MRI considered advanced imaging? 
            3. Does it matter that it's out-of-network? 
            4. Who typically initiates it? 
            5. Give overall conclusion.
        """
response = chain.invoke({"query": query})
print(response.content)
