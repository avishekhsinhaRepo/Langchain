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
query = """ Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can have 3 tennis balls. 
            How many tennis balls does he have now?
            Answer: Roger has 5 balls initially. 2 cans with 3 each balls means 5 + 2*3 = 11
            Now answer this question:
            John has 5 apples. He buys 2 more crates of apples, and each crate consists of a dozen apple. 
            How many apples does John has now?
        """
response = chain.invoke({"query": query})
print(response.content)
