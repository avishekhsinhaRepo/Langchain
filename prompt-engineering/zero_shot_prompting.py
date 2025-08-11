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
prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": "Classify this customer query into one of: Billing, Technical, Sales. Respond ONLY with the category name.",
        },
        {"role": "user", "content": "{query}"},
    ]
)

chain = prompt | llm
response = chain.invoke(
    {
        "query": "My invoice for order #1234 seems incorrect. Can you clarify the charges?"
    }
)


print("The said query belongs to " + response.content + " Section")
