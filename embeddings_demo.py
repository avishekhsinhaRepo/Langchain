from langchain_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Set your Azure OpenAI credentials
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_FOR_EMBEDDING"]

# Initialize the Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Example usage
try:
    text = input("Enter the text to embed: ")
    response = embeddings.embed_query(text)
    print(f"Embedding vector length: {len(response)}")
    print(f"First 5 dimensions: {response[:5]}")
except Exception as e:
    print(f"Error generating embeddings: {e}")
