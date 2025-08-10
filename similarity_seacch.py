from langchain_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
import numpy as np

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
    text1 = input("Enter the text1: ")
    text2 = input("Enter the text2: ")
    # Get the embedding of the text
    response1 = embeddings.embed_query(text1)
    response2 = embeddings.embed_query(text2)
    # Calculate the cosine similarity
    similarity = np.dot(response1, response2)
    print(similarity)

except Exception as e:
    print(f"Error generating embeddings: {e}")
