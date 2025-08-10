# Import Azure OpenAI embeddings class for generating vector embeddings
from langchain_openai import AzureOpenAIEmbeddings

# Import TextLoader to load text documents from files
from langchain_community.document_loaders import TextLoader

# Import text splitter to break documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import Chroma vector database for storing and searching embeddings
from langchain_chroma import Chroma

# Import os module to access environment variables
import os

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables from .env file into the current environment
load_dotenv()

# Set your Azure OpenAI credentials
# Get the API key from environment variable for authentication
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
# Get the Azure OpenAI endpoint URL from environment variable
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
# Get the API version to use for Azure OpenAI service
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
# Get the specific deployment name for the embedding model
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_FOR_EMBEDDING"]

# Initialize the Azure OpenAI Embeddings
# Create an embeddings client with Azure OpenAI configuration
llm = AzureOpenAIEmbeddings(
    azure_deployment=deployment,  # Specify which deployment to use
    api_version=api_version,  # Set the API version
    azure_endpoint=endpoint,  # Set the Azure endpoint URL    api_key=subscription_key,     # Provide authentication key
)

# Load the job listings document from a text file
document = TextLoader("job_listings.txt").load()
# Create a text splitter to break the document into smaller chunks (200 chars with 10 char overlap)
text_spiltter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
# Split the loaded document into manageable chunks for processing
chunks = text_spiltter.split_documents(document)
# Create a Chroma vector database from the document chunks using embeddings
vector_db = Chroma.from_documents(chunks, llm)

# Convert the vector database to a retriever interface for easier querying
retriever = vector_db.as_retriever()

# Get user input for the search query
query = input("Enter the query: ")

# Use the retriever to find relevant documents based on the query
docs = retriever.invoke(query)

# Iterate through the retrieved relevant documents
for doc in docs:
    # Print the content of each relevant document
    print(doc.page_content)
