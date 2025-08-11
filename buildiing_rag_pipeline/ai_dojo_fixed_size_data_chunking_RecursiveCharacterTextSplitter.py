from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

# Set your Azure OpenAI credentials (single resource for both models)
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
embedding_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_FOR_EMBEDDING"]
gpt_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]  # Your GPT model deployment

# Initialize the Azure OpenAI Chat model for text generation
llm = AzureChatOpenAI(
    azure_deployment=gpt_deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


def load_pdf_with_langchain(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    document = loader.load()
    return document


def recursive_chunking(docs, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into fixed-size chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


docs = load_pdf_with_langchain("rag/academic_research_data.pdf")

print("\n Sample Extracted Content:")
for i, doc in enumerate(docs[:3]):
    print(f"\n--- Chunk {i + 1} ---")
    print(doc.page_content[:500])  # Show first 500 characters
    print("Metadata:", doc.metadata)

recursive_chunks = recursive_chunking(docs)
print("\nTotal Recursive Chunks:", len(recursive_chunks))
print("\n--- First Recursive Chunk ---")
print(recursive_chunks[0].page_content[:])  # Show first 500 characters
