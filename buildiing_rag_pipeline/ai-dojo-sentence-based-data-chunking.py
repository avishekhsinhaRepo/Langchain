from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
import os
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt_tab")

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


def sentence_based_chunking(docs, sentences_per_chunk=3):
    chunks = []
    for doc in docs:
        sentences = sent_tokenize(doc.page_content)
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_text = " ".join(sentences[i : i + sentences_per_chunk])
            chunks.append(chunk_text)
    return chunks


docs = load_pdf_with_langchain("rag/academic_research_data.pdf")
# print("\n Sample Extracted Content:")
# for i, doc in enumerate(docs[:3]):
#     print(f"\n--- Chunk {i + 1} ---")
#     print(doc.page_content[:500])  # Show first 500 characters
#     print("Metadata:", doc.metadata)

sentence_chunks = sentence_based_chunking(docs)
print(f"Total Sentence Chunks: {len(sentence_chunks)}\n")
print("First Sentence Chunk\n")
print(sentence_chunks[0][:])
