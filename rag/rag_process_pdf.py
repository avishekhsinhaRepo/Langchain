from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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

# Initialize the Azure OpenAI Embeddings for vector creation
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embedding_deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Initialize the Azure OpenAI LLM

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for answering questions.
                Use the provided context to respond.If the answer 
                isn't clear, acknowledge that you don't know. 
                Limit your response to three concise sentences.
                {context}
    
    """,
        ),
        ("human", "{input}"),
    ]
)
document = PyPDFLoader("rag/academic_research_data.pdf").load()
text_spiltter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_spiltter.split_documents(document)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

qa_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain)


question = input("Enter Your Question=")
if question:
    response = rag_chain.invoke({"input": question})
    print(response["answer"])
