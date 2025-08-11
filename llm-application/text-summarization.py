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


# Read Doctor and Patient conversation


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "conversation.txt")

with open(file_path, "r") as file:
    conversation = file.read()


prompt = [
    # {
    #     "role": "developer",
    #     "content": "You are an assistant that summarizes conversations between Doctor and Patient",
    # },
    # {
    #     "role": "developer",
    #     "content": "You are an assistant that summarizes conversations \
    #   between Doctor and Patient. Make sure to provide concise summary in 100 words.",
    # },
    {
        "role": "developer",
        "content": "You are an AI assistant specializing in summarizing conversations between a doctor and a patient. \
     Your task is to generate a clear, concise, and structured summary in 150 words or less. \
     Ensure that the summary captures the key points, including the patient's symptoms, \
     diagnosis, prescribed treatment, and follow-up instructions. \
     Present the summary in bullet points for easy readability",
    },
    {
        "role": "user",
        "content": f"Please summarize the following text:\n\n{conversation}\n\nSummary:",
    },
]


response = llm.invoke(prompt)
summary = response.content
summary_length = len(summary.split())
print(f"Summary Length(In words):{summary_length}")
print(summary)
