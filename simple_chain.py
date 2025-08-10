import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

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

topic_prompt = PromptTemplate(
    input_variables=["topic"], template="Create a speech on following topic: {topic}"
)


speech_prompt = PromptTemplate(
    input_variables=["no_of_words"],
    template="keep the speech under {no_of_words} words. "
    "Make it engaging and informative. ",
)

first_chain = topic_prompt | llm | StrOutputParser()
second_chain = speech_prompt | llm | StrOutputParser()

final_chain = first_chain | second_chain

response = final_chain.invoke({"topic": "Artificial Intelligence", "no_of_words": 1000})
print(response)
