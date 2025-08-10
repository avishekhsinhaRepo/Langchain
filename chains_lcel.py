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


prompt_template = PromptTemplate(
    input_variables=["language", "text"],
    template="Translate the following text to {language}: {text}",
)

text = input("Enter the text to translate: ")
language = input("Enter the target language: ")


parser = StrOutputParser()
# Example usage

chain = prompt_template | llm | parser
response = chain.invoke({"text": text, "language": language})
print(response)
# Example usage
