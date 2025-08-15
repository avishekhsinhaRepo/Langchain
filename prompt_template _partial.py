import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
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

prompt_template = PromptTemplate(
    input_variables=["country", "no_of_paras", "language"],
    template="""You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cuisine of {country}?
    Answer in {no_of_paras} short paras in {language}
    Say thanks to {my_name}
    """,
)

country = input("Enter the country:")
no_of_paras = int(input("Enter the number of paras"))
language = input("Enter the language:")


# Create a partial prompt with initial variables
question = prompt_template.partial(
    country=country,
    no_of_paras=no_of_paras,
    language=language
)

# Create a chain that combines the LLM with the prompt
chain = {"my_name": lambda name_dict: name_dict["my_name"]} | question | llm

# Run the chain
response = chain.invoke({"my_name": "Avi"})
print(response.content)
