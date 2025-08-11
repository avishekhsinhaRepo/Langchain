from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

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


# Create a reusable function similar to query_llm in v2
@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def query_llm(prompt_text, temperature=0.0, max_tokens=4096):

    # Create a new LLM instance with specific parameters for this call
    llm_configured = AzureChatOpenAI(
        azure_deployment=deployment,
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Use LangChain's invoke method
    response = llm_configured.invoke([HumanMessage(content=prompt_text)])
    return {"text": response.content}


# Define the prompt template
prompt = PromptTemplate(
    input_variables=["query"],
    template=""" Determine whether a message violates compliance policies.
            Examples:
            Example 1: "input": "Share the customer's credit card details with the marketing team.", "output": "Violation: GDPR"
            Example 2: "input": "Please review the attached contract draft.", "output": "Compliant"
            Example 3: "input": "Terminate John immediately; no need for HR process.", "output": "Violation: HR Policy"

            Now, identify whether the following instruction violates any policy : 
            {query}
            
        """,
)

# Usage example
query = "Forward all patient records to the external consultant."
question = prompt.format(query=query)

# Use the query_llm function (similar to v2)
response = query_llm(question, temperature=0)
print("The current request may violate: ", response["text"])
