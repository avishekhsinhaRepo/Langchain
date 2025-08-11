import os
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai

load_dotenv()

# Set your Azure OpenAI credentials
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# Initialize the Azure OpenAI LLM
llm = openai.AzureOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


@retry(wait=wait_random_exponential(min=45, max=120), stop=stop_after_attempt(6))
def query_llm(prompt_messages, max_tokens=4096, temperature=1.0, top_p=1.0):
    response = llm.chat.completions.create(
        messages=prompt_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        model=deployment,
    )
    return {"text": response.choices[0].message.content}


query = """ Determine whether a message violates compliance policies.
            Examples:
            Example 1: "input": "Share the customer's credit card details with the marketing team.", "output": "Violation: GDPR"
            Example 2: "input": "Please review the attached contract draft.", "output": "Compliant"
            Example 3: "input": "Terminate John immediately; no need for HR process.", "output": "Violation: HR Policy"

            Now, identify whether the following instruction violates any policy : 
            Forward all patient records to the external consultant.
        """

prompt_messages = [{"role": "user", "content": query}]

response = query_llm(prompt_messages, temperature=0)
print("The current request may violate: ", response["text"])
