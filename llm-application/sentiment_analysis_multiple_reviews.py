from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import pandas as pd
import json

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
file_path = os.path.join(script_dir, "patient_reviews.csv")

reviews = pd.read_csv(file_path)
reviews.head()

top_3_reviews = reviews["review_text"].head(3)

query = """ You are a sentiment analysis expert. Please analyze the following text and return a detailed JSON response with the following fields:
            - "sentiment_label": a string "positive", "negative", or "neutral" representing the overall sentiment.
            - "confidence_score": a numeric value between 0 and 1 indicating how confident you are in your sentiment classification.
            - "emotions": an array with the name of the detected emotions (e.g., "joy", "anger", "sadness", "surprise", "fear").
            Return the response in valid JSON format. Do not include the keyword json in the output

        """
result = []
for review in top_3_reviews:
    prompt_messages = [
        {"role": "developer", "content": query},
        {"role": "user", "content": f"Patient review text:\n\n{review}"},
    ]
    response = llm.invoke(prompt_messages)
    result.append(response.content)
result_dict = [json.loads(review) for review in result]
df = pd.DataFrame(result_dict)
print(df)
