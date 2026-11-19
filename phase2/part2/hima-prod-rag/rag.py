import boto3
import os
from dotenv import  load_dotenv
from google import genai
from google.genai import types 
load_dotenv()

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite")
NUM_RESULTS = int(os.getenv("NUM_RESULTS", "5"))

print("f starting RAG app | region={AWS_REGION}|kb={KNOWLEDGE_BASE_ID} | model={GOOGLE_MODEL}")
bedrock = boto3.client("bedrock-agent-runtime",region_name=AWS_REGION)
gemini = genai.Client(api_key=GOOGLE_API_KEY)
SYSTEM_PROMPT = (
    "You are a business and financial assistant. "
    "Answer only from the knowledge base context you are given. "
    "If the context does not contain the answer, say that the information "
    "was not found in the knowledge base. Never invent numbers. "
    "Keep answers clear and concise."
)
