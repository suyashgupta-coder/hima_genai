
import boto3
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite")
NUM_RESULTS = int(os.getenv("NUM_RESULTS", "5"))

# Initialize clients
bedrock = boto3.client(
    "bedrock-agent-runtime",
    region_name=AWS_REGION
)

gemini = genai.Client(api_key=GOOGLE_API_KEY)

# System prompt
SYSTEM_PROMPT = """
You are a business and financial assistant.
Answer only using the knowledge base context.
If information is not available, say so.
Never invent numbers.
Keep answers clear and simple.
"""


# Step 1: Retrieve information from Bedrock
def retrieve(question: str) -> list[str]:

    response = bedrock.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={
            "managedSearchConfiguration": {
                "numberOfResults": NUM_RESULTS
            }
        }
    )

    chunks = [
        r["content"]["text"]
        for r in response.get("retrievalResults", [])
        if r.get("content", {}).get("text")
    ]

    print(f"Retrieved {len(chunks)} chunks")

    return chunks


# Step 2: Generate answer using Gemini
def answer(question: str, history=None) -> str:

    chunks = retrieve(question)

    if not chunks:
        return "Information not found in the knowledge base."

    context = "\n\n".join(chunks)

    recent = str(history[-6:]) if history else "No previous conversation."

    prompt = f"""
    Knowledge Base Context:
    {context}

    Previous Conversation:
    {recent}

    Question:
    {question}
    """

    response = gemini.models.generate_content(
        model=GOOGLE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        )
    )

    return response.text or "No answer generated."


# Step 3: Run the application
if __name__ == "__main__":

    import sys

    question = " ".join(sys.argv[1:]) or "What is the past revenue?"

    print("\nANSWER:\n")
    print(answer(question))
