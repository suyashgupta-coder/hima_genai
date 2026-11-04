from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

texts =[
    "Hello i am suyash gupta",
    "i am an AI developer",
    "I am working in techsara"
]
vector = embeddings.embed_documents(texts)
print(vector)