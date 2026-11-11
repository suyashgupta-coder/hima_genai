
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

# 1. Create embedding model
from langchain_huggingface import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load PDF
loader = PyPDFLoader(
    "D:/hima_genai/phase2/denserag/document loaders/genAI pdf.pdf"
)

docs = loader.load()

# 3. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))

# 4. Store chunks in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=model,
    persist_directory="D:/hima_genai/phase2/denserag/vector_store/chroma_db",
    collection_name="dense_rag"
)

print("Vector store created successfully!")
print("Total stored chunks:", vectorstore._collection.count())
