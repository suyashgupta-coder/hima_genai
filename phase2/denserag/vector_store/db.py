from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
loader = PyPDFLoader("D:/hima_genai/phase2/denserag/document loaders/genAI pdf.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chunks = splitter.split_documents(docs)
vectorstore=Chroma.from_documents(chunks,embeddings,persist_directory="D:/hima_genai/phase2/denserag/vector_store/chroma_db")
print("vector store created ")
