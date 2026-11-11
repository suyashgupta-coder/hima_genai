from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

#1 create document loader
loader = PyPDFLoader("D:/hima_genai/phase2/hybrid_rag/genAI pdf.pdf")
doc = loader.load()

#2 creating text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size =1000,
    chunk_overlap =100
)
chunks = splitter.split_documents(doc)

#3 create Dense retriever using Chroma DB
embeddings = GoogleGenerativeAIEmbeddings(
    model= "models/gemini-embedding-001"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="hybrid_rag"
)

dense_retriever = vectorstore.as_retriever(search_kwargs={"k":3})

#4 Create Sparse retriver uing BM25

sparse_retriever = BM25Retriever.from_documents(chunks)
sparse_retriever.k = 3 

#5 combine dense + sparse

retriever = EnsembleRetriever(
    retrievers=[dense_retriever,sparse_retriever] ,
    weights=[0.5,0.5]
    )

question = input("Ask your question:")
results = retriever.invoke(question)
context = "/n /n".join([doc.page_content for doc in results])

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}
Question:{question}
"""
res = model.invoke(prompt)
print(res.content[0]["text"])
