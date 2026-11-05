from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

#1 create embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

#2 Create a chat model

llm = ChatGroq(model="openai/gpt-oss-120b")

#3 connect vector store
vectorstore = Chroma(
    persist_directory="D:/hima_genai/phase2/denserag/vector_store/chroma_db",
    embedding_function=embedding_model
)

#4 create retriever
retriever = vectorstore.as_retriever(
    search_type ="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.7

    }
)

# 5. Create prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the question using only the given context.

If the answer is not available in the context, say:
"I don't know."

Context:
{context}

Question:
{question}
""")


print("RAG System is ready")
print("Type 0 to exit")

while True:
    qurey = input("\n You:")
    if qurey =="0":
        break
    #7 Retrive relevent documents
    docs = retriever.invoke(qurey)
#8 combaine retrived document
    context = "\n \n ".join(doc.page_content for doc in docs)
    print("\n Retrieved Context: \n",context)
    final_prompt = prompt.invoke({
        "context":context,
        "question":qurey
    })
    response = llm.invoke(final_prompt)
    print("AI:", response.content)