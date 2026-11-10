
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

# 1. Create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Create chat model
llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

# 3. Connect to vector store
vectorstores = Chroma(
    persist_directory="D:/hima_genai/phase2/denserag/vector_store/chroma_db",
    embedding_function=embedding_model,
    collection_name="dense_rag"
)

print("Total stored chunks:", vectorstores._collection.count())

# 4. Create retriever
retriever = vectorstores.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.7
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

print("RAG System is ready!")
print("Type 0 to exit")

# 6. Chat loop
while True:

    query = input("\nYOU: ")

    if query == "0":
        break

    # 7. Retrieve relevant documents
    docs = retriever.invoke(query)

    # 8. Combine retrieved documents
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    print("\nRetrieved Context:\n", context)

    # 9. Generate answer
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print("\nAI:", response.content)
