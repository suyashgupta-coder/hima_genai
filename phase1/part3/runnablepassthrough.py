from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system","you are a code generator"),
    ("human","{topic}")
])

explain_prompt =ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant who explains code in simple terms"),
    ("human","{topic}")
])

seq = prompt| model| parser
seq2 = RunnableParallel(
    {
        "code":RunnablePassthrough(),
        "explanation": explain_prompt|model|parser
    }
)
chain =seq|seq2

result= chain.invoke({
    "topic" :"Write a code of bubble sort in C++"
})
print(result['code'])
print("\n")
print(result['explanation'])