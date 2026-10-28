from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


#1 prompt template
prompt_template = ChatPromptTemplate.from_template(
    "Explain {topic} in simple term"
)

# 2 model 
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

#sequence Runnable
#prompt -> model -> parser 

chain = prompt_template | model | parser

result = chain.invoke(
    {
        "topic":"Genrative AI"
    }


)

print("Result :" ,result)