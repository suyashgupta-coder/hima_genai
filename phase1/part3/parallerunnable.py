from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines in very simple terms"
)
detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail but it should be in simple terms"
)

chain =RunnableParallel({
    # "topic":"Machine Learning"
    "short": short_prompt|model|parser,
    "detail":detailed_prompt|model|parser
})

result = chain.invoke({
    "topic":"Machine Learning"
})
print("Short Explanation",result['short'])
print("detailed Explanation",result['detail'])
