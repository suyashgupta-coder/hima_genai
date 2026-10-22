from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
print("___________ welcome to my chatbot and press 0  to exit ____________")

while True:
    prompt = input("YOU:")
    if prompt == "0":
        print("\n Thank you for using my chatbot! \n")
        break
    response = model.invoke(prompt) 
    print("BOT:",response.content[0]["text"])