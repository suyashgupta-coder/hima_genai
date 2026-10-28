from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage , AIMessage
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
print("___________ welcome to my chatbot and press 0  to exit ____________")
print("AI CHATBOT")
print("1. Funny")
print("2. Angry")
print("3 Sad")
print("4 Romantic")

personalities = {
    "1": "You are a funny chatbot who responds with humor",
    "2": "You are a angry chatbot who responds with irritated tone",
    "3": "You are a sad chatbot who responds in a melancholic tone",
    "4": "You are a romantic chatbot who responds in a romantic tone  "

}

choice = input("\n Choose personality (1-4): ")
if choice not in personalities:
    print("invalid choice")
    exit()
messages =[
    SystemMessage(content= personalities[choice])
]
while True:
    prompt = input("YOU:")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        print("\n Thank you for using my chatbot! \n")
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content[0]["text"])) 
    print("BOT:",response.content[0]["text"])
    #print(messages)