from dotenv import load_dotenv
load_dotenv()

#way 1
# from langchain_google_genai import ChatGoogleGenerativeAI

# model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",temperture=0.9,max_token=200)
# response = model.invoke("tell me about virat kholi") 
# print(response.content)

#way 2
# from langchain.chat_models import init_chat_model 
# model = init_chat_model("groq:openai/gpt-oss-120b")
# response = model.invoke("Why do parrots talk?")
# print(response.content)

# #way 3 
# from langchain.chat_models import init_chat_model 
# model = init_chat_model(model="openai/gpt-oss-120b",model_provider="groq")
# response = model.invoke("what is gen ai ?")
# print(response.content)


#Huggingface(using huggingface server)

# from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

# llm = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-20b",
   
# )
# model = ChatHuggingFace(llm=llm)
# response = model.invoke("give me 5 simple ai interview questions  ?")
# print(response.content)

#Download model locally using hugging face 

# from langchain_huggingface import HuggingFacePipeline
# from transformers import pipeline

# model = HuggingFacePipeline(
#     pipeline=pipeline(
#         "text-generation",
#         model="Qwen/Qwen2.5-0.5B-Instruct"

#     )
# )

# response = model.invoke("What is langchain?")
# print(response)



