#invoke()
#batch()
#stream()

from langchain_core.runnables import RunnableLambda
runnable =RunnableLambda(lambda x:x.upper())
#result = runnable.invoke("hello")
# result = runnable.batch([
#     "hello",
#     'python',
#     'langchain'
# ])
# print(result)

for chunk in runnable.stream("hello"):
    print(chunk,end="")


import logging

logging.basicConfig(level=logging.INFO)

logging.info("Application started")
logging.warning("Something needs attention")
logging.error("An error occurred")