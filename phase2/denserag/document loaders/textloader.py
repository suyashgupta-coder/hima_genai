from langchain_community.document_loaders import TextLoader

loader = TextLoader("D:/hima_genai/phase2/denserag/document loaders/notes.txt")
docs = loader.load()
print(docs[0].page_content)