from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("D:/hima_genai/phase2/denserag/document loaders/genAI pdf.pdf")
docs = loader.load()
print(docs[0].page_content)