from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("D:/hima_genai/phase2/denserag/document loaders/notes.txt")
docs = loader.load()

splitter = CharacterTextSplitter(separator="",chunk_size=20,chunk_overlap=1)

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
