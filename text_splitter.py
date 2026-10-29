from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader=TextLoader("documents/company_policy.txt",encoding="utf-8")

documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    length_function=len
)

chunks=text_splitter.split_documents(documents)

print(f"Total chunks created:{len(chunks)}")

for index,chunk in enumerate(chunks,start=1):
    print(chunk.page_content)
    print("Metadata:",chunk.metadata)
    print("Character count:",len(chunk.page_content))
