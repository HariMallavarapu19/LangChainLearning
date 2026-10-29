import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("api key missing")

embeddings=GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

loader=TextLoader(
    "documents/company_policy.txt",
    encoding="utf-8"
)

documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks=text_splitter.split_documents(documents)

print(f"Total chunks created:{len(chunks)}")

vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="company_policy_learning",
    persist_directory="./chroma_db"
)

print("Documents stored in ChromaDB")

question=input("\nAsk a question about company policy:")
results=vector_store.similarity_search_with_score(question,k=3)
for index,(document,distance) in enumerate(results,start=1):
    print(document.page_content)
    print("Source:", document.metadata.get("source"))
    print("Distance:",round(distance,4))

    
