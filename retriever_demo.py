import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("api key is missing")

embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
loader=TextLoader("documents/company_policy.txt",encoding="utf-8")
documents=loader.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=30)
chunks=splitter.split_documents(documents)
vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="retriever_learning",
    persist_directory="./chroma_db"
)

retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})
question=input("Ask a question about companny policy:")
results=retriever.invoke(question)

for index,document in enumerate(results,start=1):
    print(document.page_content)
    print("Source:",document.metadata.get("source"))

    
     