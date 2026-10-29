import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("api key is missing")

embeddings=GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

texts=[
    "Employees receive 20 annual leave days each calendar year.",
    "Workers get 20 vacation days every year.",
    "Employees may work remotely up to 3 days per week.",
]

vectors=embeddings.embed_documents(texts)

print(f"Number of texts:{len(texts)}")
print(f"Number of embeddings:{len(vectors)}")

for index,vector in enumerate(vectors,start=1):
    print(f"\nText{index}:{texts[index-1]}")
    print(f"Vector dimensions:{len(vectors)}")
    print(f"First 10 values:{vector[:10]}")

question="How many vacation days do employees receive?"

query_vector=embeddings.embed_query(question)

print(question)
print("Query vector dimensiions:",len(query_vector))
