import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("api is missing")

llm=ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.2
)

prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI tutor."
            "The students experience level is {level}"
            "Explain concepts at the appropriate level"
            "Use examples when helpful"
        ),
        ("human","{question}")
    ]
)

chain= prompt | llm
def ask_question(question:str,level:str)->str:
    response=chain.invoke({"question":question,"level":level})
    return response.text()

if __name__=="__main__":
    question=input("Ask me a question")
    level=input("Choose your level(begineer/intermediate/advanced)")
    try:
        answer=ask_question(question,level)
        print(answer)

    except Exception as error:
        print(f"\n Something went wrong:{error}")



