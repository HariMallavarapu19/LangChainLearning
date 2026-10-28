import os
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing")

class Answer(BaseModel):
    answer:str=Field(description="A direct answer to the user question")
    explanation:str=Field(description="A short explanation supporting the answer")
    confidence:float=Field(ge=0.0,le=1.0,description="The model self assessed confidence from 0 to 1")

llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature=0)

structured_llm=llm.with_structured_output(Answer)

if __name__=="__main__":
    question=input("Ask a question:")
    try:
        result=structured_llm.invoke(f"Answer the following question:{question}")
        print('\nStructured result:')
        print(result.model_dump_json(indent=2))
        print("\nAccess individual fields:")
        print("Answer:",result.answer)
        print("Explantion:",result.explanation)
        print("Confidence:",result.confidence)

    except Exception as error:
        print(f"Something went wrong:{error}")

