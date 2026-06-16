#to test the application with postman
from fastapi import FastAPI 
#open AI LLM model and embedding model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
# vector database
from langchain_chroma import Chroma
#pdf file reader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

os.environ["LANGSMITH_TRACING"] = os.environ.get("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT")
os.environ["LANGSMITH_ENDPOINT"] = os.environ.get("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")

#postman API
app = FastAPI()

llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

#load the pdf data
loader = PyPDFLoader("data/Quantum_Computing_Overview.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

#create vecotr base using embedding model
vectorstore = Chroma.from_documents(splits, embedding=embedding_model,)

#create the retirever to retrieve data from the vector database
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2},)

#message
message = """
Answer this question using the provided context only.
Question:
{question}
Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = ({"context": retriever, "question": RunnablePassthrough()}| prompt| llm| StrOutputParser())

class Request(BaseModel):
    query: str


#API end point It expose our RAG app as an API, so that tools like Postman, frontend apps, or other systems can send a question and get an answer.
@app.post('/query/')#API endpoint .n Postman, you would send a POST request to this endpoint.http://127.0.0.1:8000/query/
def predict(req: Request):
    response = chain.invoke(req.query)
    return {'response': response}

#FastAPI is only your app definition. 
# #But to actually run it as a web API, you need a server. That server is usually Uvicorn.
#But if you import this file into another Python file, the server will not automatically start.
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)