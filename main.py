from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import shutil
from typing import Optional
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever state
retriever = None

class QueryRequest(BaseModel):
    input_query: str

@app.post("/upload-file/")
async def upload_file(file: UploadFile):
    try:
        # Save file temporarily
        temp_file_path = f"./temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Determine file type and load accordingly
        if file.filename.endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()
        elif file.filename.endswith(".txt"):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(temp_file_path)
            data = loader.load()
        elif file.filename.endswith(".docx"):
            from docx import Document

            def load_docx(file_path):
                document = Document(file_path)
                text = "\n".join([paragraph.text for paragraph in document.paragraphs])
                return [{"page_content": text}]

            data = load_docx(temp_file_path)
        elif file.filename.endswith(".ppt"):
            from pptx import Presentation

            def load_ppt(file_path):
                presentation = Presentation(file_path)
                text = []
                for slide in presentation.slides:
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            text.append(shape.text)
                return [{"page_content": "\n".join(text)}]

            data = load_ppt(temp_file_path)
        else:
            raise ValueError("Unsupported file type")



        # data = loader.load()
        os.remove(temp_file_path)

        if not data:
            return JSONResponse(status_code=400, content={"message": "No data found in the file."})

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)

        # Create Chroma DB vector store
        global retriever
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
            persist_directory="./chroma_db",
        )

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        return {"message": "File processed successfully."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing file: {str(e)}"})

@app.post("/query/")
async def query_model(request: QueryRequest):
    try:
        if retriever is None:
            return JSONResponse(status_code=400, content={"message": "Retriever not initialized. Upload a file first."})

        # System prompt for the model
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Set up the question-answer chain
        question_answer_chain = create_stuff_documents_chain(
            ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0),
            prompt
        )
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Get the response from the RAG chain
        response = rag_chain.invoke({"input": request.input_query})
        return {"answer": response['answer']}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing query: {str(e)}"})

@app.get("/health/")
def health_check():
    return {"status": "ok"}

if __name__ =='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)