import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import answer
from ingest import ingest_pdf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    result = answer(q.question)
    return result

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    pdf_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    doc_name = file.filename.replace(".pdf", "")
    ingest_pdf(pdf_path, doc_name)
    return {"message": f"✅ '{file.filename}' uploaded and indexed successfully!"}

@app.get("/")
def root():
    return {"message": "RAG API is running!"}