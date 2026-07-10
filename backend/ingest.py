import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

CHROMA_PATH = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="documents")
embedder = SentenceTransformer(EMBED_MODEL)

def ingest_pdf(pdf_path: str, doc_name: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.encode(texts).tolist()

    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": doc_name,
            "page": chunk.metadata.get("page", 0) + 1
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"✅ Ingested {len(chunks)} chunks from '{doc_name}'")

if __name__ == "__main__":
    ingest_pdf("sample.pdf", "sample")