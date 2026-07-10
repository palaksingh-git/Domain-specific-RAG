import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

load_dotenv()

CHROMA_PATH = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="documents")
embedder = SentenceTransformer(EMBED_MODEL)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def retrieve(question: str, top_k: int = 3):
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"]
        })
    return chunks

def answer(question: str):
    chunks = retrieve(question)

    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the documents.",
            "sources": []
        }

    context = "\n\n".join([
        f"[{c['source']} p.{c['page']}]: {c['text']}"
        for c in chunks
    ])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    response = model.generate_content(prompt)
    sources = [{"source": c["source"], "page": c["page"]} for c in chunks]

    return {"answer": response.text, "sources": sources}

if __name__ == "__main__":
    result = answer("What is this document about?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])