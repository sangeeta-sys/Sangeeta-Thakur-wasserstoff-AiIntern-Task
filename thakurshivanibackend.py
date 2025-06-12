import os
import logging
import pdfplumber
import pytesseract
from PIL import Image
from pdfminer.high_level import extract_text
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ChromaDB setup (NEW CONFIG) ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# OCR Functions
def extract_text_from_pdf(file_path):
    return extract_text(file_path)

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

# Add to vector store
def add_to_vectordb(doc_id, text, embedding):
    collection.add(documents=[text], ids=[doc_id], embeddings=[embedding.tolist()])

# Search in vector store
def search_similar(query_embedding, n_results=5):
    return collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)

# Root route
@app.get("/")
def read_root():
    return {"message": "Wasserstoff GenAI API is running!"}

# File upload route
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Text extraction logic
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_image(file_path)

        embedding = model.encode(text)
        add_to_vectordb(file.filename, text, embedding)

        return {"filename": file.filename, "status": "uploaded"}

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return {"error": str(e)}

# Pydantic model for question requests
class QuestionRequest(BaseModel):
    question: str

# Question answering route
@app.post("/ask/")
async def ask_question(payload: QuestionRequest):
    question = payload.question
    logger.info(f"Received question: {question}")

    query_embedding = model.encode(question)
    results = search_similar(query_embedding)

    answer_list = []
    for doc, dist, doc_id in zip(results['documents'][0], results['distances'][0], results['ids'][0]):
        answer_list.append({
            "document": doc_id,
            "extracted_answer": doc,
            "distance": dist
        })

    return {"answers": answer_list}
