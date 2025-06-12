import os
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Allow frontend (Streamlit etc.) to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    logger.info("Health check route called.")
    return {"message": "Wasserstoff GenAI API is running!"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")

        # Ensure the 'data' folder exists
        os.makedirs("data", exist_ok=True)

        # Read contents of uploaded file
        contents = await file.read()

        # Write file to 'data/' directory
        file_path = os.path.join("data", file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"Saved file to: {file_path}")
        return {"filename": file.filename, "status": "uploaded"}

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "failed"},
        )


import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import logging

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # update path as needed

logger = logging.getLogger(__name__)

def extract_text_from_file(file_path):
    text = ""
    metadata = {}

    if file_path.endswith(".pdf"):
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num+1}]\n" + page_text

            if len(text.strip()) == 0:
                images = convert_from_path(file_path, poppler_path=r"C:\path\to\poppler\Library\bin")  # update path
                for i, image in enumerate(images):
                    ocr_text = pytesseract.image_to_string(image)
                    text += f"\n[OCR Page {i+1}]\n" + ocr_text
        except Exception as e:
            logger.error(f"OCR fallback error: {e}")
            text += f"\n[OCR Fallback Error] {str(e)}"
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            text = ""

    metadata["length"] = len(text)
    return text, metadata
import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Wasserstoff GenAI", layout="wide")
st.title("Document Research & Theme Identification Chatbot")

# --- File Upload Section ---
st.header("Upload Documents")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "jpg", "png"])

if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    response = requests.post(f"{BACKEND_URL}/upload/", files=files)

    if response.status_code == 200:
        st.success(f"Uploaded {uploaded_file.name}")
    else:
        st.error("Upload failed!")

# --- Query Section ---
st.header("Ask a Question")
question = st.text_input("Type your question here:")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            res = requests.post(f"{BACKEND_URL}/ask/", json={"question": question})
            if res.status_code == 200:
                answer = res.json().get("answer")
                st.success(answer)
            else:
                st.error("Failed to get answer.")

     
