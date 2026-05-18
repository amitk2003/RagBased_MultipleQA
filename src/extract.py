# src/extract.py
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use absolute import so this works whether run from src/ or project root
try:
    from config import CHUNK_SIZE, CHUNK_OVERLAP
except ModuleNotFoundError:
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[extract] Error reading {pdf_path}: {e}")
    return text.strip()


def chunk_text(text: str) -> list:
    """Split text into overlapping chunks respecting word boundaries."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_text(text)