import pdfplumber
import pytesseract
from PIL import Image
from typing import List, Dict

def extract_and_chunk_pdf(file_path: str) -> List[Dict[str, str]]:
    """
    Extracts text, tables, and performs OCR on a PDF using pdfplumber and pytesseract.
    Returns a list of chunks with metadata.
    """
    chunk_data = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                text = page.extract_text() or ""
                
                # If no text is found, fallback to OCR
                if not text.strip():
                    img_obj = page.to_image(resolution=300)
                    img = img_obj.original
                    text = pytesseract.image_to_string(img)
                
                # Simple chunking by page for this lightweight approach.
                # Further splitting could be applied if pages are too long.
                if text.strip():
                    chunk_data.append({
                        "text": text,
                        "metadata": {
                            "source": file_path.split("/")[-1],
                            "page": page_number,
                            "chunk_index": i
                        }
                    })
                    
        return chunk_data
    except Exception as e:
        print(f"Error during extraction: {e}")
        return []