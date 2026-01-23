import os
import pymupdf
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdfs(pdf_folder: str) -> str:
    """Extracts text from all PDF files in a folder."""
    text_data = ""
    if not os.path.exists(pdf_folder):
        logger.warning(f"PDF folder {pdf_folder} does not exist.")
        return text_data

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            try:
                with pymupdf.open(pdf_path) as doc:
                    for page in doc:
                        text_data += page.get_text("text") + "\n\n"
            except Exception as e:
                logger.exception(f"Exception reading {pdf_path}: {e}")
    return text_data
