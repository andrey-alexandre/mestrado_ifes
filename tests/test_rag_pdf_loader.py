import pytest
from app.rag.pdf_loader import extract_text_from_pdfs
import os

def test_extract_text_no_folder():
    text = extract_text_from_pdfs("non_existent_folder")
    assert text == ""

def test_extract_text_empty_folder(tmp_path):
    text = extract_text_from_pdfs(str(tmp_path))
    assert text == ""

def test_extract_text_with_pdf(tmp_path):
    # Setup dummy pdf
    import fitz # pymupdf
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Hello World")
    pdf_path = tmp_path / "test.pdf"
    doc.save(pdf_path)

    text = extract_text_from_pdfs(str(tmp_path))
    assert "Hello World" in text
