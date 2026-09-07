"""PDF extraction and RAG construction for dermatology literature."""
import logging
import os
import zipfile
from pathlib import Path

import pymupdf  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdfs(pdf_folder: str) -> str:
    """Extracts and concatenates text from all PDFs in a folder.

    Args:
        pdf_folder: Directory containing PDF files.

    Returns:
        Concatenated text from all PDFs.
    """
    text_data = ""
    pdf_folder = Path(pdf_folder)
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    logger.info("Extracting text from %d PDF files in %s", len(pdf_files), pdf_folder)

    for pdf_file in pdf_files:
        pdf_path = pdf_folder / pdf_file
        doc = pymupdf.open(str(pdf_path))
        for page in doc:
            text_data += page.get_text("text") + "\n\n"
        logger.debug("Extracted text from %s", pdf_file)

    return text_data


def prepare_medical_texts(zip_path: str, extracted_path: str, output_txt_path: str) -> str:
    """Extracts PDFs from a zip, processes them, and saves consolidated text.

    Args:
        zip_path: Path to the RAG.zip file containing PDFs.
        extracted_path: Directory to extract PDFs into.
        output_txt_path: Path to save the consolidated medical_texts.txt.

    Returns:
        Path to the saved text file.
    """
    os.makedirs(extracted_path, exist_ok=True)

    logger.info("Extracting ZIP: %s -> %s", zip_path, extracted_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    medical_texts = extract_text_from_pdfs(extracted_path)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(medical_texts)

    logger.info(
        "Saved medical texts to %s (%d chars)", output_txt_path, len(medical_texts)
    )
    return output_txt_path


def build_rag(
    medical_texts_path: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Builds a FAISS vectorstore retriever from the medical texts file.

    Args:
        medical_texts_path: Path to the consolidated medical_texts.txt file.
        embedding_model: SentenceTransformer model name (e.g. "all-MiniLM-L6-v2").
        chunk_size: Character size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A LangChain retriever backed by a FAISS vectorstore.
    """
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores.faiss import FAISS

    logger.info(
        "Building RAG from %s with model '%s'", medical_texts_path, embedding_model
    )

    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)

    # Carregar e processar documentos médicos (extraídos dos PDFs)
    loader = TextLoader(medical_texts_path)
    texts = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(texts)

    # Criar banco vetorial FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    logger.info("RAG built with %d document chunks", len(documents))
    return retriever
