import pymupdf
import os
import zipfile

from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def unzip(zip_path: str, extract_path: str) -> None:
    # Extraindo os PDFs
    os.makedirs(extracted_path, exist_ok=True)
    with zipfile.ZipFile(extract_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)


def extract_text_from_pdfs(pdf_folder: str) -> str:
    text_data = ""
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc = pymupdf.open(pdf_path)
            for page in doc:
                text_data += page.get_text("text") + "\n\n"
    return text_data

def write_to_txt(content: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def create_retriever(path: str, model: str = "all-MiniLM-L6-v2"):
    embeddings = SentenceTransformerEmbeddings(model_name=model)

    # Carregar e processar documentos médicos (já extraídos dos PDFs)
    loader = TextLoader(path)  # Arquivo consolidado com informações médicas extraídas
    texts = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(texts)

    # Criar banco vetorial FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    return retriever