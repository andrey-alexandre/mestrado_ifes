import os
from langchain_community.vectorstores import FAISS
from app.rag.embeddings import get_embeddings
import logging

logger = logging.getLogger(__name__)

def build_vector_index(documents, index_dir: str):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_dir)
    return vectorstore

def load_vector_index(index_dir: str):
    if not os.path.exists(index_dir):
        logger.error(f"Index directory {index_dir} does not exist.")
        raise FileNotFoundError(f"Index not found at {index_dir}. Please run build_index script first.")

    embeddings = get_embeddings()
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
