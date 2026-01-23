from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.config import settings

def get_embeddings():
    return SentenceTransformerEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
