from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal

class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    PDF_PATH: str = "data/pdfs"
    INDEX_DIR: str = "data/faiss_index"
    SEGMENTATION_WEIGHTS_PATH: str = "models/U_Net.pkl"
    DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
