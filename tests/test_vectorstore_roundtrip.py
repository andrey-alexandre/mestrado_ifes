import pytest
from app.rag.vectorstore import build_vector_index, load_vector_index
from langchain_core.documents import Document
import shutil
import os

def test_vectorstore_roundtrip(tmp_path):
    docs = [Document(page_content="Test document", metadata={"id": 1})]
    index_dir = str(tmp_path / "index")

    build_vector_index(docs, index_dir)

    assert os.path.exists(index_dir)

    vectorstore = load_vector_index(index_dir)
    assert vectorstore is not None

    results = vectorstore.similarity_search("Test", k=1)
    assert len(results) == 1
    assert results[0].page_content == "Test document"
