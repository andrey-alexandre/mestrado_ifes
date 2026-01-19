import logging
from app.agents.graph import build_graph
from app.config import settings
from app.rag.vectorstore import load_vector_index

logger = logging.getLogger(__name__)

def run_pipeline(image_path: str, model_name: str = "llava:13b", index_dir: str = None):
    index_dir = index_dir or settings.INDEX_DIR

    try:
        vectorstore = load_vector_index(index_dir)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        logger.warning(f"Could not load vector index: {e}. Running without RAG validation or with dummy.")
        # Create a dummy retriever if needed or handle gracefully
        class DummyRetriever:
            def invoke(self, query):
                return []
        retriever = DummyRetriever()

    graph = build_graph(model_name, retriever)

    initial_state = {"image_path": image_path}

    final_state = graph.invoke(initial_state)
    return final_state
