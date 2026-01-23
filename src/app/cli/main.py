import typer
import logging
from app.agents.runner import run_pipeline
from app.rag.pdf_loader import extract_text_from_pdfs
from app.rag.splitter import split_text
from app.rag.vectorstore import build_vector_index
from app.config import settings
import json

app = typer.Typer()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.command()
def build_index(
    pdf_path: str = typer.Option(settings.PDF_PATH, help="Path to folder containing PDFs"),
    index_dir: str = typer.Option(settings.INDEX_DIR, help="Path to save FAISS index")
):
    """
    Builds the FAISS vector index from PDFs.
    """
    logger.info(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdfs(pdf_path)
    if not text:
        logger.error("No text extracted. Check if PDFs exist in the folder.")
        raise typer.Exit(code=1)

    logger.info("Splitting text...")
    documents = split_text(text)

    logger.info(f"Building index at {index_dir}...")
    build_vector_index(documents, index_dir)
    logger.info("Index built successfully.")

@app.command()
def run(
    image_path: str = typer.Option(..., help="Path to the image to analyze"),
    model_name: str = typer.Option("llava:13b", help="LLM model name (e.g. llava:13b, gpt-4o)"),
    output: str = typer.Option("result.json", help="Output JSON file path")
):
    """
    Runs the medical RAG and segmentation pipeline on an image.
    """
    logger.info(f"Running pipeline on {image_path} using {model_name}...")
    try:
        result = run_pipeline(image_path, model_name=model_name)

        # Serialize result (some fields might be Pydantic models)
        serialized_result = {}
        for k, v in result.items():
            if hasattr(v, "model_dump"):
                serialized_result[k] = v.model_dump()
            else:
                serialized_result[k] = v

        with open(output, "w") as f:
            json.dump(serialized_result, f, indent=4, ensure_ascii=False)

        logger.info(f"Analysis complete. Results saved to {output}")
        print(json.dumps(serialized_result, indent=2, default=str)[:500] + "...") # Preview

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
