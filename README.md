# Medical RAG & Segmentation Pipeline

A structured Python project for dermatological analysis using RAG (Retrieval Augmented Generation) and Image Segmentation.

## Features
- **RAG**: Ingests medical PDFs to answer questions using FAISS and LangChain.
- **Image Segmentation**: Uses U-Net (simulated or real if weights provided) to segment skin lesions.
- **Agents**: LangGraph-based agents for ABCD, Menzies, and Seven-Point Checklist analysis.
- **CLI**: Command-line interface for running the pipeline.

## Installation

```bash
pip install .
```

## Configuration

Copy `.env.example` to `.env` and set your variables.

## Usage

### Build Index
```bash
python scripts/build_index.py
```

### Run Pipeline
```bash
python scripts/run_pipeline.py --image-path path/to/image.jpg
```
