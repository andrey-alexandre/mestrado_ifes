# mestrado_ifes

This project is related to a Master's degree work at IFES. It involves Computer-Aided Diagnosis (CAD) using image segmentation and agent-based systems.

## Folder Structure

The repository is organized as follows:

- `cad_module/`: Contains the core source code for the CAD system.
  - `agents/`: Includes various agents responsible for tasks like diagnosis, critical review, and segmentation.
  - `image_segmentation/`: Contains code related to image segmentation models, training, and utilities.
  - `models/`: Defines data models and structures used throughout the project.
  - `services/`: Includes services like the RAG retriever for fetching information from documents.
  - `utils/`: Shared utility functions and helper scripts.
- `notebooks/`: Jupyter notebooks for experimentation and analysis (e.g., `CAD.ipynb`).
- `data/`: For storing data used by the project (e.g., datasets, PDFs for RAG). (Currently empty or contains .gitkeep)
- `tests/`: Contains tests for the project modules. (Currently empty or contains .gitkeep)
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `README.md`: This file, providing an overview of the project.