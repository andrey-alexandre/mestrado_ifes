import sys
import os

# Add src to python path for scripts
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from app.cli.main import app

if __name__ == "__main__":
    app()
