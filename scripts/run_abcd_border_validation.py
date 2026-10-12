"""Run synthetic ABCD-B border validation and write plots/results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.border_validation import main


if __name__ == "__main__":
    main()
