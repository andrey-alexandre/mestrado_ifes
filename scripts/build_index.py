import sys
import os

# Add src to python path for scripts
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


if __name__ == "__main__":
    # We invoke it via typer to match CLI usage
    from app.cli.main import app
    # This is a bit hacky to call via python script, but valid
    # It parses sys.argv by default
    app()
