"""Entry point for Hugging Face Spaces. Runs the Streamlit demo."""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import and run demo main
from demo.app import main

if __name__ == "__main__":
    main()
