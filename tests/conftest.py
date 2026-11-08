# Make `src/` importable when running `pytest` from the repository root (no install needed).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
