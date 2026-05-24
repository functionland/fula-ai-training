"""Shared pytest fixtures for the intake server tests."""
import os
import sys
from pathlib import Path

# Make `app`, `anonymization_check`, `storage` importable without packaging.
_SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SERVER_DIR))

# Force a deterministic storage dir BEFORE app.py imports so the FastAPI
# app picks up our path instead of defaulting to /tmp.
os.environ.setdefault("BLOX_AI_STORAGE_DIR", str(_SERVER_DIR / "_test_storage"))
