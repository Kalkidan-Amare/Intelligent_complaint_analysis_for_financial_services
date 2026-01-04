from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"


@dataclass(frozen=True)
class Settings:
    raw_cfpb_path: Path = Path(os.getenv("CFPB_RAW_PATH", RAW_DIR / "cfpb_complaints.csv"))
    filtered_csv_path: Path = Path(os.getenv("FILTERED_CSV_PATH", DATA_DIR / "filtered_complaints.csv"))

    # Sample / indexing (Task 2)
    sample_size: int = int(os.getenv("SAMPLE_SIZE", "12000"))
    sample_seed: int = int(os.getenv("SAMPLE_SEED", "42"))
    sample_chroma_dir: Path = Path(os.getenv("SAMPLE_CHROMA_DIR", VECTOR_STORE_DIR / "sample_chroma"))

    # Full vector store (Task 3/4): point this to the provided Chroma directory
    full_chroma_dir: Path = Path(os.getenv("FULL_CHROMA_DIR", VECTOR_STORE_DIR / "full_chroma"))
    full_chroma_collection: str = os.getenv("CHROMA_COLLECTION", "complaints_full")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Lightweight default LLM for CPU (override with env var)
    llm_model: str = os.getenv("LLM_MODEL", "google/flan-t5-base")
    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))


settings = Settings()
