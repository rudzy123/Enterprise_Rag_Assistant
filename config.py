import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("1", "true", "yes")
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.35"))
MIN_CHUNK_SIMILARITY = float(os.getenv("MIN_CHUNK_SIMILARITY", "0.20"))
CHUNK_WORD_LIMIT = int(os.getenv("CHUNK_WORD_LIMIT", "250"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TRACES_DIR = BASE_DIR / "traces"
TRACE_DB_PATH = TRACES_DIR / "traces.db"
