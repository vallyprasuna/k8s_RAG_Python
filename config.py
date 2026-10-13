# DATA_DIR = './data'
# CHUNK_WORDS = 500
# OVERLAP_WORDS = 50
# CHROMA_DIR = './chroma_db'
# COLLECTION = 'kubedocs'
# EMBED_MODEL = 'all-MiniLM-L6-v2'
# LLM_MODEL = ''
# REFUSAL = 'I don\'t know'
# TOP_K = 5

"""Shared settings. Override any of these with environment variables to run experiments,
e.g.  CHUNK_WORDS=200 python ingest.py   or   TOP_K=8 python eval/run_eval.py --label top8
"""
import os

DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = "k8s_docs"

EMBED_MODEL = "all-MiniLM-L6-v2"                       # local, free
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")      # any model you have pulled with `ollama pull`
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))            # Ollama's default context is small; 5 chunks need more

CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "350"))     # ~500 tokens
OVERLAP_WORDS = int(os.getenv("OVERLAP_WORDS", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))

REFUSAL = "I don't know based on the docs."
HF_HUB_OFFLINE=int(os.getenv("HF_HUB_OFFLINE", "1")) 