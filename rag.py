"""Core retrieve-then-answer logic, shared by ask.py, app.py and the eval script.
LLM backend: local Ollama (default model: llama3.1).
"""
from functools import lru_cache

from config import (CHROMA_DIR, COLLECTION, EMBED_MODEL, LLM_MODEL, NUM_CTX,
                    OLLAMA_HOST, REFUSAL, TOP_K)

SYSTEM_PROMPT = f"""You answer questions about Kubernetes using ONLY the numbered context passages provided.

Rules:
- Use only information found in the context. Do not use outside knowledge.
- Cite the passages you used inline like [1] or [2][3].
- If the context does not contain enough information to answer, reply with exactly: {REFUSAL}
- Be concise and practical. Include short YAML or commands only if they appear in the context.
"""


@lru_cache(maxsize=1)
def _collection():
    import chromadb
    from chromadb.utils import embedding_functions

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION, embedding_function=ef)


@lru_cache(maxsize=1)
def _llm():
    import ollama
    return ollama.Client(host=OLLAMA_HOST)


def chat(system: str, user: str, max_tokens: int = 700, json_mode: bool = False) -> str:
    """One call to the local model. Temperature 0 keeps answers repeatable for evals."""
    kwargs = {"format": "json"} if json_mode else {}
    resp = _llm().chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        options={"temperature": 0, "num_predict": max_tokens, "num_ctx": NUM_CTX},
        **kwargs,
    )
    return resp["message"]["content"].strip()


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    res = _collection().query(query_texts=[question], n_results=k)
    return [
        {"text": d, "distance": dist, **m}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks, 1))


def generate(question: str, chunks: list[dict]) -> str:
    return chat(SYSTEM_PROMPT, f"Context:\n{build_context(chunks)}\n\nQuestion: {question}")


def answer(question: str, k: int = TOP_K) -> dict:
    chunks = retrieve(question, k)
    return {"answer": generate(question, chunks), "chunks": chunks}