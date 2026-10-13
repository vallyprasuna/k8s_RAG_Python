"""Read Markdown docs from data/, split into chunks, embed, and store in Chroma.

Usage:  python ingest.py
"""
import os
import re
from pathlib import Path

from config import DATA_DIR, CHUNK_WORDS, OVERLAP_WORDS, CHROMA_DIR, COLLECTION, EMBED_MODEL


def clean(text: str) -> tuple[str, str]:
    """Strip Hugo front matter and shortcodes. Returns (title, body)."""
    title = ""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.S)
    if m:
        t = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", m.group(1), flags=re.M)
        if t:
            title = t.group(1)
        text = text[m.end():]
    text = re.sub(r"\{\{[<%].*?[>%]\}\}", "", text, flags=re.S)  # {{< note >}} etc.
    return title, text.strip()


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split on Markdown headings (ignoring '#' lines inside code fences)."""
    sections, heading, buf, in_code = [], "", [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,4}\s+\S", line):
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def words(s: str) -> int:
    return len(s.split())


def chunk_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Pack paragraphs into chunks of ~max_words, carrying a small overlap between chunks."""
    paras = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if not p:
            continue
        if words(p) <= max_words:
            paras.append(p)
        else:  # very long block (e.g. big YAML): split by lines
            cur = []
            for line in p.splitlines():
                cur.append(line)
                if words("\n".join(cur)) >= max_words:
                    paras.append("\n".join(cur))
                    cur = []
            if cur:
                paras.append("\n".join(cur))

    chunks, cur = [], []
    for p in paras:
        if cur and words("\n\n".join(cur + [p])) > max_words:
            chunks.append("\n\n".join(cur))
            last = cur[-1]
            cur = [last] if words(last) <= overlap_words else []
        cur.append(p)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def to_url(rel_path: str) -> str:
    p = re.sub(r"\.md$", "", rel_path.replace(os.sep, "/"))
    p = re.sub(r"/?_index$", "", p)
    return f"https://kubernetes.io/docs/{p}/"


def build_chunks(data_dir: str) -> list[dict]:
    out = []
    for path in sorted(Path(data_dir).rglob("*.md")):
        rel = str(path.relative_to(data_dir))
        title, body = clean(path.read_text(encoding="utf-8", errors="ignore"))
        title = title or path.stem
        n = 0
        for heading, section in split_sections(body):
            for piece in chunk_text(section, CHUNK_WORDS, OVERLAP_WORDS):
                if words(piece) < 15:  # skip tiny fragments
                    continue
                label = f"{title} > {heading}" if heading else title
                out.append({
                    "id": f"{rel}::{n}",
                    # Prefixing title/heading gives the embedding useful context
                    "text": f"{label}\n\n{piece}",
                    "meta": {"source": rel, "title": title},
                })
                n += 1
    return out


def main():
    import chromadb
    from chromadb.utils import embedding_functions

    chunks = build_chunks(DATA_DIR)
    # with open("chunks", "w") as file:
        # file.write('********'.join(str(d) for d in chunks))
    if not chunks:
        raise SystemExit(f"No .md files found in {DATA_DIR}/ - copy the docs there first.")
    files = len({c["meta"]["source"] for c in chunks})
    print(f"{files} files -> {len(chunks)} chunks "
          f"(CHUNK_WORDS={CHUNK_WORDS}, OVERLAP_WORDS={OVERLAP_WORDS})")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)  # fresh index each run so experiments are clean
    except Exception:
        pass
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    col = client.create_collection(COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), 128):
        batch = chunks[i:i + 128]
        col.add(ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["meta"] for c in batch])
        print(f"  embedded {min(i + 128, len(chunks))}/{len(chunks)}")
    print("Ingestion Complete")


if __name__ == "__main__":
    main()
