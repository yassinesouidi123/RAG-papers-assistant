import json
import os

import numpy as np

from src.config import DATA_DIR, INDEX_DIR
from src.loader import load_pdfs
from src.chunker import chunk_pages
from src.embedder import embed
from src.retriever import search
from src.generator import generate


def build_or_load_index():
    """Compute chunks and embeddings once, then reuse them from disk."""
    emb_path = os.path.join(INDEX_DIR, "embeddings.npy")
    chunks_path = os.path.join(INDEX_DIR, "chunks.json")

    if os.path.exists(emb_path) and os.path.exists(chunks_path):
        print("Loading index from disk...")
        with open(chunks_path) as f:
            return json.load(f), np.load(emb_path)

    print("Building index (first run)...")
    chunks = chunk_pages(load_pdfs(DATA_DIR))
    embeddings = embed([c["text"] for c in chunks])
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(emb_path, embeddings)
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)
    return chunks, embeddings


def main():
    chunks, embeddings = build_or_load_index()
    print(f"{len(chunks)} chunks ready. Ask a question (or 'quit').\n")

    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        results = search(question, chunks, embeddings)
        print("\n" + generate(question, results))
        print("\nSources:", ", ".join(f"{r['source']} p.{r['page']}" for r in results), "\n")


if __name__ == "__main__":
    main()