"""
STEP 2 (offline): Embed every chunk and save the index to disk.

Storage choice: a flat file (chunks.json + embeddings.npy).
With a few hundred chunks, a brute-force numpy dot product takes well under 1 ms,
so a vector database would add setup and deployment complexity for no speed gain.

Run:  python -m rag.index
"""
import json
import sys
import numpy as np

from .config import INDEX_DIR, CHUNK_WORDS, CHUNK_OVERLAP
from .ingest import load_pages, chunk_pages
from .llm import embed, Usage


def build_index(out_dir=INDEX_DIR, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP, verbose=True):
    chunks = chunk_pages(load_pages(), chunk_words, overlap)
    # We embed the section heading together with the text: it gives the vector extra context
    # ("Grading Scale" helps a chunk that is just a table of letters and numbers).
    usage = Usage()
    vectors = embed([f"{c['section']}\n{c['text']}" for c in chunks], usage)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks.json").write_text(json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "embeddings.npy", vectors)
    if verbose:
        print(f"Indexed {len(chunks)} chunks into {out_dir}  (embedding cost ${usage.cost:.4f})")
    return chunks, vectors


def load_index(index_dir=INDEX_DIR):
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(index_dir / "embeddings.npy")
    return chunks, vectors


if __name__ == "__main__":
    build_index()
    sys.exit(0)
