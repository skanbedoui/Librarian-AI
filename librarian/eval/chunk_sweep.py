"""
Chunk size is a decision to TEST, not a default.
Rebuilds the index with several chunk sizes and compares hybrid-search recall.

Run:  python -m eval.chunk_sweep        (costs < $0.01 in embeddings)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import ROOT, RESULTS_DIR
from rag.index import build_index
from rag.retrieve import Retriever
from eval.eval_retrieval import load_questions, evaluate

SETTINGS = [(100, 20), (200, 40), (400, 80)]   # (chunk words, overlap words)


def main():
    questions = load_questions()
    rows = []
    for words, overlap in SETTINGS:
        out = ROOT / "index_sweep" / f"w{words}"
        chunks, _ = build_index(out, words, overlap, verbose=False)
        r = evaluate(Retriever(out), questions, modes=["vector", "hybrid"])
        rows.append({"chunk_words": words, "overlap": overlap, "n_chunks": len(chunks),
                     "vector_r@5": r["vector"]["recall@5"], "hybrid_r@1": r["hybrid"]["recall@1"],
                     "hybrid_r@5": r["hybrid"]["recall@5"]})
    md = "| Chunk words | Overlap | # chunks | Vector R@5 | Hybrid R@1 | Hybrid R@5 |\n|---|---|---|---|---|---|\n"
    md += "".join(f"| {r['chunk_words']} | {r['overlap']} | {r['n_chunks']} | {r['vector_r@5']:.2f} | "
                  f"{r['hybrid_r@1']:.2f} | {r['hybrid_r@5']:.2f} |\n" for r in rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "chunk_sweep.json").write_text(json.dumps(rows, indent=2))
    (RESULTS_DIR / "chunk_sweep.md").write_text(md)
    print(md)
    print("Note: bigger chunks make page-level recall easier but send more tokens to the LLM.")


if __name__ == "__main__":
    main()
