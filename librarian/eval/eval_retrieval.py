"""
Retrieval evaluation + ablation.

Question: "Is the right passage in the top k results?"  (Recall@k)
We test the 4 search modes, so we can see what hybrid search and the reranker each add.

A hit = a retrieved chunk comes from the same document AND page as the gold passage
(or is a near-copy of it in the other guide - see is_hit).
(Matching on page, not chunk id, keeps the metric fair when chunk size changes.)

Only answerable questions are used. No query rewriting here: we test search alone.

Run:  python -m eval.eval_retrieval
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import DATA_DIR, RESULTS_DIR, INDEX_DIR
from rag.retrieve import Retriever
from rag.llm import Usage

MODES = ["keyword", "vector", "hybrid", "hybrid+rerank"]
KS = [1, 3, 5, 10]


def load_questions():
    qs = json.loads((DATA_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    return [q for q in qs if q["answerable"]]


def is_hit(chunk, q):
    """Same document + page as the gold passage, OR (because the two guides copy many
    paragraphs word for word) a chunk that contains most of the gold passage's words."""
    if chunk["file"] == q["file"] and chunk["page"] == q["page"]:
        return True
    gold, got = set(q["passage"].lower().split()), set(chunk["text"].lower().split())
    return len(gold & got) / len(gold) >= 0.7


def first_hit_rank(retriever, ranked, q):
    for rank, (idx, _) in enumerate(ranked, start=1):
        if is_hit(retriever.chunks[idx], q):
            return rank
    return None


def evaluate(retriever, questions, modes=MODES, top_k=10, workers=6):
    results = {}
    for mode in modes:
        def run(q):
            usage = Usage()
            t = time.perf_counter()
            ranked, _ = retriever.search(q["question"], mode, top_k, usage=usage)
            return first_hit_rank(retriever, ranked, q), time.perf_counter() - t, usage.cost

        with ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(run, questions))
        ranks = [r for r, _, _ in rows]
        lat = [l for _, l, _ in rows]
        results[mode] = {
            **{f"recall@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in KS},
            "mrr": float(np.mean([1 / r if r else 0 for r in ranks])),
            "latency_p50_ms": float(np.percentile(lat, 50) * 1000),
            "latency_p95_ms": float(np.percentile(lat, 95) * 1000),
            "cost_per_query": float(np.mean([c for _, _, c in rows])),
        }
        print(mode, {k: round(v, 3) for k, v in results[mode].items()})
    return results


def to_markdown(results):
    head = "| Mode | " + " | ".join(f"R@{k}" for k in KS) + " | MRR | p50 ms | p95 ms |\n"
    head += "|---" * (len(KS) + 4) + "|\n"
    for mode, r in results.items():
        head += (f"| {mode} | " + " | ".join(f"{r[f'recall@{k}']:.2f}" for k in KS)
                 + f" | {r['mrr']:.2f} | {r['latency_p50_ms']:.0f} | {r['latency_p95_ms']:.0f} |\n")
    return head


def main():
    questions = load_questions()
    print(f"{len(questions)} answerable questions")
    results = evaluate(Retriever(INDEX_DIR), questions)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval.json").write_text(json.dumps(results, indent=2))
    md = to_markdown(results)
    (RESULTS_DIR / "retrieval.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
