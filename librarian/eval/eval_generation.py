"""
Generation + service evaluation for one configuration (baseline or optimised).

For every question we run the FULL pipeline, then an LLM judge checks:
  correct            - does the answer match the gold answer?
  faithful           - is every claim supported by the sources the bot was given?
  citations_accurate - do the cited [n] sources really say what they are cited for?
For unanswerable questions, "correct" means the bot said "I don't know".

We also record end-to-end latency (p50/p95) and cost per request.

Run:  python -m eval.eval_generation --config baseline
      python -m eval.eval_generation --config optimised
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import DATA_DIR, RESULTS_DIR, BASELINE, OPTIMISED
from rag.llm import chat
from rag.pipeline import Librarian

CONFIGS = {"baseline": BASELINE, "optimised": OPTIMISED}
JUDGE_MODEL = "gpt-4.1-mini"
REQUESTS_PER_DAY = 200   # assumption for the cost model: ~200 questions/day today

JUDGE_SYSTEM = """You grade a Q&A bot that answers from numbered sources.
Given the question, the gold answer, the bot's answer and the sources it was shown, return JSON:
{"correct": true/false,            // bot answer contains the key fact(s) of the gold answer, no contradiction
 "faithful": true/false,           // every factual claim in the bot answer is supported by the sources
 "citations_accurate": true/false, // each [n] citation points to a source that supports that claim
 "reason": "one short sentence"}"""


def judge(q, r):
    if r["is_idk"]:
        # Saying "I don't know" invents nothing, so it is faithful; it is correct only if
        # the question really was unanswerable.
        return {"correct": not q["answerable"], "faithful": True, "citations_accurate": True,
                "reason": "bot said I don't know"}
    if not q["answerable"]:
        return {"correct": False, "faithful": False, "citations_accurate": False,
                "reason": "answered a question the guides cannot answer"}
    sources = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(r["context"], start=1))
    raw = chat(JUDGE_MODEL, JUDGE_SYSTEM,
               f"Question: {q['question']}\nGold answer: {q['answer']}\n\nBot answer: {r['answer']}\n\n"
               f"Sources:\n{sources}", json_mode=True, max_tokens=200)
    try:
        return json.loads(raw)
    except ValueError:
        return {"correct": False, "faithful": False, "citations_accurate": False, "reason": "judge error"}


def run_eval(settings, name, limit=None, save=True):
    """Run the full bot on the eval set with these settings; return the summary dict."""
    questions = json.loads((DATA_DIR / "eval_questions.json").read_text(encoding="utf-8"))[:limit]
    bot = Librarian()

    def run(q):
        r = bot.ask(q["question"], settings)
        j = judge(q, r)
        return {"id": q["id"], "question": q["question"], "gold": q["answer"], "answerable": q["answerable"],
                "answer": r["answer"], "is_idk": r["is_idk"], "latency": r["latency"], "cost": r["cost"], "cost_list": r["cost_list"],
                "reranked": r["search_info"]["reranked"], **j}

    with ThreadPoolExecutor(4) as pool:   # same concurrency for every config -> fair comparison
        rows = list(pool.map(run, questions))

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["is_idk"]]
    lat = [r["latency"] for r in rows]
    cost = float(np.mean([r["cost"] for r in rows]))
    summary = {
        "config": name,
        "settings": settings.__dict__,
        "n_questions": len(rows),
        "accuracy_overall": float(np.mean([r["correct"] for r in rows])),   # the Week-8 "accuracy"
        "correctness_answerable": float(np.mean([r["correct"] for r in ans])),
        "faithfulness": float(np.mean([r["faithful"] for r in answered])) if answered else None,
        "citation_accuracy": float(np.mean([r["citations_accurate"] for r in answered])) if answered else None,
        "idk_when_should": float(np.mean([r["is_idk"] for r in una])) if una else None,
        "idk_when_should_not": float(np.mean([r["is_idk"] for r in ans])),
        "latency_p50_s": float(np.percentile(lat, 50)),
        "latency_p95_s": float(np.percentile(lat, 95)),
        "cost_per_request_usd": cost,
        "cost_per_request_list_usd": float(np.mean([r["cost_list"] for r in rows])),
        "cost_per_month_today_usd": cost * REQUESTS_PER_DAY * 30,
        "cost_per_month_100x_usd": cost * REQUESTS_PER_DAY * 30 * 100,
        "rerank_rate": float(np.mean([r["reranked"] for r in rows])),
    }
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"generation_{name}.json"
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Saved -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, default="baseline")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = ap.parse_args()
    # A partial run (--limit) is for quick testing only, so it never overwrites saved results.
    summary = run_eval(CONFIGS[args.config], args.config, args.limit, save=args.limit is None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
