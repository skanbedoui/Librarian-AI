"""
Generation + service evaluation for one configuration (baseline or optimised).

For every question we run the FULL pipeline, then an LLM judge checks:
  correct            - does the answer match the gold answer?
  faithful           - is every claim supported by the sources the bot was given?
  citations_accurate - do the cited [n] sources really say what they are cited for?
For unanswerable questions, "correct" means the bot said "I don't know".

We also record end-to-end latency (p50/p95) and cost per request, and the judge's own cost and
latency per item.

Harness rules reused from Project 1:
  * one row per item, ALWAYS: if something fails, the item still gets a row with status "error";
  * the judge answers in a fixed JSON schema, parsed with P1's parse_json_response.

Run:  python -m eval.eval_generation --config baseline
      python -m eval.eval_generation --config optimised
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import DATA_DIR, RESULTS_DIR, BASELINE, OPTIMISED
from rag.llm import chat, Usage
from rag.pipeline import Librarian
from p1_harness.core import parse_json_response

CONFIGS = {"baseline": BASELINE, "optimised": OPTIMISED}
JUDGE_MODEL = "gpt-4.1-mini"
REQUESTS_PER_DAY = 200   # assumption for the cost model: ~200 questions/day today

JUDGE_SYSTEM = """You grade a Q&A bot that answers from numbered sources.
Given the question, the gold answer, the bot's answer and the sources it was shown, return JSON:
{"correct": true/false,            // bot answer contains the key fact(s) of the gold answer, no contradiction
 "faithful": true/false,           // every factual claim in the bot answer is supported by the sources
 "citations_accurate": true/false, // each [n] citation points to a source that supports that claim
 "reason": "one short sentence"}"""


def judge_answer(question, gold, answer, sources, usage=None):
    """One call to the LLM judge. Also used by eval/judge_reliability.py."""
    numbered = "\n\n".join(f"[{n}] {text}" for n, text in enumerate(sources, start=1))
    raw = chat(JUDGE_MODEL, JUDGE_SYSTEM,
               f"Question: {question}\nGold answer: {gold}\n\nBot answer: {answer}\n\n"
               f"Sources:\n{numbered}", usage, json_mode=True, max_tokens=200)
    try:
        v = parse_json_response(raw)
    except ValueError:
        return {"correct": False, "faithful": False, "citations_accurate": False, "reason": "judge error"}
    return {"correct": bool(v.get("correct")), "faithful": bool(v.get("faithful")),
            "citations_accurate": bool(v.get("citations_accurate")), "reason": str(v.get("reason", ""))}


def judge(q, r):
    """Grade one bot result. Returns the verdict + what the judge itself cost."""
    usage, t = Usage(), time.perf_counter()
    if r["is_idk"]:
        # Saying "I don't know" invents nothing, so it is faithful; it is correct only if
        # the question really was unanswerable. (No LLM call needed.)
        v = {"correct": not q["answerable"], "faithful": True, "citations_accurate": True,
             "reason": "bot said I don't know"}
    elif not q["answerable"]:
        v = {"correct": False, "faithful": False, "citations_accurate": False,
             "reason": "answered a question the documents cannot answer"}
    else:
        v = judge_answer(q["question"], q["answer"], r["answer"], [c["text"] for c in r["context"]], usage)
    v["verdict"] = "good" if v["correct"] else "bad"   # same label as Project 1
    v["judge_cost"] = usage.cost
    v["judge_latency"] = time.perf_counter() - t
    return v


def run_eval(settings, name, limit=None, save=True):
    """Run the full bot on the eval set with these settings; return the summary dict."""
    questions = json.loads((DATA_DIR / "eval_questions.json").read_text(encoding="utf-8"))[:limit]
    bot = Librarian()

    def run(q):
        row = {"id": q["id"], "question": q["question"], "gold": q["answer"], "answerable": q["answerable"]}
        try:
            r = bot.ask(q["question"], settings)
            row.update({"answer": r["answer"], "is_idk": r["is_idk"], "latency": r["latency"],
                        "cost": r["cost"], "cost_list": r["cost_list"],
                        "reranked": r["search_info"]["reranked"],
                        "sources": [{"file": c["file"], "page": c["page"], "text": c["text"]} for c in r["context"]],
                        **judge(q, r), "status": "ok"})
        except Exception as exc:   # P1 rule: keep one result row for every input item
            row.update({"status": "error", "error": repr(exc)})
        return row

    with ThreadPoolExecutor(4) as pool:   # same concurrency for every config -> fair comparison
        all_rows = list(pool.map(run, questions))
    rows = [r for r in all_rows if r["status"] == "ok"]
    errors = len(all_rows) - len(rows)
    if errors:
        print(f"WARNING: {errors} item(s) failed - see the 'error' field in the JSONL")

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["is_idk"]]
    lat = [r["latency"] for r in rows]
    cost = float(np.mean([r["cost"] for r in rows]))
    summary = {
        "config": name,
        "settings": settings.__dict__,
        "n_questions": len(rows),
        "n_errors": errors,
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
        # The judge itself (P1 "What to measure": cost and latency per judged item)
        "judge_cost_per_item_usd": float(np.mean([r["judge_cost"] for r in rows])),
        "judge_latency_p50_s": float(np.percentile([r["judge_latency"] for r in rows], 50)),
    }
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"generation_{name}.json"
        out.write_text(json.dumps({"summary": summary, "rows": all_rows}, indent=1, ensure_ascii=False),
                       encoding="utf-8")
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
