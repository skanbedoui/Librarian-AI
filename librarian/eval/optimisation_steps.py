"""
Week 8: apply the optimisations ONE AT A TIME (cumulatively) so we can see what each one
saved and what it cost in accuracy. Same test set, same concurrency, for every row.

Run:  python -m eval.optimisation_steps      (~10 min, ~$0.50)
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import BASELINE, RESULTS_DIR
from eval.eval_generation import run_eval

s0 = BASELINE
s1 = replace(s0, gen_model="gpt-4o-mini")          # 1. smaller answer model
s2 = replace(s1, n_context=3)                      # 2. 3 chunks instead of 5
s3 = replace(s2, rewrite_query=False)              # 3. drop the rewrite call
s4 = replace(s3, skip_rerank_if_confident=True)    # 4. skip reranker when search agrees (= OPTIMISED minus cache)
STEPS = [("0. baseline", s0), ("1. + gpt-4o-mini for answers", s1), ("2. + 3 chunks instead of 5", s2),
         ("3. + no query rewrite", s3), ("4. + skip reranker when confident", s4)]


def main():
    rows = []
    for name, settings in STEPS:
        s = run_eval(settings, name, save=False)
        rows.append({"step": name, **{k: s[k] for k in ("accuracy_overall", "latency_p50_s", "latency_p95_s",
                                                         "cost_per_request_usd", "cost_per_request_list_usd", "rerank_rate")}})
        print(rows[-1])
    base = rows[0]
    # "List-price cost" ignores OpenAI's prompt-cache discount, which varies from run to run
    # (each step re-sends prompts the previous step already sent, so later steps get more hits).
    md = ("| Step | Accuracy | p50 (s) | p95 (s) | List-price cost / request | vs baseline | Billed cost / request |\n"
          "|---|---|---|---|---|---|---|\n")
    for r in rows:
        md += (f"| {r['step']} | {r['accuracy_overall'] * 100:.1f}% | {r['latency_p50_s']:.2f} | "
               f"{r['latency_p95_s']:.2f} | ${r['cost_per_request_list_usd']:.6f} | "
               f"{(r['cost_per_request_list_usd'] / base['cost_per_request_list_usd'] - 1) * 100:+.0f}% | "
               f"${r['cost_per_request_usd']:.6f} |\n")
    (RESULTS_DIR / "optimisation_steps.json").write_text(json.dumps(rows, indent=2))
    (RESULTS_DIR / "optimisation_steps.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
