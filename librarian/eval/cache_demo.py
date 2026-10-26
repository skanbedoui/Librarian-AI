"""
The eval set has no repeated questions, so it cannot show what the answer cache saves.
This script asks 20 questions twice with the OPTIMISED settings and compares.

Run:  python -m eval.cache_demo
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import DATA_DIR, RESULTS_DIR, OPTIMISED
from rag.pipeline import Librarian


def main():
    questions = json.loads((DATA_DIR / "eval_questions.json").read_text(encoding="utf-8"))[:20]
    bot = Librarian()
    first = [bot.ask(q["question"], OPTIMISED) for q in questions]
    second = [bot.ask(q["question"], OPTIMISED) for q in questions]
    out = {
        "first_time_mean_latency_s": float(np.mean([r["latency"] for r in first])),
        "repeat_mean_latency_s": float(np.mean([r["latency"] for r in second])),
        "first_time_mean_cost_usd": float(np.mean([r["cost"] for r in first])),
        "repeat_mean_cost_usd": float(np.mean([r["cost"] for r in second])),
        "repeat_hits": sum(r["cached"] for r in second),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "cache_demo.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
