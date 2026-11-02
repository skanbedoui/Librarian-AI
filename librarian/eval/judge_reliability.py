"""
Judge reliability report: the Project 1 checks, run on the Project 2 judge.

"A judge you have not tested is just another model with an opinion." (Project 1 brief)

  1. Agreement with humans   two team members label answers good/bad; we report labeller-vs-labeller
                             and judge-vs-humans agreement with P1's compute_cohen_kappa.
  2. Position bias           (P1 swap test) the compare judge sees the baseline and the optimised
                             answer to the same question, then again in the other order.
                             Biased = it picks the same POSITION both times.
  3. Verbosity bias          (P1 padding test) clearly wrong answers are padded with polite filler.
                             Biased = the padded version is suddenly graded correct.

Run:  python -m eval.judge_reliability                       (tests 2-3 + test 1 if labels exist)
      python -m eval.judge_reliability --make-labelling-set  (once: writes the sheet for test 1)
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import DATA_DIR, RESULTS_DIR
from rag.llm import chat, Usage
from p1_harness.core import parse_json_response, compute_cohen_kappa, human_consensus_label
from eval.eval_generation import judge_answer, JUDGE_MODEL

COMPARE_SYSTEM = """You compare two answers from a Q&A bot about MedTech documents. Both answers were written from numbered sources.
Given the question, the gold answer and the two candidate answers, decide which answer is better:
more correct first (matches the gold answer, no contradiction), then more complete, then clearer.

Return ONLY a JSON object matching this schema:

{
  "winner": "A" | "B" | "tie",
  "reason": "short explanation, <= 30 words"
}

Do not include Markdown fences or extra text outside the JSON object.
"""
LABEL_SET = DATA_DIR / "judge_labelling_set.jsonl"      # what the two labellers see and fill in
LABEL_KEY = DATA_DIR / "judge_labelling_key.json"       # the judge's verdicts, hidden from labellers
N = 30                                                  # items per bias test (P1 used 25-30)

# Same idea as P1's PADDING_TEXT: longer and more polite, but no new information.
PADDING_TEXT = (" I hope this helps! For more details, students are encouraged to consult the official"
                " MedTech documents and to contact the administration, which will be happy to answer"
                " any further questions about this topic.")

def load_rows(name):
    data = json.loads((RESULTS_DIR / f"generation_{name}.json").read_text(encoding="utf-8"))
    return {r["id"]: r for r in data["rows"] if r.get("status", "ok") == "ok"}


def questions():
    return json.loads((DATA_DIR / "eval_questions.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 1. agreement with humans
def make_labelling_set(n=60, seed=0):
    """Pick 60 answers from the latest optimised run: every one the judge called bad, plus random good ones.
    The sheet does NOT show the judge's verdict, so the labellers are not influenced."""
    rows = list(load_rows("optimised").values())
    qs = {q["id"]: q for q in questions()}
    bad = [r for r in rows if not r["correct"]]
    good = [r for r in rows if r["correct"]]
    random.Random(seed).shuffle(good)
    picked = bad + good[:n - len(bad)]
    random.Random(seed).shuffle(picked)
    with open(LABEL_SET, "w", encoding="utf-8") as f:
        for r in picked:
            q = qs[r["id"]]
            f.write(json.dumps({
                "id": r["id"], "question": r["question"], "gold_answer": r["gold"],
                "gold_source": f"{q['file']} p.{q['page']}" if q.get("file") else "none (unanswerable)",
                "bot_answer": r["answer"], "label_1": None, "label_2": None,
            }, ensure_ascii=False) + "\n")
    LABEL_KEY.write_text(json.dumps({str(r["id"]): r["verdict"] if "verdict" in r else
                                     ("good" if r["correct"] else "bad") for r in picked}, indent=1))
    print(f"Wrote {len(picked)} items to {LABEL_SET} ({len(bad)} the judge called bad). "
          f"Two people fill label_1 / label_2 with 'good' or 'bad' (see data/labelling_guide.md).")


def human_agreement():
    if not LABEL_SET.exists():
        return {"status": "no labelling set yet: run with --make-labelling-set"}
    items = [json.loads(l) for l in LABEL_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    key = json.loads(LABEL_KEY.read_text())
    done = [i for i in items if i["label_1"] in ("good", "bad") and i["label_2"] in ("good", "bad")]
    if not done:
        return {"status": f"waiting for human labels (0/{len(items)} items labelled twice)"}
    l1, l2 = [i["label_1"] for i in done], [i["label_2"] for i in done]
    humans = [human_consensus_label(a, b) for a, b in zip(l1, l2)]
    judge = [key[str(i["id"])] for i in done]
    lab_pct, lab_kappa = compute_cohen_kappa(l1, l2)
    jud_pct, jud_kappa = compute_cohen_kappa(humans, judge)
    return {"status": "ok", "items_labelled": len(done), "items_total": len(items),
            "labeller_agreement_pct": lab_pct, "labeller_kappa": lab_kappa,
            "judge_human_agreement_pct": jud_pct, "judge_human_kappa": jud_kappa,
            "disagreements": [{"id": i["id"], "question": i["question"], "humans": h, "judge": j}
                              for i, h, j in zip(done, humans, judge) if h != j]}


# ------------------------------------------------------------------ 2. position bias (P1 swap test)
def compare(question, gold, a, b, usage):
    raw = chat(JUDGE_MODEL, COMPARE_SYSTEM,
               f"Question: {question}\nGold answer: {gold}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}",
               usage, json_mode=True, max_tokens=120)
    try:
        w = str(parse_json_response(raw).get("winner", "tie")).upper()
    except ValueError:
        w = "TIE"
    return w if w in ("A", "B", "TIE") else "TIE"


def position_bias(usage, seed=0):
    base, opt = load_rows("baseline"), load_rows("optimised")
    pairs = [(base[i], opt[i]) for i in base if i in opt and base[i]["answer"] != opt[i]["answer"]]
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:N]

    def run(pair):
        b, o = pair
        w1 = compare(b["question"], b["gold"], b["answer"], o["answer"], usage)   # A = baseline
        w2 = compare(b["question"], b["gold"], o["answer"], b["answer"], usage)   # A = optimised
        return {"id": b["id"], "order1_AB": w1, "order2_BA": w2,
                "position_biased": w1 == w2 and w1 in ("A", "B"),             # same letter = followed position
                "consistent": (w1, w2) in (("A", "B"), ("B", "A"), ("TIE", "TIE"))}

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(run, pairs))
    flips = sum(r["position_biased"] for r in rows)
    # Which version does the judge prefer, when it is consistent?
    pref_base = sum(r["order1_AB"] == "A" and r["order2_BA"] == "B" for r in rows)
    pref_opt = sum(r["order1_AB"] == "B" and r["order2_BA"] == "A" for r in rows)
    return {"pairs": len(rows), "flip_count": flips, "flip_rate_pct": round(100 * flips / len(rows), 1),
            "consistent_count": sum(r["consistent"] for r in rows),
            "prefers_baseline": pref_base, "prefers_optimised": pref_opt, "details": rows}


# ------------------------------------------------------------------ 3. verbosity bias (P1 padding test)
def verbosity_bias(usage, seed=0):
    """Wrong answer = the gold answer of ANOTHER question. Judge it plain, then padded."""
    qs = [q for q in questions() if q["answerable"]]
    random.Random(seed).shuffle(qs)
    items = [(qs[i], qs[(i + len(qs) // 2) % len(qs)]) for i in range(N)]   # far-apart partner question

    def run(item):
        q, other = item
        wrong = f"{other['answer']} [1]"
        plain = judge_answer(q["question"], q["answer"], wrong, [q["passage"]], usage)
        padded = judge_answer(q["question"], q["answer"], wrong + PADDING_TEXT, [q["passage"]], usage)
        return {"id": q["id"], "wrong_answer": wrong, "plain_correct": plain["correct"],
                "padded_correct": padded["correct"],
                "verbosity_biased": (not plain["correct"]) and padded["correct"]}

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(run, items))
    flips = sum(r["verbosity_biased"] for r in rows)
    return {"items": len(rows), "plain_wrongly_accepted": sum(r["plain_correct"] for r in rows),
            "flip_count": flips, "flip_rate_pct": round(100 * flips / len(rows), 1), "details": rows}


# ------------------------------------------------------------------ report
def to_markdown(rep):
    h, p, v = rep["human_agreement"], rep["position_bias"], rep["verbosity_bias"]
    md = ["# Judge Reliability Report (Project 1 harness checks)\n",
          f"Judge: `{JUDGE_MODEL}` (JUDGE_SYSTEM in eval/eval_generation.py). Test cost: ${rep['cost_usd']:.3f}.\n",
          "\n## 1. Agreement with humans\n"]
    if h["status"] == "ok":
        md += [f"- Items labelled by two people: {h['items_labelled']}/{h['items_total']}\n",
               f"- Labeller vs labeller: {h['labeller_agreement_pct']}% (Cohen's kappa {h['labeller_kappa']})\n",
               f"- Judge vs human consensus: {h['judge_human_agreement_pct']}% (Cohen's kappa {h['judge_human_kappa']})\n",
               f"- Disagreements: {len(h['disagreements'])} (listed in judge_reliability.json)\n"]
    else:
        md += [f"- **Not measured yet:** {h['status']}.\n"]
    md += ["\n## 2. Position bias (swap test)\n",
           f"- Pairs (baseline vs optimised answer to the same question): {p['pairs']}\n",
           f"- Verdict followed the position instead of the answer: **{p['flip_count']}/{p['pairs']} "
           f"({p['flip_rate_pct']}%)**\n",
           f"- Consistent in both orders: {p['consistent_count']}/{p['pairs']} "
           f"(prefers baseline {p['prefers_baseline']}, prefers optimised {p['prefers_optimised']}, rest ties)\n",
           "\n## 3. Verbosity bias (padding test)\n",
           f"- Clearly wrong answers (the gold answer of another question): {v['items']}\n",
           f"- Wrongly accepted even without padding: {v['plain_wrongly_accepted']}/{v['items']}\n",
           f"- Became 'correct' after adding polite filler: **{v['flip_count']}/{v['items']} ({v['flip_rate_pct']}%)**\n"]
    return "".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-labelling-set", action="store_true")
    args = ap.parse_args()
    if args.make_labelling_set:
        make_labelling_set()
        return
    usage = Usage()
    rep = {"human_agreement": human_agreement()}
    print("1. human agreement:", rep["human_agreement"]["status"])
    rep["position_bias"] = position_bias(usage)
    print("2. position bias flip rate:", rep["position_bias"]["flip_rate_pct"], "%")
    rep["verbosity_bias"] = verbosity_bias(usage)
    print("3. verbosity bias flip rate:", rep["verbosity_bias"]["flip_rate_pct"], "%")
    rep["cost_usd"] = usage.cost
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "judge_reliability.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(rep)
    (RESULTS_DIR / "judge_reliability.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
