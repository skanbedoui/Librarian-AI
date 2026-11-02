"""Put the prompts back in the code, go back to the original judge, drop the extras."""
import textwrap


def read(p):
    return open(p, encoding="utf-8").read()


def write(p, s):
    open(p, "w", encoding="utf-8").write(s)


def sub(p, old, new):
    s = read(p)
    assert s.count(old) == 1, (p, old[:70])
    write(p, s.replace(old, new))


def py_concat(text, indent="    "):
    """Exact text as ("..." "...") pieces, so the prompt is byte-for-byte the same."""
    parts = textwrap.wrap(text, 88, drop_whitespace=False, break_long_words=False)
    assert "".join(parts) == text
    return "(\n" + "\n".join(indent + repr(x).replace("'", '"') if '"' not in x else indent + repr(x) for x in parts) + "\n)"


answer = read("prompts/answer_prompt.md")
rewrite = read("prompts/rewrite_prompt.md")
rerank = read("prompts/rerank_prompt.md")
judge = read("prompts/judge_prompt_v1.md")
compare = read("prompts/judge_prompt_compare.md")
assert '"""' not in answer + judge + compare and "\\" not in answer + judge + compare

# ---- config: no prompt files any more
p = "rag/config.py"
sub(p, '''PROMPTS_DIR = ROOT / "prompts"    # every prompt lives in a file (versioned, see prompts/CHANGELOG.md)


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")

''', "")

# ---- pipeline: prompts back in the code
p = "rag/pipeline.py"
sub(p, "from .config import Settings, OPTIMISED, load_prompt", "from .config import Settings, OPTIMISED")
sub(p, 'REWRITE_SYSTEM = load_prompt("rewrite_prompt.md")', "REWRITE_SYSTEM = " + py_concat(rewrite))
sub(p, '''ANSWER_SYSTEM = load_prompt("answer_prompt.md")
assert IDK in ANSWER_SYSTEM, "answer_prompt.md must contain the exact I-don't-know sentence"''',
    'ANSWER_SYSTEM = """' + answer + '"""\nassert IDK in ANSWER_SYSTEM')

# ---- retrieve: rerank prompt back in the code
p = "rag/retrieve.py"
sub(p, "from .config import INDEX_DIR, load_prompt", "from .config import INDEX_DIR")
sub(p, 'RERANK_SYSTEM = load_prompt("rerank_prompt.md")', "RERANK_SYSTEM = " + py_concat(rerank))

# ---- eval_generation: original judge (v1), judge sees the text only, no per-run JSONL
p = "eval/eval_generation.py"
s = read(p)
s = s.replace('''  * every run is saved as per-item JSONL in eval/results/runs/<config>_<timestamp>.jsonl
    (input, output, score, reason), so old runs are never lost;
''', "")
s = s.replace("from datetime import datetime, timezone\n", "")
s = s.replace("import os\n", "")
s = s.replace("from rag.config import DATA_DIR, RESULTS_DIR, BASELINE, OPTIMISED, load_prompt",
              "from rag.config import DATA_DIR, RESULTS_DIR, BASELINE, OPTIMISED")
s = s.replace("from rag.pipeline import Librarian, format_source", "from rag.pipeline import Librarian")
start = s.index("# v3 is the official judge")
end = s.index("JUDGE_SYSTEM = load_prompt(JUDGE_PROMPT)") + len("JUDGE_SYSTEM = load_prompt(JUDGE_PROMPT)")
s = s[:start] + 'JUDGE_SYSTEM = """' + judge + '"""' + s[end:]
s = s.replace("usage, json_mode=True, max_tokens=400)", "usage, json_mode=True, max_tokens=200)")
s = s.replace('''    out = {"correct": bool(v.get("correct")), "faithful": bool(v.get("faithful")),
           "citations_accurate": bool(v.get("citations_accurate")), "reason": str(v.get("reason", ""))}
    if "unsupported_claims" in v:                        # judge v2 lists them explicitly
        out["unsupported_claims"] = v["unsupported_claims"]
    return out''', '''    return {"correct": bool(v.get("correct")), "faithful": bool(v.get("faithful")),
            "citations_accurate": bool(v.get("citations_accurate")), "reason": str(v.get("reason", ""))}''')
s = s.replace('[format_source(c) for c in r["context"]], usage)', '[c["text"] for c in r["context"]], usage)')
s = s.replace('''        "judge_prompt": JUDGE_PROMPT,
''', "")
start = s.index('''        runs = RESULTS_DIR / "runs"''')
end = s.index('''        print(f"Saved -> {out} and {run_file}")''')
s = s[:start] + s[end:].replace('''print(f"Saved -> {out} and {run_file}")''', '''print(f"Saved -> {out}")''', 1)
s = s.replace('''out = RESULTS_DIR / f"generation_{name}.json"          # latest run (read by the app)''',
              '''out = RESULTS_DIR / f"generation_{name}.json"''')
for bad in ("load_prompt", "JUDGE_PROMPT", "format_source", "runs", "datetime"):
    assert bad not in s, bad
write(p, s)

# ---- judge_reliability: compare prompt in the code, remove the fabrication test
p = "eval/judge_reliability.py"
s = read(p)
s = s.replace('''  4. Fabrication test        (new in P2) an invented, cited claim is added to a correct answer.
                             A good judge must mark it NOT faithful. This checks our 100% faithfulness.
''', "")
s = s.replace("(tests 2-4 + test 1 if labels exist)", "(tests 2-3 + test 1 if labels exist)")
s = s.replace("from rag.config import DATA_DIR, RESULTS_DIR, OPTIMISED, load_prompt", "from rag.config import DATA_DIR, RESULTS_DIR")
s = s.replace("from rag.pipeline import Librarian, format_source\n", "")
s = s.replace("from eval.eval_generation import judge_answer, JUDGE_MODEL, JUDGE_PROMPT",
              "from eval.eval_generation import judge_answer, JUDGE_MODEL")
s = s.replace('COMPARE_SYSTEM = load_prompt("judge_prompt_compare.md")', 'COMPARE_SYSTEM = """' + compare + '"""')
start = s.index("# Invented claims (none of these appear")
end = s.index("def load_rows(name):")
s = s[:start] + s[end:]
start = s.index("# ------------------------------------------------------------------ 4. fabrication test")
end = s.index("# ------------------------------------------------------------------ report")
s = s[:start] + s[end:]
s = s.replace('''    h, p, v, f = rep["human_agreement"], rep["position_bias"], rep["verbosity_bias"], rep["fabrication"]''',
              '''    h, p, v = rep["human_agreement"], rep["position_bias"], rep["verbosity_bias"]''')
s = s.replace('''          f"Judge: `{JUDGE_MODEL}` with `prompts/{JUDGE_PROMPT}`. Test cost: ${rep['cost_usd']:.3f}.\\n",''',
              '''          f"Judge: `{JUDGE_MODEL}` (JUDGE_SYSTEM in eval/eval_generation.py). Test cost: ${rep['cost_usd']:.3f}.\\n",''')
start = s.index('''           f"- Became 'correct' after adding polite filler:''')
end = s.index("    return \"\".join(md)")
s = s[:start] + '''           f"- Became 'correct' after adding polite filler: **{v['flip_count']}/{v['items']} ({v['flip_rate_pct']}%)**\\n"]
''' + s[end:]
s = s.replace('''    rep["fabrication"] = fabrication(usage)
    print("4. fabrication catch rate:", rep["fabrication"]["catch_rate_pct"], "%")
''', "")
start = s.index('''    rep["judge_prompt"] = JUDGE_PROMPT''')
end = s.index("    print(\"\\n\" + md)")
s = s[:start] + '''    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "judge_reliability.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(rep)
    (RESULTS_DIR / "judge_reliability.md").write_text(md, encoding="utf-8")
''' + s[end:]
for bad in ("load_prompt", "JUDGE_PROMPT", "format_source", "fabricat", "Librarian", "OPTIMISED"):
    assert bad not in s, bad
write(p, s)
print("trimmed")
