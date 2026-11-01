"""
Build the evaluation set: data/eval_questions.json

For every chunk with enough content, an LLM writes ONE question a real student could ask,
plus the answer, using only that chunk. The chunk is saved as the "source passage".
We then add hand-written questions the guides CANNOT answer, to test "I don't know".

IMPORTANT: generated questions must be reviewed by a human (see postmortem.md).

Run:  python -m eval.make_questions --new-docs-only   (after adding PDFs: adds questions for the
                                                        new documents, keeps the reviewed ones)
      python -m eval.make_questions                   (start over: OVERWRITES the reviewed file)
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import DATA_DIR
from rag.index import load_index
from rag.llm import chat

SYSTEM = """You write evaluation questions for a Q&A bot about a university's documents
(student guides, program catalogues, brochures).
Given ONE passage, write a question that a real student might ask and that this passage answers.
- Phrase it naturally, like a student would (no "according to the passage").
- The answer must be a specific fact found in the passage (a number, rule, condition, name...).
- If the passage has no useful fact (e.g. a title page or a mission statement), set "skip": true.
Reply in JSON: {"skip": false, "question": "...", "answer": "short answer"}"""

# Questions the guides do not answer. The correct behaviour is "I don't know".
UNANSWERABLE = [
    "What is the wifi password on campus?",
    "Who won the 2022 football World Cup?",
    "How much does a parking permit cost at MedTech?",
    "What is the name of the professor who teaches CS496?",
    "Can I bring my dog to class?",
    "What is the exchange rate between the dinar and the euro?",
    "How do I apply for a Tunisian driving licence?",
    "What are the opening hours of the student gym?",
    "Which companies hired the most MedTech graduates last year?",
    "What is the best laptop to buy for engineering courses?",
    "How many students live in the MedTech dormitory?",
    "What is the recipe for couscous?",
    "Is there a MedTech campus in Paris?",
    "What is the menu at the student restaurant this week?",
]


def make_one(chunk):
    if len(chunk["text"].split()) < 60:
        return None
    raw = chat("gpt-4.1-mini", SYSTEM, f"Section: {chunk['section']}\n\nPassage:\n{chunk['text']}",
               json_mode=True, max_tokens=200)
    try:
        q = json.loads(raw)
    except ValueError:
        return None
    if q.get("skip") or not q.get("question"):
        return None
    return {"question": q["question"], "answer": q["answer"], "answerable": True,
            "source_chunk_id": chunk["id"], "file": chunk["file"],
            "page": chunk["page"], "passage": chunk["text"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-docs-only", action="store_true",
                    help="only write questions for documents that have none yet, and keep the existing ones")
    args = ap.parse_args()
    path = DATA_DIR / "eval_questions.json"
    chunks, _ = load_index()

    if args.new_docs_only:
        items = json.loads(path.read_text(encoding="utf-8"))
        covered = {q.get("file") for q in items}
        new_chunks = [c for c in chunks if c["file"] not in covered]
        with ThreadPoolExecutor(8) as pool:
            new = [q for q in pool.map(make_one, new_chunks) if q]
        next_id = max(q["id"] for q in items) + 1
        for n, q in enumerate(new):
            q["id"] = next_id + n
        items += new
        print(f"Added {len(new)} questions for {len({c['file'] for c in new_chunks})} new documents")
    else:
        with ThreadPoolExecutor(8) as pool:
            items = [q for q in pool.map(make_one, chunks) if q]
        items += [{"question": q, "answer": "I don't know", "answerable": False} for q in UNANSWERABLE]
        random.Random(0).shuffle(items)
        for n, item in enumerate(items):
            item["id"] = n
    DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(items)} questions ({sum(i['answerable'] for i in items)} answerable)")


if __name__ == "__main__":
    main()
