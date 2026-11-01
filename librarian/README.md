# 📚 The Librarian: Q&A over MedTech's documents

A retrieval-augmented (RAG) assistant that answers questions about **MedTech** using
**14 of its public documents (127 pages)**: the two 2026-2027 student guides (rules, exams,
grades, absences, scholarships), the June 2026 school catalogue, and brochures for its masters
(Cyber Security, BIM & ERP, Digital Health & AI) and cybersecurity certifications. Every claim
comes **with a document + page citation**, and the bot says *"I don't know"* when the documents
don't have the answer.

CS496 AI Engineering · Project 2 · Fall 2026

**Live app:** `<add your Streamlit Cloud URL here>`

---

## Setup (one command)

```bash
pip install -r requirements.txt
```

Then create a file named `.env` next to this README containing your key
(see `.env.example`):

```
OPENAI_API_KEY=sk-...
```

## Run (one command)

```bash
streamlit run app.py
```

The index is already built and committed (`index/`), so the app works straight from a clean
clone.

**Adding a document:** put the PDF in `docs/` and restart the app. The index rebuilds itself
(about 10 seconds). Optionally give it a nicer title in `TITLES` in `rag/config.py`, and run
`python -m eval.make_questions --new-docs-only` so the test set covers it (review the new
questions by hand).

## Results

### Retrieval: Recall@k and ablation (167 answerable questions)

| Mode | R@1 | R@3 | R@5 | R@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| keyword (BM25) only | 0.86 | 0.94 | 0.98 | 0.99 | 0.91 | 4 | 9 |
| vector only | 0.75 | 0.94 | 0.96 | 0.99 | 0.84 | 209 | 424 |
| hybrid (RRF) | 0.83 | 0.97 | 0.99 | 0.99 | 0.90 | 212 | 300 |
| hybrid + reranker | **0.89** | **0.98** | **0.99** | 0.99 | **0.93** | 1027 | 1316 |

- **Hybrid** search fixes misses that each single method makes at k ≥ 3 (R@3: 0.94 / 0.94 → 0.97).
- The **reranker** puts the right passage *first* much more often (R@1 0.83 → 0.89,
  MRR 0.90 → 0.93), for about 0.8 s of extra latency.
- Keyword search beats vector search at R@1 partly because our test questions were generated
  from the passages and reuse their wording (see `postmortem.md` §4).

Chunk-size test (`eval/results/chunk_sweep.md`): 100 / 200 / 400 words gave hybrid R@1 of
0.71 / 0.83 / 0.85. We kept 200 as the balance between precision and the tokens sent to the
LLM (`design.md`).

### Generation and service (181 questions: 167 answerable + 14 unanswerable)

The app runs the **optimised** version. The baseline is our first version, kept to measure against.

| | Baseline | Optimised |
|---|---|---|
| Accuracy (overall) | 97.2% | 97.2% |
| Faithfulness / citation accuracy | 100% / 100% | 100% / 100% |
| "I don't know" when it should | 92.9% | 100% |
| p50 / p95 latency | 2.47 s / 3.24 s | **1.47 s / 2.50 s** |
| Cost per request (list price) | $0.00099 | **$0.00032 (−68%)** |
| Cost per request (billed, incl. prompt-cache discount) | $0.00096 | $0.00025 (−74%) |
| Billed cost per month at 200 req/day → at 100× | $5.75 → $575 | $1.51 → $151 |

Details are in `optimisation.md`. The judge is an LLM, so treat 100% faithfulness with care (`postmortem.md`).

## Repository map

| Path | What's inside |
|---|---|
| `app.py` | Streamlit app: chat, "How it works" explainer, live evaluation dashboard, 👍/👎 feedback |
| `rag/` | The pipeline: `ingest.py` (load + chunk every PDF in `docs/`), `index.py` (embed + save, rebuilds itself when `docs/` changes), `retrieve.py` (BM25, vectors, RRF, rerank), `pipeline.py` (cache → retrieve → rerank → generate), `llm.py` (OpenAI + cost tracking), `config.py` (all settings, document titles, baseline vs optimised) |
| `docs/` | The corpus (14 public PDFs, 127 pages) |
| `index/` | Pre-built index: `chunks.json` + `embeddings.npy` + `manifest.json` (which PDFs it was built from) |
| `data/eval_questions.json` | 181 eval questions with gold answers and source passages |
| `data/eval_review_log.json` | Questions we removed or fixed during manual review, and why |
| `eval/` | `eval_retrieval.py` (recall@k + ablation), `chunk_sweep.py`, `eval_generation.py` (correctness, faithfulness, citations, latency, cost), `optimisation_steps.py`, `cache_demo.py`, `make_questions.py`; results in `eval/results/` |
| `design.md` | Chunking, index choice, retrieval design, known failure cases |
| `optimisation.md` | Week 8 report with before/after numbers |
| `postmortem.md` | What went wrong and what we learned |
| `report.md` | The 6-page report |
| `GUIDE.md` | Plain-language walkthrough of every file + likely demo questions |

## Re-running the evaluation

```bash
python -m eval.eval_retrieval                     # retrieval + ablation        (~1 min)
python -m eval.chunk_sweep                        # chunk-size test             (~2 min)
python -m eval.eval_generation --config baseline  # full pipeline, judged       (~4 min, ~$0.25)
python -m eval.eval_generation --config optimised
python -m eval.optimisation_steps                 # each Week 8 change alone    (~10 min)
python -m eval.cache_demo
```

## Deploying (Streamlit Community Cloud, free)

1. Push this repo to GitHub (the `.gitignore` keeps `.env` out — **never commit the key**).
2. On https://share.streamlit.io → *New app* → pick the repo, main file `app.py`.
3. *Advanced settings → Secrets*: paste `OPENAI_API_KEY = "sk-..."`.
4. Deploy and put the URL at the top of this README.

## Contributions

| Member | Contributions |
|---|---|
| *Name 1* | *e.g. ingestion & chunking, chunk-size sweep* |
| *Name 2* | *e.g. retrieval, reranker, ablation* |
| *Name 3* | *e.g. generation prompt, evaluation set review, judge* |
| *Name 4* | *e.g. Streamlit app, deployment, Week 8 optimisation* |
