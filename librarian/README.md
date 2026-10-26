# 📚 The Librarian: Q&A over the MedTech student guides

A retrieval-augmented (RAG) assistant that answers students' questions about the
**MedTech 2026-2027 student guides** (Pre-Engineering/Engineering and Licence/Master),
**with a page citation for every claim**, and says *"I don't know"* when the guides don't have
the answer.

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
clone. To rebuild it from the PDFs: `python -m rag.index`.

## Results

### Retrieval: Recall@k and ablation (108 answerable questions)

| Mode | R@1 | R@3 | R@5 | R@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| keyword (BM25) only | 0.84 | 0.95 | 0.98 | 1.00 | 0.90 | 3 | 8 |
| vector only | 0.75 | 0.94 | 0.96 | 1.00 | 0.84 | 249 | 1017 |
| hybrid (RRF) | 0.81 | 0.98 | 1.00 | 1.00 | 0.89 | 255 | 474 |
| hybrid + reranker | **0.88** | **1.00** | **1.00** | 1.00 | **0.94** | 1267 | 1874 |

- **Hybrid** fixes the misses of each single method at k ≥ 3 (R@5: 0.98 / 0.96 → 1.00).
- The **reranker** puts the right passage *first* much more often (R@1 0.81 → 0.88, MRR 0.89 → 0.94),
  for about 1 s of extra latency.
- Keyword search beating vector search at R@1 is partly an artifact of our generated questions,
  which reuse the passages' wording (see `postmortem.md` §4).

Chunk size test (`eval/results/chunk_sweep.md`): 100 / 200 / 400 words → hybrid R@1 0.65 / 0.80 / 0.87.
We kept 200 as the balance between precision and tokens sent to the LLM (`design.md`).

### Generation and service (122 questions: 108 answerable + 14 unanswerable)

| | Baseline (Week 7) | Optimised (Week 8) |
|---|---|---|
| Accuracy (overall) | 99.2% | 99.2% |
| Faithfulness / citation accuracy | 100% / 100% | 100% / 100% |
| "I don't know" when it should | 92.9% | 100% |
| p50 / p95 latency | 3.19 s / 4.88 s | **1.62 s / 3.20 s** |
| Cost per request (list price) | $0.00097 | **$0.00030 (−69%)** |
| Cost per request (billed, incl. prompt-cache discount) | $0.00074 | $0.00022 (−71%) |
| Billed cost per month at 200 req/day → at 100× | $4.45 → $445 | $1.29 → $129 |

Details: `optimisation.md`. The judge is an LLM, so treat 100% faithfulness with care (`postmortem.md`).

## Repository map

| Path | What's inside |
|---|---|
| `app.py` | Streamlit app: chat, "How it works" explainer, live evaluation dashboard, 👍/👎 feedback |
| `rag/` | The pipeline: `ingest.py` (load + chunk), `index.py` (embed + save), `retrieve.py` (BM25, vectors, RRF, rerank), `pipeline.py` (rewrite → retrieve → rerank → generate), `llm.py` (OpenAI + cost tracking), `config.py` (all settings, baseline vs optimised) |
| `docs/` | The corpus (2 public PDFs) |
| `index/` | Pre-built index: `chunks.json` + `embeddings.npy` |
| `data/eval_questions.json` | 122 eval questions with gold answers and source passages |
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
python -m eval.eval_generation --config baseline  # full pipeline, judged       (~3 min, ~$0.15)
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
