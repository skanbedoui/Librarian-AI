# Week 8: making it cheaper and faster

**Target:** cut cost per request **or** p95 latency by 50%, losing at most 2 points of accuracy,
measured before and after on the same test set.

**Result: cost −68%, p50 latency −41%, p95 latency −23%, accuracy unchanged (97.2% → 97.2%).**

Test set: `data/eval_questions.json`, with 181 questions over the 14 documents (167 answerable,
14 unanswerable). "Accuracy" = the share of questions handled correctly (right answer, or
"I don't know" when the documents don't have the answer). Every row uses the same questions,
the same judge and the same concurrency (4 threads).

The app now always runs the optimised version. The baseline is kept in `rag/config.py` only so
we can measure against it.

## Before / after

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| Accuracy (all 181) | 97.2% | 97.2% | **±0 pts** |
| Correct on answerable | 97.6% | 97.0% | −0.6 pts (1 question) |
| "I don't know" when it should | 92.9% | 100.0% | +7.1 pts (1 question) |
| "I don't know" when it shouldn't | 0.6% | 0.6% | — |
| Faithfulness | 100% | 100% | — |
| Citation accuracy | 100% | 100% | — |
| p50 latency | 2.47 s | 1.47 s | **−41%** |
| p95 latency | 3.24 s | 2.50 s | −23% |
| Cost / request (list price) | $0.000991 | $0.000317 | **−68%** |
| Cost / request (billed, with prompt-cache discount) | $0.000958 | $0.000251 | **−74%** |

*Source: `eval/results/generation_baseline.json` and `eval/results/generation_optimised.json`.
The same comparison on the first corpus (2 guides only) gave −69% cost at ±0 accuracy; see
`eval/results/history/guides_only/`.*

## What each change did

The changes were applied one after the other (`python -m eval.optimisation_steps`):

| Step | Accuracy | p50 (s) | p95 (s) | List-price cost | vs baseline |
|---|---|---|---|---|---|
| 0. Baseline | 97.2% | 2.42 | 3.17 | $0.000993 | — |
| 1. + `gpt-4o-mini` writes the answers (was `gpt-4.1-mini`) | 96.7% | 2.77 | 3.50 | $0.000594 | −40% |
| 2. + 3 chunks sent to the model (was 5) | 95.6% | 2.67 | 3.30 | $0.000526 | −47% |
| 3. + no query-rewrite call | 97.2% | 2.06 | 2.88 | $0.000506 | −49% |
| 4. + skip the reranker when BM25 and vector search agree on the top chunk | 97.2% | 1.61 | 2.66 | $0.000346 | −65% |

*Source: `eval/results/optimisation_steps.md`. This is a separate run from the table above, so
the numbers differ slightly (run-to-run noise). With 181 questions, one question is 0.55 points.*

**What we learned from each step:**

1. **A smaller answer model gave the biggest cost saving (−40%).** `gpt-4o-mini` is about 2.7×
   cheaper per token. Accuracy barely moved (one question): the right passage is already in the
   prompt, so the model only has to copy and cite. It is not faster on its own.
2. **Fewer chunks (5 → 3) saved 7 more points of cost.** Recall@3 is 0.98 with the reranker, so
   chunks 4 and 5 were mostly noise. The −1.1-point dip (2 questions) is within the noise and
   disappears in the next step.
3. **Dropping the rewrite gave the biggest latency saving so far (p50 −0.6 s).** It removes one
   whole LLM round trip *before* search can even start. Accuracy didn't drop. The student's
   original words were enough, and the postmortem shows the rewrite once made things worse.
4. **Skipping the reranker when search is confident.** When BM25's #1 and the vector search's
   #1 are the same chunk, the reranker almost never changes the answer. That happens for
   **48%** of questions, so about half of all requests skip a whole LLM call. This step gave the
   biggest p95 gain.

**Answer cache (not in the tables).** The eval set has no repeated questions, so the cache can't
show up there. `python -m eval.cache_demo` asks 20 questions twice: a repeat costs **$0** and
takes **~0.003 ms** instead of ~1.9 s (`eval/results/cache_demo.json`). In real use, popular
questions ("when are finals?") repeat a lot, so this saves more as traffic grows.

**Prompt caching.** The fixed system prompt comes *first* in every request, so OpenAI's
automatic prompt caching can reuse it. Cached tokens are billed at a discount, and that is the
gap between the "list-price" and "billed" columns. We use list price as the main number because
cache hits vary from run to run.

## Ideas we tried or considered but didn't keep

| Idea | Why not |
|---|---|
| Cheaper embeddings / smaller index | Embeddings are already ~0% of cost (~$0.0000003 per query). Nothing to save. |
| Bigger chunks (400 words) | Recall@1 is barely better (0.85 vs 0.83), but every chunk sent to the model doubles in size. That works against step 2. |
| Removing the reranker entirely | Recall@1 drops from 0.89 to 0.83 (see the ablation). Skipping it only when search is confident keeps most of its benefit. |

## Cost model

Assumption: about 200 questions a day today (a few hundred active students in exam periods,
plus prospective students asking about programs).

| | Per request (billed) | Per month today (6k req) | Per month at 100× (600k req) |
|---|---|---|---|
| Baseline | $0.00096 | $5.75 | $575 |
| Optimised | $0.00025 | $1.51 | $151 |

- **Hosting:** Streamlit Community Cloud is free. At 100× traffic we'd need a paid host
  (~$20–50/month for a small VM), and the answer cache would cut LLM costs further, since
  popular questions repeat.
- **Indexing:** building the index for all 14 documents costs about $0.001, once (and again
  whenever a PDF is added).
- **Evaluation:** one full run of 181 questions costs about $0.25, judge included.
