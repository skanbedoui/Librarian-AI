# Week 8: making it cheaper and faster

**Target:** cut cost per request **or** p95 latency by 50%, losing at most 2 points of accuracy,
measured before and after on the same test set.

**Result: cost −69%, p95 latency −34%, p50 latency −49%, accuracy unchanged (99.2% → 99.2%).**

Test set: `data/eval_questions.json` (122 questions: 108 answerable + 14 unanswerable).
"Accuracy" = share of questions handled correctly (right answer, or "I don't know" when the
guides don't have the answer). Same questions, same judge, same concurrency (4 threads) for
every row.

## Before / after

| Metric | Baseline (Week 7) | Optimised (Week 8) | Change |
|---|---|---|---|
| Accuracy (all 122) | 99.2% | 99.2% | **±0 pts** |
| Correct on answerable | 100.0% | 99.1% | −0.9 pts (1 question) |
| "I don't know" when it should | 92.9% | 100.0% | +7.1 pts (1 question) |
| Faithfulness | 100% | 100% | — |
| Citation accuracy | 100% | 100% | — |
| p50 latency | 3.19 s | 1.62 s | **−49%** |
| p95 latency | 4.88 s | 3.20 s | **−34%** |
| Cost / request (list price) | $0.000973 | $0.000300 | **−69%** |
| Cost / request (billed, with prompt-cache discount) | $0.000741 | $0.000215 | **−71%** |

*Source: `eval/results/generation_baseline.json`, `eval/results/generation_optimised.json`.*

## What each change did

Changes applied one after the other (`python -m eval.optimisation_steps`):

| Step | Accuracy | p50 (s) | p95 (s) | List-price cost | vs baseline |
|---|---|---|---|---|---|
| 0. Baseline | 99.2% | 3.25 | 4.48 | $0.000974 | — |
| 1. + `gpt-4o-mini` writes the answers (was `gpt-4.1-mini`) | 99.2% | 3.54 | 4.74 | $0.000596 | −39% |
| 2. + 3 chunks sent to the model (was 5) | 98.4% | 3.26 | 4.52 | $0.000527 | −46% |
| 3. + no query-rewrite call | 99.2% | 2.48 | 3.72 | $0.000510 | −48% |
| 4. + skip the reranker when BM25 and vector search agree on the top chunk | 99.2% | 1.96 | 3.37 | $0.000349 | −64% |

*Source: `eval/results/optimisation_steps.md`. This is a separate run from the table above, so
numbers differ slightly, by about the run-to-run noise.*

**What we learned from each step:**

1. **Smaller answer model: the biggest cost saving (−39%).** `gpt-4o-mini` is ~2.7× cheaper per
   token. Accuracy held because the job is now easy: the right passage is already in the prompt,
   so the model only has to copy and cite. It didn't make things faster on its own.
2. **Fewer chunks (5 → 3): −7 more points of cost.** Recall@3 is already 1.00 with the reranker,
   so chunks 4 and 5 were mostly noise. The 0.8-point dip is one question, inside the noise.
3. **No rewrite: the biggest latency saving so far (p50 −0.8 s).** It removes one full LLM round
   trip *before* search can start. Accuracy didn't drop: the student's original words were
   already enough (see the postmortem: the rewrite once made things worse).
4. **Skip the reranker when confident.** When BM25's #1 and the vector search's #1 are the same
   chunk, the reranker almost never changes the answer. This happens for **54%** of questions,
   so half the requests skip a whole LLM call. Biggest p95 gain.

**Answer cache (not in the tables).** The eval set has no repeated questions, so the cache can't
show up there. `python -m eval.cache_demo` asks 20 questions twice: the repeat costs **$0** and
takes **~0.003 ms** instead of ~2.1 s (`eval/results/cache_demo.json`). In real use, popular
questions ("when are finals?") repeat a lot, so this saves more as traffic grows.

**Prompt caching.** The fixed system prompt comes *first* in every request, so OpenAI's
automatic prompt caching can reuse it. Cached tokens are billed at a discount. That is the gap
between the "list-price" and "billed" columns. We report list price as the main number because
cache hits vary from run to run.

## Ideas we tried or considered but didn't keep

| Idea | Why not |
|---|---|
| Cheaper embeddings / smaller index | Embeddings are already ~0% of cost: ~$0.0000003 per query. Nothing to save. |
| Bigger chunks (400 words) | Better Recall@1 (0.87 vs 0.80) but twice the tokens per chunk sent to the model. That works against step 2. |
| Remove the reranker entirely | Recall@1 drops from 0.88 to 0.81 (see ablation). Skipping it only when confident keeps most of its benefit. |

## Cost model

Assumption: ~200 questions/day today (a few hundred active students during exam periods).

| | Per request | Per month today (6k req) | Per month at 100× (600k req) |
|---|---|---|---|
| Baseline | $0.00074 | $4.45 | $445 |
| Optimised | $0.00022 | $1.29 | $129 |

- Hosting: Streamlit Community Cloud is free. At 100× traffic we'd need a paid host (~$20–50/month
  for a small VM), and the answer cache would cut LLM costs further since popular questions repeat.
- One-time indexing cost: $0.0005 for the whole corpus.
- The evaluation itself costs about $0.15 per full run of 122 questions (including the judge).
