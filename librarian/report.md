# The Librarian: RAG over the MedTech Student Guides

CS496 AI Engineering · Project 2 · Fall 2026 · Team: *names*

## 1. The corpus and its users

**Corpus.** The two official 2026-2027 MedTech student guides: *Pre-Engineering & Engineering*
(33 pages) and *Licence & Master* (32 pages). They cover the academic calendar, attendance,
exams, grading, GPA, probation, scholarships, the honor code, sanctions, appeals and campus
services. Both are public documents handed to every student and contain no personal data.

**Users.** MedTech students. The guides are long and formal, and students mostly ask
classmates instead ("how many absences can I have?", "what happens if I miss the midterm?"),
and classmates are sometimes wrong. A wrong answer about exams or absences can cost a student a
course, so every answer must point to its source page.

**Why it needs retrieval.** The two guides are ~40k tokens together. Sending them with every
question would cost ~40× more and make citations vague. They also overlap a lot: many
paragraphs are identical, and some rules differ (absence limits, opening hours). That makes
search, de-duplication and metadata filtering by program real problems. **Limitation:** 65
pages is smaller than the "hundreds of pages" the brief suggests. The natural extension is the
course catalogue and prerequisite rules.

**Evaluation set.** 122 questions (`data/eval_questions.json`). For each content chunk,
`gpt-4.1-mini` wrote one student-style question with its answer, and the chunk is kept as the
gold source passage. We added 15 hand-written questions the guides can't answer. A manual
review removed or fixed 3 bad items (postmortem §3). Final set: 108 answerable + 14 unanswerable.

## 2. Pipeline design

**Offline.** `pypdf` extracts each page. We remove dotted table-of-contents leaders and page
numbers, and **skip TOC pages entirely**, because they contain every keyword and no content. A
regex tracks the current heading (`VII. Assessment and Grades > 5. Grading Scale`). Pages are cut
into **200-word chunks with 40-word overlap** that never cross a page, so each chunk has one page
number for its citation. Each chunk stores doc, program, page and section. We embed
`section + text` with `text-embedding-3-small` (the heading adds meaning to table-like chunks)
and store vectors in a **flat file** (129 × 1536 floats, 0.8 MB). Brute-force cosine search
takes under 1 ms, so a vector database would only add deployment complexity. BM25 is rebuilt
at start-up in milliseconds.

**Online.**
1. *Rewrite* (baseline only): `gpt-4o-mini` rewrites vague questions. We search with
   original + rewrite, so the student's own words are never lost.
2. *Hybrid retrieval*: BM25 top-20 and vector top-20 fused with Reciprocal Rank Fusion
   (`Σ 1/(60+rank)`), top 10 kept, with an optional **program filter** on metadata.
3. *Rerank*: `gpt-4o-mini` scores all 10 candidates 0–10 in one JSON call. It reads question and
   passage together, like a cross-encoder, without shipping PyTorch to the host.
4. *Generate*: the top 5 (baseline) or 3 (optimised) chunks are numbered `[1]..[n]`. The system
   prompt forbids outside knowledge and requires a citation after every factual sentence.
5. *"I don't know"*: if the best rerank score is under 3/10, we answer without calling the
   generator. The generator also has a strict rule to reply only with the "I don't know" sentence
   when no source answers.

**App.** Streamlit, with three tabs: chat (answer, expandable sources, latency and cost per
answer, a "behind the scenes" panel showing every pipeline step, 👍/👎 feedback), an
explanation of the pipeline, and a live dashboard of all evaluation results.

## 3. Retrieval results and ablation

Hit = a top-k chunk comes from the gold passage's page, or is a near-copy of the gold passage
from the other guide. Raw questions, no rewrite.

| Mode | R@1 | R@3 | R@5 | R@10 | MRR | p95 ms |
|---|---|---|---|---|---|---|
| keyword only | 0.84 | 0.95 | 0.98 | 1.00 | 0.90 | 8 |
| vector only | 0.75 | 0.94 | 0.96 | 1.00 | 0.84 | 1017 |
| hybrid | 0.81 | 0.98 | 1.00 | 1.00 | 0.89 | 474 |
| hybrid + rerank | **0.88** | **1.00** | **1.00** | 1.00 | **0.94** | 1874 |

**What each part added.** Hybrid search raised recall at k ≥ 3 (R@5 1.00 vs 0.98 / 0.96), because
each method catches the other's misses. RRF slightly lowers R@1 compared with keyword alone. The
**reranker** gives the biggest gain at the top: R@1 +7 points, MRR +0.05. That is what lets us
send only 3 chunks in the optimised version. Its cost is ~1 s of latency, which motivated
"skip when confident" in Week 8.

**Chunk size.** Hybrid R@1 was 0.65 / 0.80 / 0.87 for 100 / 200 / 400-word chunks (R@5 = 1.00 for
all). Larger chunks win at page level but double the tokens per chunk sent to the generator, so
we kept 200.

**Caveat.** Generated questions reuse the passage's wording, which favours BM25 and pushes
recall towards 1.0. Real student phrasing would be harder.

## 4. Generation results

An LLM judge (`gpt-4.1-mini`) sees the question, the gold answer, the bot's answer and the
chunks it was given. It marks *correct* (key facts match, no contradiction), *faithful* (every
claim is supported by the given chunks) and *citations accurate* (each [n] supports its claim).
"I don't know" is correct only on unanswerable questions.

| | Baseline | Optimised |
|---|---|---|
| Accuracy, all 122 | 99.2% | 99.2% |
| Correct, answerable | 100% | 99.1% |
| Faithfulness | 100% | 100% |
| Citation accuracy | 100% | 100% |
| "I don't know" when it should (14) | 92.9% | 100% |
| "I don't know" when it shouldn't (108) | 0% | 0% |

The scores were not this high at first. The first full run scored **88.7%**. Reading every
failure found a pipeline bug (the rewrite dropped key terms), a detection bug (partial "I don't
know" answers) and three wrong gold answers. The remaining errors are the gym question (baseline
answered "the guides don't specify…" instead of using the exact "I don't know" sentence) and a
calendar date the optimised model misread from a flattened table. 100% faithfulness should be
read with care: the judge may be lenient (postmortem §4).

## 5. Week 8: what we changed and what it saved

| Step (cumulative) | Accuracy | p50 | p95 | List-price cost | Δ cost |
|---|---|---|---|---|---|
| Baseline | 99.2% | 3.25 s | 4.48 s | $0.000974 | — |
| + `gpt-4o-mini` for answers | 99.2% | 3.54 s | 4.74 s | $0.000596 | −39% |
| + 3 chunks instead of 5 | 98.4% | 3.26 s | 4.52 s | $0.000527 | −46% |
| + no rewrite call | 99.2% | 2.48 s | 3.72 s | $0.000510 | −48% |
| + skip reranker when BM25 & vector agree (54% of requests) | 99.2% | 1.96 s | 3.37 s | $0.000349 | −64% |

The final head-to-head on the same 122 questions gave **cost −69% (list) / −71% (billed), p95
−34%, p50 −49%, accuracy ±0**. The target (≥50% cheaper, ≤2 points lost) is met. An answer cache
makes repeated questions free and instant (measured: ~2.1 s → 0.003 ms). It doesn't show in the
eval because the eval has no repeats. Prompt caching of the fixed system prompt explains the gap
between list and billed cost.

## 6. Known failure cases and cost model

**Failure cases** (details in `design.md`): mixing the two guides' rules when no program is
selected; flattened tables (grading scale, calendar dates); headings separated from their content
at page breaks; list answers with missing or invented items; near-topic questions answered from
a nearby passage (gym → gym discounts); French or Arabic questions, which BM25 can't match;
query-rewrite drift (mitigated).

**Cost model** (billed, optimised): **$0.00022 per request**. At ~200 questions/day that's
**$1.29/month**; at 100× (20k/day) it's **$129/month** plus ~$20–50 for a paid host. Indexing the
whole corpus once costs $0.0005. A full evaluation run costs ~$0.15. Latency doesn't grow with
traffic, since requests are independent. OpenAI rate limits would be the first limit at 100×,
and the answer cache helps more as traffic grows.
