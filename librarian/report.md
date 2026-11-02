# The Librarian: RAG over MedTech's Documents

CS496 AI Engineering · Project 2 · Fall 2026 · Team: *names*

## 1. The corpus and its users

**Corpus.** 14 public MedTech / SMU documents, **127 pages** in all:
- the two 2026-2027 student guides, *Pre-Engineering & Engineering* (33 pages) and
  *Licence & Master* (32 pages), covering the calendar, attendance, exams, grading, GPA,
  probation, scholarships, conduct, appeals and campus services;
- the June 2026 school catalogue (21 pages): programs, curricula, partnerships, campus life;
- the Executive Master in BIM & ERP catalogue and flyer (2023), the BIM Day Tunisia 2025 report;
- brochures for the Master in Cyber Security, the Executive Master in Digital Health & AI, and six
  cybersecurity certifications (C|EH, CND, CHFI and three "Essentials" courses).

They are all public documents with no personal data.

**Users.** Current students ask about rules ("how many absences can I have?"). Prospective
students and professionals ask about programs ("what does the C|EH program cover?", "who is the
digital health master for?"). Students usually ask classmates, who are sometimes wrong, and a
wrong answer about exams or absences can cost a course. So every answer must point to its page.

**Why it needs retrieval.** 127 pages is about 60k tokens. Sending all of it with every question
would cost ~60× more and make citations vague. The corpus is also full of near-duplicates: the two
guides copy many paragraphs word for word, the 2023 and 2026 catalogues share text, and the six
certification brochures share the same layout. Finding the *right* copy is a real retrieval
problem. **Limitation:** 127 pages is still short of "hundreds"; the course syllabi would be the
next addition. **Growing the corpus is automatic:** every PDF in `docs/` is used, and the index
rebuilds itself when that folder changes.

**Evaluation set.** 181 questions (`data/eval_questions.json`). For each content chunk,
`gpt-4.1-mini` wrote one student-style question and its answer, and the chunk is kept as the gold
source passage. We added 15 hand-written questions the documents can't answer. Manual review
removed or fixed 24 bad items (duplicates, ambiguous "the program" questions, wrong
gold answers); see postmortem §3, §7 and `data/eval_review_log.json`. Final set: 167 answerable
+ 14 unanswerable.

## 2. Pipeline design

**Offline.** `pypdf` extracts each page. We apply Unicode NFKC normalisation, which turns the
brochures' "ﬁ" ligatures back into "fi" (otherwise keyword search sees "officials" as "of" +
"cials"). We remove dotted table-of-contents leaders and page numbers, and skip TOC pages and
near-empty cover pages. A regex tracks the current heading (`VII. Assessment and Grades > 5.
Grading Scale`); brochures have no numbered headings, so their chunks use the document title.
Pages are cut into **200-word chunks with 40-word overlap** that never cross a page, so each
chunk has exactly one page to cite. We embed `section + text` with `text-embedding-3-small` and
store the vectors in a **flat file** (227 × 1536 floats, 1.4 MB). Brute-force cosine search takes
under 1 ms, so a vector database would only add deployment work. BM25 is rebuilt at start-up in
milliseconds. A manifest of the indexed PDFs triggers an automatic rebuild when `docs/` changes.

**Online (the optimised version the app runs).**
0. *Cache*: a question asked before returns its saved answer (free, instant).
1. *Hybrid retrieval*: BM25 top-20 and vector top-20 are fused with Reciprocal Rank Fusion
   (`Σ 1/(60+rank)`) and the top 10 are kept. A document filter (`files=`) exists in the code;
   the app searches every document.
2. *Rerank*: `gpt-4o-mini` scores the 10 candidates 0–10 in one JSON call. It reads the question
   and passage together, like a cross-encoder, without shipping PyTorch to the host. **Skipped**
   when BM25's and the vector search's #1 chunk are the same (48% of questions).
3. *Generate*: the top 3 chunks are numbered `[1]..[3]` with their document, page and section.
   The system prompt forbids outside knowledge, requires a citation after every factual
   sentence, and says to prefer the most recent document when two disagree.
4. *"I don't know"*: if the best rerank score is under 3/10, we answer without calling the
   generator. The generator also has a strict rule to reply with only the "I don't know"
   sentence when no source answers.

The baseline (first version) also rewrote the question with an LLM, always reranked, sent 5
chunks and used `gpt-4.1-mini`.

**App.** Streamlit, with three tabs. **Ask**: chat with the answer, expandable sources, latency
and cost per answer, a "behind the scenes" panel showing every step, and 👍/👎 feedback. **How it
works**: a diagram and plain-language explanation. **Evaluation**: a live dashboard of every
result. There are no settings to pick; the sidebar lists the documents searched.

## 3. Retrieval results and ablation

A hit means a top-k chunk comes from the gold passage's page, or is a near-copy of the gold
passage in another document. Questions are used raw, without rewriting.

| Mode | R@1 | R@3 | R@5 | R@10 | MRR | p95 ms |
|---|---|---|---|---|---|---|
| keyword only | 0.86 | 0.94 | 0.98 | 0.99 | 0.91 | 9 |
| vector only | 0.75 | 0.94 | 0.96 | 0.99 | 0.84 | 424 |
| hybrid | 0.83 | 0.97 | 0.99 | 0.99 | 0.90 | 300 |
| hybrid + rerank | **0.89** | **0.98** | **0.99** | 0.99 | **0.93** | 1316 |

**What each part added.** Hybrid search raised recall at k ≥ 3 (R@3 0.97 vs 0.94 / 0.94), because
each method catches the other's misses. RRF slightly lowers R@1 compared with keyword search
alone. The **reranker** brings the biggest gain at the top: R@1 +6 points and MRR +0.03. That is
what makes it safe to send only 3 chunks. Its cost, about 1 s of latency, is why we added "skip
when confident".

**Chunk size.** Hybrid R@1 was 0.71 / 0.83 / 0.85 for 100 / 200 / 400-word chunks (R@5 ≈ 0.98–0.99
for all). 400 words is barely better but doubles the tokens per chunk sent to the generator, so
we kept 200.

**Caveat.** The generated questions reuse each passage's wording, which favours BM25 and pushes
recall towards 1.0. Real student phrasing would be harder.

## 4. Generation results

An LLM judge (`gpt-4.1-mini`) sees the question, the gold answer, the bot's answer and the chunks
it was given, and marks three things: *correct* (the key facts match, with no contradiction),
*faithful* (every claim is supported by the chunks) and *citations accurate* (each [n] supports
its claim). "I don't know" counts as correct only on unanswerable questions.

| | Baseline | Optimised |
|---|---|---|
| Accuracy, all 181 | 97.2% | 97.2% |
| Correct, answerable (167) | 97.6% | 97.0% |
| Faithfulness | 100% | 100% |
| Citation accuracy | 100% | 100% |
| "I don't know" when it should (14) | 92.9% | 100% |
| "I don't know" when it shouldn't | 0.6% | 0.6% |

On the first corpus (the two guides only), both versions reached 99.2%, but only after we read
every failure. The very first run scored **88.7%**: a pipeline bug (the rewrite dropped key
terms), a detection bug (partial "I don't know" answers) and three wrong gold answers.

The remaining errors on the full corpus fall into four groups:
- partial answers: "starts in August" instead of "31st August"; "the AECO industry" instead of
  the list of roles;
- a misread flattened calendar table;
- a name the bot could not find (Dr. Ons Najjar's role);
- a near-topic answer for the gym question.

**Is the judge reliable?** We reuse our Project 1 harness, as the brief asks: P1's Cohen's kappa
and JSON-parsing functions (`p1_harness/`) and its two bias checks (`eval/judge_reliability.py`).
Position bias: the verdict followed the order of the answers in 1/30 swapped pairs. Verbosity
bias: 0/30 padded wrong answers became "correct". Agreement with humans is **pending**: 60 answers
are waiting for two labellers. Until then, treat 100% faithfulness with care (postmortem §4, §9).

## 5. Week 8: what we changed and what it saved

| Step (cumulative) | Accuracy | p50 | p95 | List-price cost | Δ cost |
|---|---|---|---|---|---|
| Baseline | 97.2% | 2.42 s | 3.17 s | $0.000993 | — |
| + `gpt-4o-mini` for answers | 96.7% | 2.77 s | 3.50 s | $0.000594 | −40% |
| + 3 chunks instead of 5 | 95.6% | 2.67 s | 3.30 s | $0.000526 | −47% |
| + no rewrite call | 97.2% | 2.06 s | 2.88 s | $0.000506 | −49% |
| + skip reranker when BM25 & vector agree | 97.2% | 1.61 s | 2.66 s | $0.000346 | −65% |

The final head-to-head on the same 181 questions:
- **cost −68%** (list price) / −74% (billed);
- **p50 −41%**, p95 −23%;
- **accuracy ±0**.

The target (≥50% cheaper, ≤2 points lost) is met, and it was also met on the first corpus (−69%,
±0). The answer cache makes repeated questions free and instant (measured: ~1.9 s → 0.003 ms).
That doesn't show in the eval, because the eval has no repeats. Prompt caching of the fixed
system prompt explains the gap between list and billed cost.

## 6. Known failure cases and cost model

**Failure cases** (details in `design.md`):
- rules that differ between the two guides can be mixed, since the app searches everything;
- old vs new documents can disagree (2023 vs 2026);
- look-alike certification brochures make vague questions ambiguous;
- flattened tables: grading scale, calendar dates, and catalogue course lists interleaved across
  programs;
- headings separated from their content at page breaks;
- list answers with missing items;
- near-topic questions answered from a nearby passage;
- French or Arabic questions, which BM25 can't match.

**Cost model** (billed, optimised): **$0.00025 per request**. At ~200 questions a day that is
**$1.51/month**; at 100× (20k a day) it is **$151/month**, plus ~$20–50 for a paid host.
Indexing all 14 documents costs ~$0.001, and a full evaluation run costs ~$0.25. Latency doesn't
grow with traffic because requests are independent. OpenAI rate limits would be the first wall
at 100×, and the answer cache helps more as traffic grows.
