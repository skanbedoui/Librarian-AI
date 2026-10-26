# Understanding The Librarian — a guide for the team

The rules say *"you can explain every line"*. This file walks through the project in the order
the data flows, in plain words, and ends with questions you'll likely be asked at the demo.

---

## 1. The idea in one paragraph

A language model doesn't know MedTech's rules, and if you ask it anyway it will make something
up. **RAG** (Retrieval-Augmented Generation) fixes that: before asking the model, we **search**
the student guides for the passages that matter, put those passages in the prompt, and tell the
model *"answer only from these, and cite which one you used"*. If the search finds nothing
relevant, we say *"I don't know"* instead of guessing.

## 2. The files, in the order things happen

```
docs/*.pdf ──► rag/ingest.py ──► rag/index.py ──► index/chunks.json + embeddings.npy
                (cut into chunks)   (embed + save)
                                                        │
question ──► rag/pipeline.py ──► rag/retrieve.py ───────┘
             (rewrite, generate)   (BM25 + vectors + rerank)
                    │
                    ▼
               app.py (Streamlit UI)        eval/*.py (measure everything)
```

| File | What it does | Lines worth knowing |
|---|---|---|
| `rag/config.py` | Every setting in one place, plus the **BASELINE** and **OPTIMISED** configurations. | `Settings` dataclass; `OPTIMISED = replace(BASELINE, ...)` shows exactly what Week 8 changed. |
| `rag/ingest.py` | Reads each PDF page, cleans it, tracks the current section heading, and cuts it into overlapping 200-word chunks. | `is_table_of_contents` (why we skip TOC pages), `chunk_pages` (the sliding window). |
| `rag/llm.py` | Wraps the OpenAI API. Every call adds its tokens and dollars to a `Usage` object, so each answer knows its own cost. | `Usage.add` (the cost formula), `embed` (normalises vectors so dot product = cosine similarity). |
| `rag/index.py` | Embeds all chunks and saves them as two files. Run once: `python -m rag.index`. | Why a flat file and not a vector DB (comment at the top). |
| `rag/retrieve.py` | The 4 search modes. | `hybrid_search` (RRF), `rerank` (LLM scores 0-10), `search` (one entry point). |
| `rag/pipeline.py` | The whole question → answer flow. | `ANSWER_SYSTEM` (the rules for the model), the `min_relevance` check (I don't know), the cache. |
| `app.py` | The Streamlit website: chat, "how it works", evaluation dashboard, 👍/👎 feedback. | `st.cache_resource` (index loaded once for everyone). |
| `eval/make_questions.py` | Builds the 122-question test set. | Each question keeps its source passage → we can check retrieval. |
| `eval/eval_retrieval.py` | Recall@k and the ablation table. | `is_hit` (what counts as finding the right passage). |
| `eval/chunk_sweep.py` | Tests chunk sizes 100 / 200 / 400. | |
| `eval/eval_generation.py` | Runs the full bot on every question; an LLM judge grades correctness, faithfulness, citations. Also latency and cost. | `judge`, the `summary` dict. |
| `eval/cache_demo.py` | Shows what the answer cache saves on repeated questions. | |

## 3. Key concepts, explained simply

**Chunk.** A ~200-word piece of a page. Small enough to be precise and cheap, big enough to hold
a whole rule. Neighbouring chunks share 40 words (*overlap*) so a sentence cut in half at a
boundary still exists whole in one of them.

**Embedding.** A list of 1536 numbers that represents the *meaning* of a text. Texts with similar
meaning have vectors pointing in similar directions. We measure that with **cosine similarity**
(1 = same direction, 0 = unrelated). Because we normalise every vector to length 1, cosine
similarity is just a dot product: `self.vectors @ query_vec`.

**BM25.** The classic search-engine formula. A chunk scores high if it contains the question's
words, especially *rare* words (like "CGPA") and especially if the chunk is short. It knows
nothing about meaning: "retake" ≠ "redo".

**Hybrid search + RRF.** We run both searches and merge their rankings. *Reciprocal Rank Fusion*
gives every chunk `1/(60 + rank)` from each list and adds them up. A chunk ranked 1st by both
gets the most. We use ranks instead of raw scores because a BM25 score of 12.3 and a cosine of
0.61 aren't comparable. (60 is the standard constant from the original RRF paper; it stops the
1st place from dominating.)

**Reranker.** Search compares the question and the chunk *separately* (two vectors, or word
counts). A reranker reads them *together* and judges "does this passage answer this
question?" — more accurate, but slower, so we only run it on the top 10. Ours is a small LLM
(`gpt-4o-mini`) returning scores 0-10 as JSON.

**Citations.** Chunks go into the prompt numbered `[1]`, `[2]`, ... and the prompt orders the
model to put the number after each fact. The app then shows the page each number points to.

**"I don't know".** If the reranker's best score is below 3/10, nothing relevant was found, so we
don't even call the answer model. And the model itself is told to start with the exact sentence
*"I don't know based on the student guides."* when the sources don't answer the question.

**Recall@k.** For a test question, did the right passage appear in the top *k* results?
Recall@5 = 0.95 means it did for 95% of questions. Measures *search only*.

**Faithfulness.** Is every claim in the answer supported by the chunks the model was given?
An answer can be *correct* but *unfaithful* (it guessed right from memory) — that's still a bug.

**p50 / p95 latency.** Sort all response times. p50 is the middle one (typical user), p95 is the
one slower than 95% of requests (the unlucky user). We optimise p95 because that's what people
complain about.

**Ablation.** Turn components on one at a time to see what each one adds: keyword only → vector
only → hybrid → hybrid + reranker.

## 4. Running it

```bash
pip install -r requirements.txt          # once
python -m rag.index                      # once (already done: index/ is in the repo)
streamlit run app.py                     # the website
python -m eval.eval_retrieval            # retrieval + ablation
python -m eval.eval_generation --config baseline
python -m eval.eval_generation --config optimised
```

## 5. Questions you'll probably be asked

**Why not just put both guides in the prompt?** ~40k tokens per question = ~40× the cost and
slower, citations become vague, and it doesn't scale when the corpus grows.

**Why a flat file instead of a vector database?** 129 vectors. Brute-force search is < 1 ms.
A database would add a server to deploy and maintain for zero speed gain.

**Why 200 words per chunk?** We tested 100, 200 and 400 (`eval/chunk_sweep.py`). See the
numbers in `eval/results/chunk_sweep.md` and the reasoning in `design.md`.

**What did hybrid search add? And the reranker?** Read the ablation table in the README: the
reranker mainly moves the right passage to position 1 (Recall@1, MRR).

**How do you know the answers are right?** 108 answerable questions, each with its source
passage, plus 14 questions the guides can't answer. An LLM judge compares the bot's answer
with the gold answer and checks every claim against the cited chunks.

**Isn't an LLM judging an LLM biased?** Yes, a bit — it's in the postmortem. The judge sees the
gold answer, which limits how lenient it can be on correctness, and reading the failures by hand
is how we found the wrong gold answers (postmortem §3).

**Your eval questions were generated by an LLM from the chunks — isn't that too easy?** Yes:
they reuse the passage's words, which flatters keyword search. That's in the postmortem; the
fix is more questions from real students (the 👍/👎 feedback log is the start of that).

**How did you cut the cost by 50%?** See `optimisation.md`: smaller answer model, 3 chunks
instead of 5, no rewrite call, and the reranker is skipped when BM25 and vector search agree on
the top chunk. Each change is measured on the same test set.

**What happens at 100× traffic?** Cost scales linearly (see cost model in README). Latency
doesn't change since each request is independent; OpenAI rate limits would be the first wall.
The answer cache saves more as traffic grows, since popular questions repeat.

**What's the worst failure?** Mixing the two guides' rules when the student doesn't select a
program. A wrong answer about absences or exams can cost a student a course.
