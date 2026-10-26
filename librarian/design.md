# Design

## Corpus and users

- **Documents:** the two official MedTech student guides for 2026-2027
  (*Pre-Engineering & Engineering*, 33 pages; *Licence & Master*, 32 pages) — public documents
  handed to every student, no personal data.
- **Users:** MedTech students (our classmates) asking about exams, grades, absences,
  scholarships, conduct, appeals and campus services.
- **Why it needs search:** 65 pages ≈ 40k tokens. It could technically fit in one prompt, but
  sending it with every question would cost ~40× more per request and makes citations vague.
  The guides also overlap heavily (many paragraphs are copied word for word, some rules differ),
  which makes retrieval and metadata filtering genuinely necessary.
- **Limitation:** the brief asks for "hundreds of pages". Our corpus is 65 pages. Adding the
  course catalogue / syllabi would be the natural extension (see postmortem).

## Offline pipeline (`rag/ingest.py`, `rag/index.py`)

| Decision | Choice | Why |
|---|---|---|
| Loader | `pypdf`, page by page | Text-based PDFs extract cleanly; page numbers become citations. |
| Cleaning | Remove dotted TOC leaders, page numbers; **skip table-of-contents pages** | TOC pages contain every keyword ("Grading Scale ...... 21") and would win keyword search with no content. |
| Metadata | doc title, program (`engineering` / `licence_master`), PDF page, section heading | Citations + the program filter in the sidebar. Headings are detected with regexes for `VII. Title` and `5. Title`. |
| Chunking | 200 words, 40-word overlap, never across pages | Tested in `eval/chunk_sweep.py` (100 / 200 / 400). 100 words hurt Recall@1 badly; 400 words was slightly better at page-level Recall@1 but sends 2× the tokens to the LLM per chunk. 200 is the balance. |
| What we embed | `section heading + chunk text` | A table of grades with no words in it still gets "Grading Scale" meaning. |
| Embeddings | OpenAI `text-embedding-3-small` | Cheap ($0.02 / 1M tokens — the whole corpus costs $0.0005), good quality. |
| Vector store | **Flat file**: `index/chunks.json` + `index/embeddings.npy` | 129 chunks × 1536 floats = 0.8 MB. A numpy dot product over all of them takes < 1 ms. A vector DB would add a server to deploy and gain nothing at this size. Would switch to pgvector / FAISS above ~100k chunks. |
| Keyword index | BM25 (`rank_bm25`), rebuilt at start-up | Builds in milliseconds, so no need to store it. |

## Online pipeline (`rag/retrieve.py`, `rag/pipeline.py`)

1. **Rewrite (optional).** `gpt-4o-mini` rewrites the question into a search query. We search
   with *original + rewrite* concatenated: searching with the rewrite alone dropped key terms
   (see postmortem).
2. **Hybrid retrieval.** BM25 top-20 and vector top-20, merged with **Reciprocal Rank Fusion**
   (`score = Σ 1/(60 + rank)`), keep top 10. RRF uses only ranks, so we never have to calibrate
   BM25 scores against cosine similarities. **Metadata filter:** if the user picks a program,
   chunks from the other guide are excluded before ranking.
3. **Rerank.** `gpt-4o-mini` scores the 10 candidates 0–10 in one call (JSON output). It reads
   question and passage together, like a cross-encoder. We chose an LLM over a local
   cross-encoder (e.g. `ms-marco-MiniLM`) to avoid shipping PyTorch (~2 GB) to Streamlit Cloud.
4. **Generate.** Top-N chunks (5 baseline, 3 optimised) are numbered `[1]..[N]` with doc / page /
   section. The system prompt forces a citation after every factual sentence and forbids
   outside knowledge. Temperature 0.
5. **"I don't know".** Two layers: (a) if the best rerank score < 3/10 we answer
   "I don't know" *without calling the generator* (cheaper); (b) the generator is told to start
   with the same sentence if the sources don't contain the answer.

## Known failure cases

| Case | What happens | Example |
|---|---|---|
| **Two guides, two answers** | If the user doesn't pick a program, rules that differ (absence limits, opening hours) can be mixed. The prompt asks the model to say when programs differ, but it doesn't always. | "How many absences am I allowed?" |
| **Tables** | `pypdf` flattens tables into a line of numbers; the grading-scale table loses its column structure. Answers about exact ranges can be off by one row. | "What percentage is a B+?" |
| **Calendar dates** | The academic calendar is a table of dates; the model once wrote "May 5th" for "May 4th". | "When are Spring finals?" |
| **Chunk boundaries** | A heading at the bottom of a page and its content on the next page end up in different chunks. The content chunk gets the right section only if the heading regex catches it. | Section "7. Opening hours" split from its list. |
| **Query rewrite drift** | The rewrite can add generic words that match the wrong pages. Mitigated by keeping the original question in the query. | "Dean's List" → rewritten with the university name. |
| **Lists of items** | When the answer is a list (allowed sanctions, ACC referral sources), the model sometimes leaves items out or adds a plausible one. | "Who can refer a student to the ACC?" |
| **Questions outside the guides but near a topic** | The bot may answer from a nearby passage (e.g. gym question → it mentions gym *discounts*). | "Gym opening hours?" |
| **Bilingual questions** | Corpus is English only; French/Arabic questions work through embeddings but BM25 misses them. | "Quelle est la note de passage ?" |
