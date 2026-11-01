# Design

## Corpus and users

- **Documents:** 14 public MedTech / SMU PDFs, **127 pages**, no personal data:

  | Document | Pages |
  |---|---|
  | Pre-Engineering & Engineering Student Guide 2026-2027 | 33 |
  | Licence & Master Student Guide 2026-2027 | 32 |
  | MedTech Catalogue (June 2026) | 21 |
  | Executive Master in BIM & ERP Management - Catalogue (2023) | 16 |
  | BIM Day Tunisia 2025 - Event Report | 6 |
  | Master in Cyber Security - Brochure (Jan 2024) | 3 |
  | Executive Master in BIM & ERP - Flyer (2023), Executive Master in Digital Health and AI, and 6 cybersecurity certification brochures (C\|EH, CND, CHFI, Ethical Hacking Essentials, Network Defense Essentials, Digital Forensics Essentials) | 2 each |

- **Users:** current MedTech students (rules: exams, grades, absences, scholarships, appeals)
  and prospective students / professionals (programs, masters, certifications, campus life).
- **Why it needs search:** 127 pages ≈ 60k tokens. Sending them with every question would cost
  ~60× more per request and make citations vague. The corpus is also full of near-duplicates:
  the two guides copy many paragraphs word for word; the 2023 and 2026 catalogues share text;
  the six certification brochures share the same layout and schedule section. Finding the
  *right* copy is a real retrieval problem.
- **Adding documents:** every PDF in `docs/` is used automatically. `rag/index.py` keeps a
  manifest of the PDFs the index was built from and rebuilds it at start-up when `docs/` changes
  (~10 s, < $0.01). Titles come from `TITLES` in `rag/config.py`, or from the file name.
- **Limitation:** the brief asks for "hundreds of pages"; we have 127. The course syllabi would
  be the natural next addition.

## Offline pipeline (`rag/ingest.py`, `rag/index.py`)

| Decision | Choice | Why |
|---|---|---|
| Loader | `pypdf`, page by page | Text-based PDFs extract cleanly; page numbers become citations. |
| Cleaning | Unicode NFKC normalisation (turns PDF ligatures like "ﬁ" back into "fi", otherwise BM25 sees "officials" as "of" + "cials"); skip near-empty pages (covers, photos); remove dotted TOC leaders, page numbers; **skip table-of-contents pages** | TOC pages contain every keyword ("Grading Scale ...... 21") and would win keyword search with no content. |
| Metadata | doc title, file name, PDF page, section heading | Citations, and a filter by document (`files=` in `Retriever.search`). Headings are detected with regexes for `VII. Title` and `5. Title`; brochures have none, so their chunks use the document title. |
| Chunking | 200 words, 40-word overlap, never across pages | Tested in `eval/chunk_sweep.py` (100 / 200 / 400). 100 words hurt Recall@1 badly; 400 words was slightly better at page-level Recall@1 but sends 2× the tokens to the LLM per chunk. 200 is the balance. |
| What we embed | `section heading + chunk text` | A table of grades with no words in it still gets "Grading Scale" meaning. |
| Embeddings | OpenAI `text-embedding-3-small` | Cheap ($0.02 / 1M tokens — the whole corpus costs $0.0005), good quality. |
| Vector store | **Flat file**: `index/chunks.json` + `index/embeddings.npy` | 227 chunks × 1536 floats = 1.4 MB. A numpy dot product over all of them takes < 1 ms. A vector DB would add a server to deploy and gain nothing at this size. Would switch to pgvector / FAISS above ~100k chunks. |
| Keyword index | BM25 (`rank_bm25`), rebuilt at start-up | Builds in milliseconds, so no need to store it. |

## Online pipeline (`rag/retrieve.py`, `rag/pipeline.py`)

1. **Rewrite (optional).** `gpt-4o-mini` rewrites the question into a search query. We search
   with *original + rewrite* concatenated: searching with the rewrite alone dropped key terms
   (see postmortem).
2. **Hybrid retrieval.** BM25 top-20 and vector top-20, merged with **Reciprocal Rank Fusion**
   (`score = Σ 1/(60 + rank)`), keep top 10. RRF uses only ranks, so we never have to calibrate
   BM25 scores against cosine similarities. **Metadata filter:** `search(..., files=[...])`
   restricts the search to chosen documents. The app searches all documents (we removed the
   program picker to keep the app simple), so the generator is told to say when programs differ.
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
| **Two guides, two answers** | Rules that differ between programs (absence limits, opening hours) can be mixed, since the app searches all documents. The prompt asks the model to say when programs differ, but it doesn't always. | "How many absences am I allowed?" |
| **Old vs new documents** | The 2023 BIM catalogue/flyer and the 2026 catalogue can disagree (dates, details). The prompt says to prefer the most recent document, but the model only knows the year from the title. | "When does the BIM master start?" (the 2023 flyer says 31 August) |
| **Look-alike brochures** | The six certification brochures share the same structure; a vague question ("how many modules in the program?") matches all of them and the answer depends on which one is retrieved. | "Are there prerequisites for the security program?" |
| **Tables** | `pypdf` flattens tables into one line; the grading-scale table loses its column structure, and in the 2026 catalogue the course lists of the three engineering programs come out interleaved, so the bot cannot tell which course belongs to which program. | "What percentage is a B+?", "What are the junior-year courses in Computer Systems Engineering?" |
| **Calendar dates** | The academic calendar is a table of dates; the model once wrote "May 5th" for "May 4th". | "When are Spring finals?" |
| **Chunk boundaries** | A heading at the bottom of a page and its content on the next page end up in different chunks. The content chunk gets the right section only if the heading regex catches it. | Section "7. Opening hours" split from its list. |
| **Query rewrite drift** | The rewrite can add generic words that match the wrong pages. Mitigated by keeping the original question in the query. | "Dean's List" → rewritten with the university name. |
| **Lists of items** | When the answer is a list (allowed sanctions, ACC referral sources), the model sometimes leaves items out or adds a plausible one. | "Who can refer a student to the ACC?" |
| **Questions outside the guides but near a topic** | The bot may answer from a nearby passage (e.g. gym question → it mentions gym *discounts*). | "Gym opening hours?" |
| **Bilingual questions** | Corpus is English only; French/Arabic questions work through embeddings but BM25 misses them. | "Quelle est la note de passage ?" |
