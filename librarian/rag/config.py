"""
All the knobs of the system live here, so you can see every decision in one place.

Two ready-made configurations are defined at the bottom:
  * BASELINE  - the "working version" (Week 7)
  * OPTIMISED - the "cheaper and faster" version (Week 8)
"""
from dataclasses import dataclass, replace
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"          # the PDFs (our corpus)
INDEX_DIR = ROOT / "index"        # chunks.json + embeddings.npy are saved here
DATA_DIR = ROOT / "data"          # eval questions, user feedback
RESULTS_DIR = ROOT / "eval" / "results"

# ---------- Corpus ----------
# file name -> (nice title, program tag used for metadata filtering)
DOCUMENTS = {
    "engineering_guide_2026-2027.pdf": ("Pre-Engineering & Engineering Student Guide 2026-2027", "engineering"),
    "licence_master_guide_2026-2027.pdf": ("Licence & Master Student Guide 2026-2027", "licence_master"),
}

# ---------- Chunking ----------
CHUNK_WORDS = 200      # target chunk size in words (tested in eval/chunk_sweep.py)
CHUNK_OVERLAP = 40     # words repeated between neighbouring chunks

# ---------- Models ----------
EMBED_MODEL = "text-embedding-3-small"

# Price per 1M tokens in USD: (input, output). Check https://openai.com/api/pricing
PRICES = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}


@dataclass(frozen=True)
class Settings:
    """One full configuration of the online (question-answering) pipeline."""
    search_mode: str = "hybrid+rerank"   # "keyword" | "vector" | "hybrid" | "hybrid+rerank"
    top_k: int = 10                      # how many chunks the search step returns
    n_context: int = 5                   # how many chunks we finally show the LLM
    gen_model: str = "gpt-4.1-mini"      # model that writes the answer
    rerank_model: str = "gpt-4o-mini"    # model that re-scores the search results
    rewrite_query: bool = True           # let an LLM clean up vague questions first
    skip_rerank_if_confident: bool = False  # Week 8 trick: skip reranking when search is sure
    use_cache: bool = False              # Week 8 trick: reuse answers to repeated questions
    min_relevance: float = 3.0           # rerank score (0-10) below which we say "I don't know"


BASELINE = Settings()

OPTIMISED = replace(
    BASELINE,
    n_context=3,                    # send fewer chunks -> fewer input tokens
    gen_model="gpt-4o-mini",        # smaller, cheaper generation model
    rewrite_query=False,            # one less LLM call
    skip_rerank_if_confident=True,  # skip the reranker when keyword & vector agree
    use_cache=True,                 # repeated questions cost nothing
)
