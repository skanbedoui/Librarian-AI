"""
All the knobs of the system live here, so you can see every decision in one place.

Two ready-made configurations are defined at the bottom:
  * BASELINE  - the first working version (Week 7), kept only to measure against
  * OPTIMISED - the cheaper and faster version (Week 8): the one the app uses
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
# EVERY PDF in docs/ is used automatically: drop a file in, restart the app, and the index
# is rebuilt (see rag/index.py: ensure_index). A title shown in citations is made from the
# file name ("network_defense_essentials.pdf" -> "Network Defense Essentials").
# Optional: give a nicer title here.
TITLES = {
    "engineering_guide_2026-2027.pdf": "Pre-Engineering & Engineering Student Guide 2026-2027",
    "licence_master_guide_2026-2027.pdf": "Licence & Master Student Guide 2026-2027",
    "medtech_catalogue_june_2026.pdf": "MedTech Catalogue (June 2026)",
    "bim_erp_executive_master_catalogue_2023.pdf": "Executive Master in BIM & ERP Management - Catalogue (2023)",
    "bim_erp_executive_master_flyer_2023.pdf": "Executive Master in BIM & ERP Management - Flyer (2023)",
    "bim_day_tunisia_2025.pdf": "BIM Day Tunisia 2025 - Event Report",
    "cybersecurity_brochure_jan_2024.pdf": "Master in Cyber Security - Brochure (Jan 2024)",
    "master_digital_health_and_ai.pdf": "Executive Master in Digital Health and AI",
    "certified_ethical_hacking.pdf": "Certified Ethical Hacker (C|EH) Program",
    "computer_hacking_forensic_investigator.pdf": "Computer Hacking Forensic Investigator (CHFI) Program",
}


def list_documents():
    """Return [(file name, title), ...] for every PDF in docs/, sorted by name."""
    return [(p.name, TITLES.get(p.name, p.stem.replace("_", " ").replace("-", " ").title()))
            for p in sorted(DOCS_DIR.glob("*.pdf"))]

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
