"""
STEP 3 (online): Find the chunks that are most likely to contain the answer.

Four ways to search (these are exactly the four rows of the ablation table):
  keyword        - BM25: good at exact words ("CGPA", "Dean's List", course codes)
  vector         - embeddings: good at meaning ("can I redo a failed class?" ~ "Retaking Courses")
  hybrid         - both, merged with Reciprocal Rank Fusion (RRF)
  hybrid+rerank  - hybrid, then a small LLM re-scores each candidate for relevance (0-10)
"""
import json
import re
import numpy as np
from rank_bm25 import BM25Okapi

from .config import INDEX_DIR
from .index import load_index
from .llm import embed, chat, Usage

STOPWORDS = set("""a an the of to in on for and or is are be by with at as it this that what which who
how do does can i my me we our you your if from when where will would should there their""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class Retriever:
    def __init__(self, index_dir=INDEX_DIR):
        self.chunks, self.vectors = load_index(index_dir)
        # BM25 is cheap to build, so we rebuild it at start-up instead of saving it.
        self.bm25 = BM25Okapi([tokenize(c["section"] + " " + c["text"]) for c in self.chunks])

    # ----- helpers -----
    def _allowed(self, program):
        """Metadata filter: only search the guide for the student's program (or all)."""
        return np.array([program in (None, "all") or c["program"] == program for c in self.chunks])

    # ----- the 4 search modes -----
    def keyword_search(self, query, k, program=None):
        scores = self.bm25.get_scores(tokenize(query))
        scores[~self._allowed(program)] = -np.inf
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def vector_search(self, query, k, program=None, usage=None, query_vec=None):
        if query_vec is None:
            query_vec = embed([query], usage)[0]
        sims = self.vectors @ query_vec          # cosine similarity (vectors are normalised)
        sims[~self._allowed(program)] = -np.inf
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def hybrid_search(self, query, k, program=None, usage=None):
        """
        Reciprocal Rank Fusion: score = sum over both lists of 1 / (60 + rank).
        It only uses ranks, so we never have to compare BM25 scores with cosine scores.
        """
        kw = self.keyword_search(query, k * 2, program)
        vec = self.vector_search(query, k * 2, program, usage)
        fused = {}
        for results in (kw, vec):
            for rank, (idx, _) in enumerate(results):
                fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        # "Confident" = keyword search and vector search agree on the best chunk.
        confident = bool(kw and vec and kw[0][0] == vec[0][0])
        return ranked, confident

    def rerank(self, query, candidates, model, usage=None):
        """
        Ask a small LLM to score every candidate 0-10 for "does this passage answer the question".
        This plays the role of a cross-encoder: it reads question and passage *together*,
        which is more accurate than comparing two separate embeddings.
        """
        passages = "\n\n".join(
            f"[{n}] ({self.chunks[i]['section']})\n{self.chunks[i]['text'][:1200]}"
            for n, (i, _) in enumerate(candidates))
        system = ("You rate how useful each passage is for answering the question. "
                  "10 = directly answers it, 5 = related but incomplete, 0 = unrelated. "
                  'Reply in JSON: {"scores": [score for passage 0, score for passage 1, ...]}')
        raw = chat(model, system, f"Question: {query}\n\nPassages:\n{passages}", usage,
                   json_mode=True, max_tokens=200)
        try:
            scores = json.loads(raw)["scores"]
        except (ValueError, KeyError):
            scores = []
        scores = (list(scores) + [0] * len(candidates))[:len(candidates)]
        ranked = sorted(zip([i for i, _ in candidates], scores), key=lambda x: -float(x[1]))
        return [(i, float(s)) for i, s in ranked]

    # ----- one entry point used by the pipeline and by the evals -----
    def search(self, query, mode, k, program=None, rerank_model="gpt-4o-mini",
               skip_rerank_if_confident=False, usage=None):
        """Returns (ranked list of (chunk_index, score), info dict for the UI)."""
        info = {"mode": mode, "reranked": False, "confident": None}
        if mode == "keyword":
            return self.keyword_search(query, k, program), info
        if mode == "vector":
            return self.vector_search(query, k, program, usage), info

        ranked, confident = self.hybrid_search(query, k, program, usage)
        info["confident"] = confident
        if mode == "hybrid" or (skip_rerank_if_confident and confident):
            return ranked, info
        info["reranked"] = True
        return self.rerank(query, ranked, rerank_model, usage), info
