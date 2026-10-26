"""
The full online pipeline, in the order a question flows through it:

    question
      -> (1) rewrite      make vague / short questions searchable        [optional]
      -> (2) retrieve     keyword + vector search, metadata filter
      -> (3) rerank       small LLM re-scores the candidates              [optional]
      -> (4) generate     answer ONLY from the chunks, cite each claim [n]
      -> (5) "I don't know" if nothing relevant was found
"""
import re
import time

from .config import Settings, BASELINE
from .llm import chat, Usage
from .retrieve import Retriever

IDK = "I don't know based on the student guides."

REWRITE_SYSTEM = (
    "Rewrite the student's question about university rules into a short, specific search query. "
    "Expand abbreviations if obvious (e.g. GPA, CGPA). Keep the student's key terms exactly. "
    "Do NOT add the university's name or generic words. Keep every part of a multi-part question. "
    "Reply with the query only."
)

# This system prompt never changes, so it sits FIRST in every request:
# OpenAI automatically caches long identical prompt prefixes (cheaper + faster).
ANSWER_SYSTEM = f"""You are "The Librarian", an assistant that answers students' questions about the
MedTech student guides (academic policies, grades, exams, scholarships, conduct, calendar...).

Rules:
1. Use ONLY the numbered sources given. Never use outside knowledge.
2. After every sentence that states a fact, cite its source like [1] or [2][3].
3. If NONE of the sources answer the question, reply with exactly this sentence and nothing else:
   {IDK}
   If the sources answer it, even partly, just give the answer. Never start with "I don't know"
   and then answer anyway.
4. If the answer differs between programs (Engineering vs Licence/Master), say so.
5. Be concise: a short paragraph or a few bullet points."""


class Librarian:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()
        self.cache = {}   # (question, settings, program) -> result   (Week 8 optimisation)

    def ask(self, question: str, settings: Settings = BASELINE, program: str | None = None) -> dict:
        t0 = time.perf_counter()
        cache_key = (question.strip().lower(), settings, program)
        if settings.use_cache and cache_key in self.cache:
            hit = dict(self.cache[cache_key])
            hit.update(cached=True, cost=0.0, cost_list=0.0, latency=time.perf_counter() - t0)
            return hit

        usage = Usage()
        timings = {}

        # (1) Rewrite
        query = question
        if settings.rewrite_query:
            t = time.perf_counter()
            rewritten = chat("gpt-4o-mini", REWRITE_SYSTEM, question, usage, max_tokens=80).strip()
            # We search with the original question PLUS the rewrite: the rewrite can add useful
            # words but can never remove the student's own words. (Searching with the rewrite
            # alone lost "Dean's List" questions - see postmortem.md.)
            query = f"{question} {rewritten}"
            timings["rewrite"] = time.perf_counter() - t

        # (2) + (3) Retrieve and maybe rerank
        t = time.perf_counter()
        ranked, info = self.retriever.search(
            query, settings.search_mode, settings.top_k, program,
            settings.rerank_model, settings.skip_rerank_if_confident, usage)
        timings["retrieve"] = time.perf_counter() - t

        # (5) Nothing relevant? Say "I don't know" without paying for generation.
        if info["reranked"] and (not ranked or ranked[0][1] < settings.min_relevance):
            return self._finish(question, query, IDK, [], ranked, info, usage, timings, t0, cache_key, settings)

        context = [self.retriever.chunks[i] for i, _ in ranked[:settings.n_context]]

        # (4) Generate the answer with citations
        t = time.perf_counter()
        sources = "\n\n".join(
            f"[{n}] {c['doc']} | page {c['page']} | {c['section']}\n{c['text']}"
            for n, c in enumerate(context, start=1))
        answer = chat(settings.gen_model, ANSWER_SYSTEM,
                      f"Sources:\n{sources}\n\nQuestion: {question}", usage).strip()
        timings["generate"] = time.perf_counter() - t

        return self._finish(question, query, answer, context, ranked, info, usage, timings, t0, cache_key, settings)

    def _finish(self, question, query, answer, context, ranked, info, usage, timings, t0, cache_key, settings):
        # Keep only the sources the answer actually cites, numbered as in the answer.
        cited_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
        citations = {n: context[n - 1] for n in cited_numbers if 1 <= n <= len(context)}
        result = {
            "question": question,
            "rewritten_query": query,
            "answer": answer,
            "is_idk": answer.lower().startswith(IDK.lower()[:12]),   # "i don't know"
            "citations": citations,               # {number: chunk}
            "context": context,                   # all chunks shown to the LLM
            "retrieved": [(self.retriever.chunks[i], s) for i, s in ranked],
            "search_info": info,
            "timings": timings,
            "latency": time.perf_counter() - t0,
            "cost": usage.cost,
            "cost_list": usage.cost_list,
            "calls": usage.calls,
            "cached": False,
        }
        if settings.use_cache:
            self.cache[cache_key] = result
        return result
