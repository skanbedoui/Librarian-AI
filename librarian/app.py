"""
The Librarian - Streamlit front end.

Run:  streamlit run app.py
"""
import json
import time
from dataclasses import replace

import pandas as pd
import streamlit as st

from rag.config import BASELINE, OPTIMISED, DATA_DIR, RESULTS_DIR, DOCUMENTS
from rag.pipeline import Librarian

st.set_page_config(page_title="The Librarian - MedTech", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading the index...")
def get_librarian():
    # Loaded once and shared by every user (so the answer cache is shared too).
    return Librarian()


def save_feedback(entry: dict):
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 The Librarian")
    st.caption("Ask anything about the MedTech 2026-2027 student guides. "
               "Every answer cites the page it came from.")

    program = st.radio("Which program are you in?",
                       ["all", "engineering", "licence_master"],
                       format_func={"all": "Both guides", "engineering": "Pre-Engineering / Engineering",
                                    "licence_master": "Licence / Master"}.get)

    preset = st.radio("Pipeline version", ["Optimised (Week 8)", "Baseline (Week 7)", "Custom"],
                      help="Optimised is cheaper and faster. Baseline is the first working version.")
    settings = OPTIMISED if preset.startswith("Optimised") else BASELINE
    if preset == "Custom":
        settings = replace(
            BASELINE,
            search_mode=st.selectbox("Search mode", ["hybrid+rerank", "hybrid", "vector", "keyword"]),
            n_context=st.slider("Chunks sent to the LLM", 1, 10, 5),
            gen_model=st.selectbox("Answer model", ["gpt-4.1-mini", "gpt-4o-mini"]),
            rewrite_query=st.checkbox("Rewrite the question first", True),
            skip_rerank_if_confident=st.checkbox("Skip reranker when search is confident", False),
            use_cache=st.checkbox("Cache repeated questions", False),
        )
    show_details = st.toggle("Show what happens behind the scenes", value=True)

    st.divider()
    st.markdown("**Try:**")
    examples = ["What CGPA puts me on academic probation?",
                "Can I retake a course I failed?",
                "What happens if I miss an exam?",
                "How do I get on the Dean's List?",
                "What is the cafeteria menu?"]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex


tab_chat, tab_how, tab_eval = st.tabs(["💬 Ask", "🔍 How it works", "📊 Evaluation"])

# ---------------------------------------------------------------- chat tab
with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []

    def render_answer(r: dict, key: str):
        st.markdown(r["answer"])
        if r["citations"]:
            with st.expander(f"📄 Sources ({len(r['citations'])})", expanded=False):
                for n, c in r["citations"].items():
                    st.markdown(f"**[{n}] {c['doc']} — page {c['page']}**  \n*{c['section'] or 'no section'}*")
                    st.caption(c["text"])
        meta = f"⏱ {r['latency']:.1f}s · 💲{r['cost']:.5f}"
        if r["cached"]:
            meta += " · ⚡ answered from cache"
        st.caption(meta)

        if show_details:
            with st.expander("🔧 Behind the scenes"):
                info = r["search_info"]
                if r["rewritten_query"] != r["question"]:
                    st.markdown(f"**1. Search query** (your question + LLM rewrite): {r['rewritten_query']}")
                else:
                    st.markdown("**1. Rewrite:** skipped (question used as-is)")
                rr = ("yes" if info["reranked"] else
                      "skipped — keyword and vector search agreed on the top chunk" if info.get("confident")
                      else "no")
                st.markdown(f"**2. Search mode:** `{info['mode']}` · **reranked:** {rr}")
                st.markdown("**3. Top retrieved chunks** (score = rerank 0–10, or search score):")
                st.dataframe(pd.DataFrame([{"score": round(s, 3), "doc": c["doc"][:30], "page": c["page"],
                                            "section": c["section"][-50:], "text": c["text"][:120] + "..."}
                                           for c, s in r["retrieved"][:8]]),
                             hide_index=True, use_container_width=True)
                st.markdown("**4. Time per step (s):** " +
                            ", ".join(f"{k} {v:.2f}" for k, v in r["timings"].items()))
                st.markdown("**5. API calls:**")
                st.dataframe(pd.DataFrame(r["calls"]), hide_index=True)

        c1, c2, _ = st.columns([1, 1, 8])
        if c1.button("👍", key=f"up{key}"):
            save_feedback({"t": time.time(), "q": r["question"], "a": r["answer"], "vote": 1})
            st.toast("Thanks for the feedback!")
        if c2.button("👎", key=f"down{key}"):
            save_feedback({"t": time.time(), "q": r["question"], "a": r["answer"], "vote": -1})
            st.toast("Thanks — we'll look at this one.")

    for i, (q, r) in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="📚"):
            render_answer(r, str(i))

    question = st.chat_input("Ask a question about the student guides...")
    question = question or st.session_state.pop("pending", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="📚"):
            with st.spinner("Searching the guides..."):
                result = get_librarian().ask(question, settings, program)
            render_answer(result, str(len(st.session_state.history)))
        st.session_state.history.append((question, result))

# ---------------------------------------------------------------- how it works tab
with tab_how:
    st.header("How The Librarian answers a question")
    st.markdown("""
This is a **RAG** system (*Retrieval-Augmented Generation*): instead of trusting what the
language model remembers, we first **find** the relevant pages in the student guides, then
ask the model to **answer only from those pages** and cite them.
""")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontname="Helvetica"];
  subgraph cluster_off { label="Offline (once)"; style=dashed;
    pdf [label="2 PDF guides"]; chunk [label="Chunk\\n200 words, 40 overlap"];
    emb [label="Embed\\ntext-embedding-3-small"]; store [label="Flat file index\\n+ BM25"];
    pdf -> chunk -> emb -> store; }
  q [label="Question", fillcolor="#fff4d6"]; rw [label="1. Rewrite"];
  ret [label="2. Hybrid search\\nBM25 + vectors (RRF)"]; rr [label="3. Rerank\\nsmall LLM scores 0-10"];
  gen [label="4. Generate\\nwith [n] citations"]; idk [label="5. I don't know", fillcolor="#ffe0e0"];
  a [label="Answer + sources", fillcolor="#e0ffe4"];
  q -> rw -> ret -> rr -> gen -> a; rr -> idk [label="best score < 3", style=dashed]; store -> ret;
}""")
    st.subheader("The steps, in plain words")
    st.markdown("""
| Step | What it does | Why |
|---|---|---|
| **Chunking** | Cut each PDF page into ~200-word pieces that overlap by 40 words. Each piece keeps its document, page and section title. | The model can't read 65 pages per question; small pieces are cheaper and more precise. The metadata gives us citations. |
| **Embedding** | Turn every chunk into a vector (a list of 1536 numbers) that captures its *meaning*. | Lets us find "retake a failed class" even when the guide says "Retaking Courses". |
| **Keyword index (BM25)** | Classic search-engine scoring on exact words. | Great for exact terms: *CGPA*, *Dean's List*, *EQUALITY scholarship*. |
| **1. Rewrite** | A small LLM rewrites short/vague questions into a clear search query. | "probation?" → "What CGPA puts a student on academic probation?" |
| **2. Hybrid search** | Run both searches and merge them with *Reciprocal Rank Fusion*: each chunk gets `1/(60+rank)` from each list. | Each search catches what the other misses. RRF only uses ranks, so we never compare apples (BM25) to oranges (cosine). |
| **Metadata filter** | If you pick your program in the sidebar, only that guide is searched. | The two guides have different rules for some topics. |
| **3. Rerank** | A small LLM reads the question with each of the top 10 chunks and scores relevance 0–10. | Reading question and passage *together* is more accurate than comparing two vectors. |
| **4. Generate** | The answer model sees only the top chunks, numbered [1], [2]... and must cite one after every fact. | Every claim can be checked against its page. |
| **5. I don't know** | If the best rerank score is below 3, or the model finds no answer in the sources, it says so. | A wrong answer about exam rules is worse than no answer. |
""")
    st.subheader("Baseline vs Optimised")
    st.markdown("""
| | Baseline (Week 7) | Optimised (Week 8) |
|---|---|---|
| Rewrite question | yes | **no** (one LLM call saved) |
| Reranker | always | **skipped when keyword & vector search agree** on the top chunk |
| Chunks sent to the LLM | 5 | **3** |
| Answer model | gpt-4.1-mini | **gpt-4o-mini** (≈2.7× cheaper) |
| Answer cache | no | **yes** — a repeated question costs $0 and returns instantly |
""")

# ---------------------------------------------------------------- evaluation tab
with tab_eval:
    st.header("How good is it? (measured, not guessed)")
    st.caption("All numbers come from the scripts in `eval/` run on `data/eval_questions.json` "
               "(108 answerable questions with their source passage + 14 the guides can't answer).")

    def load(name):
        p = RESULTS_DIR / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    st.subheader("1. Retrieval — is the right passage in the top k?")
    ret = load("retrieval.json")
    if ret:
        df = pd.DataFrame(ret).T
        st.dataframe(df[["recall@1", "recall@3", "recall@5", "recall@10", "mrr",
                         "latency_p50_ms", "latency_p95_ms"]].style.format("{:.2f}"),
                     use_container_width=True)
        st.bar_chart(df[["recall@1", "recall@3"]])
        st.caption("Ablation: keyword only → vector only → hybrid → hybrid + reranker. "
                   "The reranker's main gain is putting the right passage *first* (recall@1).")
    else:
        st.info("Run `python -m eval.eval_retrieval` to fill this section.")

    sweep = load("chunk_sweep.json")
    if sweep:
        st.markdown("**Chunk size test** (hybrid search):")
        st.dataframe(pd.DataFrame(sweep), hide_index=True)

    st.subheader("2. Generation & service — baseline vs optimised")
    gens = {c: load(f"generation_{c}.json") for c in ("baseline", "optimised")}
    if all(gens.values()):
        rows = {c: g["summary"] for c, g in gens.items()}
        keys = ["accuracy_overall", "correctness_answerable", "faithfulness", "citation_accuracy",
                "idk_when_should", "idk_when_should_not", "latency_p50_s", "latency_p95_s",
                "cost_per_request_list_usd", "cost_per_request_usd",
                "cost_per_month_today_usd", "cost_per_month_100x_usd"]
        keys = [k for k in keys if all(k in rows[c] for c in rows)]
        table = pd.DataFrame({c: {k: rows[c][k] for k in keys} for c in rows})
        b, o = rows["baseline"], rows["optimised"]
        table["change"] = [f"{(o[k] - b[k]) / b[k] * 100:+.0f}%" if b[k] else "" for k in keys]
        st.dataframe(table, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Cost per request", f"${o['cost_per_request_usd']:.5f}",
                  f"{(o['cost_per_request_usd'] / b['cost_per_request_usd'] - 1) * 100:.0f}%", delta_color="inverse")
        c2.metric("p95 latency", f"{o['latency_p95_s']:.2f}s",
                  f"{(o['latency_p95_s'] / b['latency_p95_s'] - 1) * 100:.0f}%", delta_color="inverse")
        c3.metric("Accuracy", f"{o['accuracy_overall'] * 100:.1f}%",
                  f"{(o['accuracy_overall'] - b['accuracy_overall']) * 100:+.1f} pts")
        st.caption("*List-price cost* = tokens × list price, ignoring OpenAI's automatic prompt-cache "
                   "discount (which varies run to run). *Billed* includes the discount.")
        steps = load("optimisation_steps.json")
        if steps:
            st.markdown("**What each Week 8 change did** (applied one after the other):")
            st.dataframe(pd.DataFrame(steps), hide_index=True, use_container_width=True)
        with st.expander("See every question and answer (optimised)"):
            st.dataframe(pd.DataFrame(gens["optimised"]["rows"])[
                ["question", "gold", "answer", "correct", "faithful", "reason"]], use_container_width=True)
    else:
        st.info("Run `python -m eval.eval_generation --config baseline` and `--config optimised`.")

    st.subheader("3. Feedback from real users")
    fb = DATA_DIR / "feedback.jsonl"
    if fb.exists():
        votes = [json.loads(l) for l in fb.read_text(encoding="utf-8").splitlines() if l.strip()]
        st.write(f"{sum(v['vote'] > 0 for v in votes)} 👍 · {sum(v['vote'] < 0 for v in votes)} 👎")
        st.dataframe(pd.DataFrame(votes)[["q", "a", "vote"]], use_container_width=True)
    else:
        st.caption("No feedback yet — use the 👍 / 👎 buttons under each answer.")
