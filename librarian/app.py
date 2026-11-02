"""
The Librarian - Streamlit front end.

Run:  streamlit run app.py
"""
import json
import time

import pandas as pd
import streamlit as st

from rag.config import OPTIMISED, DATA_DIR, RESULTS_DIR, list_documents
from rag.pipeline import Librarian

st.set_page_config(page_title="The Librarian - MedTech", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading the documents (first start after adding a PDF takes ~10 s)...")
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
    st.caption("Ask anything about MedTech: rules, exams, grades, programs, masters, "
               "certifications... Every answer cites the document and page it came from.")

    # The app always searches ALL documents with the optimised pipeline (no choices to make).
    settings = OPTIMISED

    docs = list_documents()
    with st.expander(f"📂 {len(docs)} documents searched"):
        for _, title in docs:
            st.markdown(f"- {title}")
        st.caption("To add one: put the PDF in docs/ and restart the app.")

    show_details = st.toggle("Show what happens behind the scenes", value=True)

    st.divider()
    st.markdown("**Try:**")
    examples = ["What happens if I miss an exam?",
                "How do I get on the Dean's List?",
                "What does the Certified Ethical Hacker program cover?",
                "Who is the Executive Master in Digital Health and AI for?",
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

    question = st.chat_input("Ask a question about MedTech...")
    question = question or st.session_state.pop("pending", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="📚"):
            with st.spinner("Searching the documents..."):
                result = get_librarian().ask(question, settings)
            render_answer(result, str(len(st.session_state.history)))
        st.session_state.history.append((question, result))

# ---------------------------------------------------------------- how it works tab
with tab_how:
    st.header("How The Librarian answers a question")
    st.markdown(f"""
This is a **RAG** system (*Retrieval-Augmented Generation*): instead of trusting what the
language model remembers, we first **find** the relevant pages in MedTech's
{len(list_documents())} documents, then ask the model to **answer only from those pages** and cite them.
""")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontname="Helvetica"];
  subgraph cluster_off { label="Offline (automatic when docs/ changes)"; style=dashed;
    pdf [label="PDFs in docs/"]; chunk [label="Chunk\n200 words, 40 overlap"];
    emb [label="Embed\ntext-embedding-3-small"]; store [label="Flat file index\n+ BM25"];
    pdf -> chunk -> emb -> store; }
  q [label="Question", fillcolor="#fff4d6"]; cache [label="Asked before?\n(cache)"];
  ret [label="1. Hybrid search\nBM25 + vectors (RRF)"]; rr [label="2. Rerank\nsmall LLM scores 0-10"];
  gen [label="3. Generate\nwith [n] citations"]; idk [label="4. I don't know", fillcolor="#ffe0e0"];
  a [label="Answer + sources", fillcolor="#e0ffe4"];
  q -> cache -> ret -> rr -> gen -> a; cache -> a [label="yes: instant, $0", style=dashed];
  ret -> gen [label="both searches agree:\nskip rerank", style=dashed];
  rr -> idk [label="best score < 3", style=dashed]; store -> ret;
}""")
    st.subheader("The steps, in plain words")
    st.markdown("""
| Step | What it does | Why |
|---|---|---|
| **Chunking** | Cut each PDF page into ~200-word pieces that overlap by 40 words. Each piece keeps its document, page and section title. | The model can't read every document for every question; small pieces are cheaper and more precise. The metadata gives us citations. |
| **Embedding** | Turn every chunk into a vector (a list of 1536 numbers) that captures its *meaning*. | Lets us find "retake a failed class" even when the guide says "Retaking Courses". |
| **Keyword index (BM25)** | Classic search-engine scoring on exact words. | Great for exact terms: *CGPA*, *Dean's List*, *CHFI*, *BIM*. |
| **Cache** | If the exact same question was asked before, reuse that answer. | Popular questions cost nothing and come back instantly. |
| **1. Hybrid search** | Run both searches and merge them with *Reciprocal Rank Fusion*: each chunk gets `1/(60+rank)` from each list. | Each search catches what the other misses. RRF only uses ranks, so we never compare apples (BM25) to oranges (cosine). |
| **2. Rerank** | A small LLM reads the question with each of the top 10 chunks and scores relevance 0–10. **Skipped** when keyword and vector search already agree on the best chunk (about half the time). | Reading question and passage *together* is more accurate, but it costs time, so we only do it when the search is unsure. |
| **3. Generate** | `gpt-4o-mini` sees only the top 3 chunks, numbered [1], [2], [3], and must cite one after every fact. | Every claim can be checked against its page. |
| **4. I don't know** | If the best rerank score is below 3, or the model finds no answer in the sources, it says so. | A wrong answer about exam rules is worse than no answer. |
""")
    st.subheader("How we made it cheap and fast")
    st.markdown("""
Our first version rewrote every question with an LLM, always reranked, sent 5 chunks and used a
bigger answer model. Measuring each change on the same test set showed we could drop all of
that **without losing accuracy**: about **3× cheaper** and **2× faster** (details in the Evaluation tab).
""")

# ---------------------------------------------------------------- evaluation tab
with tab_eval:
    st.header("How good is it? (measured, not guessed)")
    qs = json.loads((DATA_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    n_ans = sum(q["answerable"] for q in qs)
    st.caption(f"All numbers come from the scripts in `eval/` run on `data/eval_questions.json` "
               f"({n_ans} answerable questions with their source passage + {len(qs) - n_ans} "
               f"the documents can't answer). *Baseline* = our first version, *optimised* = what the app runs.")

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
            st.markdown("**What each optimisation did** (applied one after the other):")
            st.dataframe(pd.DataFrame(steps), hide_index=True, use_container_width=True)
        with st.expander("See every question and answer (optimised)"):
            st.dataframe(pd.DataFrame(gens["optimised"]["rows"])[
                ["question", "gold", "answer", "correct", "faithful", "reason"]], use_container_width=True)
    else:
        st.info("Run `python -m eval.eval_generation --config baseline` and `--config optimised`.")

    st.subheader("3. Can we trust the judge? (Project 1 harness checks)")
    rep = load("judge_reliability.json")
    if rep:
        h, p, vb = rep["human_agreement"], rep["position_bias"], rep["verbosity_bias"]
        st.dataframe(pd.DataFrame([
            {"Check": "Agreement with humans (Cohen's kappa)",
             "Result": h["judge_human_kappa"] if h["status"] == "ok" else "waiting for human labels"},
            {"Check": "Position bias: verdict followed the order, not the answer",
             "Result": f"{p['flip_count']}/{p['pairs']} ({p['flip_rate_pct']}%)"},
            {"Check": "Verbosity bias: wrong answer became 'correct' after padding",
             "Result": f"{vb['flip_count']}/{vb['items']} ({vb['flip_rate_pct']}%)"},
        ]), hide_index=True, use_container_width=True)
    else:
        st.info("Run `python -m eval.judge_reliability` to fill this section.")

    st.subheader("4. Feedback from real users")
    fb = DATA_DIR / "feedback.jsonl"
    if fb.exists():
        votes = [json.loads(l) for l in fb.read_text(encoding="utf-8").splitlines() if l.strip()]
        st.write(f"{sum(v['vote'] > 0 for v in votes)} 👍 · {sum(v['vote'] < 0 for v in votes)} 👎")
        st.dataframe(pd.DataFrame(votes)[["q", "a", "vote"]], use_container_width=True)
    else:
        st.caption("No feedback yet — use the 👍 / 👎 buttons under each answer.")
