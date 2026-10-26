"""
Thin wrapper around the OpenAI API that also counts tokens and dollars,
so every answer can report exactly what it cost.
"""
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from .config import PRICES, EMBED_MODEL

load_dotenv()


def _get_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:  # Streamlit Cloud stores secrets in st.secrets instead of a .env file
            import streamlit as st
            key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in a .env file (see README).")
    return key


_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=_get_key())
    return _client


class Usage:
    """Adds up tokens and cost across all the API calls of one request."""
    def __init__(self):
        self.cost = 0.0        # what OpenAI actually bills (cached tokens are discounted)
        self.cost_list = 0.0   # same tokens at full price: does not depend on cache luck,
        self.calls = []        # so we use it to compare configurations fairly

    def add(self, model, input_tokens, output_tokens=0, cached_tokens=0):
        price_in, price_out = PRICES.get(model, (0, 0))
        # OpenAI bills cached input tokens at 25% (gpt-4.1) / 50% (gpt-4o) of the normal price;
        # we use 50% as a conservative estimate.
        billed_in = (input_tokens - cached_tokens) + 0.5 * cached_tokens
        cost = (billed_in * price_in + output_tokens * price_out) / 1_000_000
        self.cost += cost
        self.cost_list += (input_tokens * price_in + output_tokens * price_out) / 1_000_000
        self.calls.append({"model": model, "in": input_tokens, "out": output_tokens,
                           "cached": cached_tokens, "cost": cost})


def chat(model, system, user, usage: Usage | None = None, json_mode=False, max_tokens=600) -> str:
    resp = client().chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"} if json_mode else None,
    )
    if usage is not None:
        u = resp.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        usage.add(model, u.prompt_tokens, u.completion_tokens, cached)
    return resp.choices[0].message.content


def embed(texts: list[str], usage: Usage | None = None, batch=100) -> np.ndarray:
    """Return one normalised vector per text (so a dot product = cosine similarity)."""
    vectors = []
    for i in range(0, len(texts), batch):
        resp = client().embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch])
        vectors += [d.embedding for d in resp.data]
        if usage is not None:
            usage.add(EMBED_MODEL, resp.usage.prompt_tokens)
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)
