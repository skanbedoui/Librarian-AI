"""
Functions copied UNCHANGED from our Project 1 harness (AI-Grader, commit a9ddb95):

    parse_json_response   <- AI-Grader/harness/judge_client.py
    compute_cohen_kappa   <- AI-Grader/harness/reliability.py
    human_consensus_label <- AI-Grader/harness/reliability.py

Labels are "good" / "bad", exactly as in Project 1. For Project 2 an answer is "good" when it is
correct (or says "I don't know" for a question the documents can't answer).
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON object from model output, handling potential markdown code blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove triple backticks and potential json language identifier
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    
    try:
        val = json.loads(cleaned)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass
    
    # Try finding the first JSON object using regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            val = json.loads(match.group(0))
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON object from response: {text!r}")


def compute_cohen_kappa(y_true: list[str], y_pred: list[str]) -> tuple[float, float]:
    """Compute percentage agreement and Cohen's Kappa for binary labels ('good' / 'bad').
    
    Returns:
        (pct_agreement, kappa)
    """
    if not y_true or len(y_true) != len(y_pred):
        return 0.0, 0.0

    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except ImportError:
        # Fallback pure-python Cohen's kappa implementation
        n = len(y_true)
        a = sum(1 for t, p in zip(y_true, y_pred) if t == "good" and p == "good")
        b = sum(1 for t, p in zip(y_true, y_pred) if t == "good" and p == "bad")
        c = sum(1 for t, p in zip(y_true, y_pred) if t == "bad" and p == "good")
        d = sum(1 for t, p in zip(y_true, y_pred) if t == "bad" and p == "bad")
        
        po = (a + d) / n
        p_a1 = (a + b) / n
        p_a0 = (c + d) / n
        p_b1 = (a + c) / n
        p_b0 = (b + d) / n
        pe = (p_a1 * p_b1) + (p_a0 * p_b0)
        
        kappa = (po - pe) / (1.0 - pe) if abs(1.0 - pe) > 1e-9 else 1.0

    pct_agree = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    return round(pct_agree * 100, 2), round(kappa, 4)


def human_consensus_label(label_1: str, label_2: str) -> str:
    """Determine human consensus label from double-labelling."""
    l1 = str(label_1).lower().strip()
    l2 = str(label_2).lower().strip()
    if l1 == l2:
        return l1
    if l1 == "good" or l2 == "good":
        return "good"
    return "bad"
