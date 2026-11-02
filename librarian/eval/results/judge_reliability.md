# Judge Reliability Report (Project 1 harness checks)
Judge: `gpt-4.1-mini` (JUDGE_SYSTEM in eval/eval_generation.py). Test cost: $0.026.

## 1. Agreement with humans
- **Not measured yet:** waiting for human labels (0/60 items labelled twice).

## 2. Position bias (swap test)
- Pairs (baseline vs optimised answer to the same question): 30
- Verdict followed the position instead of the answer: **1/30 (3.3%)**
- Consistent in both orders: 20/30 (prefers baseline 5, prefers optimised 2, rest ties)

## 3. Verbosity bias (padding test)
- Clearly wrong answers (the gold answer of another question): 30
- Wrongly accepted even without padding: 2/30
- Became 'correct' after adding polite filler: **0/30 (0.0%)**
