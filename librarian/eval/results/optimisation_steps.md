| Step | Accuracy | p50 (s) | p95 (s) | List-price cost / request | vs baseline | Billed cost / request |
|---|---|---|---|---|---|---|
| 0. baseline | 97.2% | 2.42 | 3.17 | $0.000993 | +0% | $0.000676 |
| 1. + gpt-4o-mini for answers | 96.7% | 2.77 | 3.50 | $0.000594 | -40% | $0.000374 |
| 2. + 3 chunks instead of 5 | 95.6% | 2.67 | 3.30 | $0.000526 | -47% | $0.000351 |
| 3. + no query rewrite | 97.2% | 2.06 | 2.88 | $0.000506 | -49% | $0.000366 |
| 4. + skip reranker when confident | 97.2% | 1.61 | 2.66 | $0.000346 | -65% | $0.000243 |
