| Step | Accuracy | p50 (s) | p95 (s) | List-price cost / request | vs baseline | Billed cost / request |
|---|---|---|---|---|---|---|
| 0. baseline | 99.2% | 3.25 | 4.48 | $0.000974 | +0% | $0.000670 |
| 1. + gpt-4o-mini for answers | 99.2% | 3.54 | 4.74 | $0.000596 | -39% | $0.000431 |
| 2. + 3 chunks instead of 5 | 98.4% | 3.26 | 4.52 | $0.000527 | -46% | $0.000366 |
| 3. + no query rewrite | 99.2% | 2.48 | 3.72 | $0.000510 | -48% | $0.000459 |
| 4. + skip reranker when confident | 99.2% | 1.96 | 3.37 | $0.000349 | -64% | $0.000254 |
