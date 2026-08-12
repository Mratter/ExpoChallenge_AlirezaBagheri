# Final scenario-family analysis

This descriptive supplement compares the shipped v4 policy with the tuned constant rule and preparedness teacher on the same 200-case final roster. Each row contains the same 40 seeds. The budget, shock-probability, and severity columns are family-definition parameters, not measurements inferred from planner behavior.

| Final family | Shipped v4 PPO | Tuned constant rule | Preparedness teacher | Daily budget center | Base shock probability | Severity ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Food access | **38 / 40** | **38 / 40** | 39 / 40 | 144 | 0.23 | 0.31 |
| Coastal isolation | **34 / 40** | 30 / 40 | 26 / 40 | 157 | 0.28 | 0.35 |
| Public health | **34 / 40** | 29 / 40 | 30 / 40 | 198 | 0.30 | 0.36 |
| Grid cascade | **31 / 40** | 30 / 40 | 28 / 40 | 168 | 0.26 | 0.34 |
| Aftershock corridor | **26 / 40** | 20 / 40 | 16 / 40 | **136** | **0.30** | **0.36** |

Every planner has its lowest solve count on aftershock corridor. That family combines the lowest budget center, 136, with the joint-highest base shock probability, 0.30, and joint-highest severity ceiling, 0.36. It is therefore the explicitly most resource-constrained high-shock construction in this roster, rather than an isolated learned-policy failure. This is an observed association across five designed families, not a causal estimate of any individual parameter.

The learned policy's margin over both hand-coded comparators is widest on that hardest family: **+6** cases over the tuned rule and **+10** over the teacher. On food access, the tuned rule ties the policy exactly at **38 / 40**. The pattern is consistent with learned allocation adding its clearest value under scarcity; it does not by itself identify which family parameter causes the margin.

## Evidence boundary

- Shipped-policy counts are aggregated from the immutable [single-use final success receipt](../../internal/evaluation_runs/v4/final-evaluation-200.success.json).
- Tuned-rule counts are aggregated from the retained per-case rows in the [matched final oracle receipt](../../internal/developmental_runs/v4/clairvoyant-oracle-200-final.json); the tuned rule is the oracle's deterministic warm start, not the privileged oracle itself.
- Preparedness-teacher counts are the owner-verified family breakdown of the deterministic final regression whose aggregate is pinned at **139 / 200** in `tests/test_consolidation_gate.py`. They sum to that pinned aggregate; the repository does not claim that these five rows came from the learned-policy final receipt.
- Family construction values are read from `backend/app/city/scenarios.py`. Public health ties aftershock corridor at probability 0.30 and severity ceiling 0.36, while its budget center is 198 rather than 136.

No planner was rerun to produce this supplement. The canonical aggregate table, learned-policy Wilson interval, safety invariants, oracle pairing, and evidence hashes remain in [Final 200-case results](final-results-200.md).
