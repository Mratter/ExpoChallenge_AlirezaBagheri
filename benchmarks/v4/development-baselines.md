# Development baseline table

All planners use the same 40 development tapes. This table is diagnostic and nonauthorizing; it does not use the final split.

> **Oracle disclosure:** the CEM oracle is privileged and clairvoyant. It sees the complete future shock tape and is **not a submission baseline**.

| Planner | Solved / 40 | Wilson 95% CI | Mean resilience AUC | Mean minimum tail margin | Hard violations | Max conservation residual |
|---|---:|---:|---:|---:|---:|---:|
| Reactive heuristic | 17/40 | [0.285, 0.578] | 0.472401 | -0.023933 | 0 | 0.0e+00 |
| BC teacher | 31/40 | [0.625, 0.877] | 0.484596 | +0.026153 | 0 | 0.0e+00 |
| Tuned constant rule (mult=10.0, cap=0.50) | 33/40 | [0.681, 0.913] | 0.484222 | +0.032135 | 0 | 0.0e+00 |
| BC initialization | 32/40 | [0.652, 0.895] | 0.485856 | +0.028845 | 0 | 0.0e+00 |
| Shipped v3 PPO ONNX | 31/40 | [0.625, 0.877] | 0.496357 | +0.026749 | 0 | 0.0e+00 |
| v4 PPO at 1M active transitions | 35/40 | [0.739, 0.945] | 0.491102 | +0.049687 | 0 | 0.0e+00 |
| Causal MPC (k=1) | 18/40 | [0.307, 0.602] | 0.466546 | +0.000061 | 0 | 0.0e+00 |
| Causal MPC (k=3) | 29/40 | [0.572, 0.839] | 0.476161 | +0.039014 | 0 | 0.0e+00 |
| Causal MPC (k=5) | 30/40 | [0.598, 0.858] | 0.478196 | +0.052685 | 0 | 0.0e+00 |
| Clairvoyant CEM oracle (privileged) | 37/40 | [0.801, 0.974] | 0.497441 | +0.106968 | 0 | 0.0e+00 |

## Paired exact McNemar comparisons

The v4 PPO is the left planner in every comparison.

| Pair | Both | v4 only | Other only | Neither | Exact two-sided p |
|---|---:|---:|---:|---:|---:|
| v4 PPO vs tuned rule | 33 | 2 | 0 | 5 | 0.5 |
| v4 PPO vs BC teacher | 30 | 5 | 1 | 4 | 0.21875 |
| v4 PPO vs shipped v3 PPO | 28 | 7 | 3 | 2 | 0.34375 |
| v4 PPO vs clairvoyant oracle | 35 | 0 | 2 | 3 | 0.5 |

The receipt at `internal/developmental_runs/v4/step6-dev-baseline-table.json` contains the complete paired rows and source hashes.
