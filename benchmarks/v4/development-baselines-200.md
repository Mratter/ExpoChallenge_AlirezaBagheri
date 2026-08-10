# Development baselines — 200 cases

All four cheap planners below ran on the same 200 development tapes (5 unchanged families × 40 seeds). This evidence is development-only, nonauthorizing, and contains no learned-v4 final result.

| Planner | Solved / 200 | Wilson 95% CI | Mean resilience AUC | Mean minimum tail margin | Hard violations | Max conservation residual |
|---|---:|---:|---:|---:|---:|---:|
| Reactive heuristic | **91/200** | [0.387, 0.524] | 0.473249 | -0.024201 | 0 | 0.0e+00 |
| Preparedness teacher | **151/200** | [0.691, 0.809] | 0.484241 | +0.024468 | 0 | 0.0e+00 |
| Tuned constant rule | **160/200** | [0.739, 0.850] | 0.483710 | +0.030980 | 0 | 0.0e+00 |
| Legacy ONNX regression fixture | **141/200** | [0.638, 0.764] | 0.495359 | +0.027491 | 0 | 0.0e+00 |

## Historical 40-case evidence

The earlier table at `benchmarks/v4/development-baselines.md` remains byte-identical, receipt-bound historical evidence from the original eight-seed subset. Its PPO, BC, MPC, and rule scores must not be compared numerically with the 200-case results above.

The privileged clairvoyant CEM result remains **37/40 on that original subset only**. It establishes constructive headroom; it is not a submission baseline, a 200-case result, or a mathematical upper bound.

Machine receipt: `internal/developmental_runs/v4/development-baselines-200.json`.
