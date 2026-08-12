# Development baselines — 200 cases

<!-- BEGIN ACHIEVED-COUNT REPORTING OVERLAY -->
## Demonstrated-achievable reference

**Demonstrated-achievable reference denominator = the 187 of 200 development cases solved by the privileged future-aware CEM run; its 13 search failures are not proofs of infeasibility.**

| Development result | Raw solved / 200 | Achieved-count ratio (/187 reference) | Wilson 95% CI on /187 |
|---|---:|---:|---:|
| Privileged CEM | 187/200 | 187/187 = 100.0% | [0.9799, 1.0000] |
| Selected shipped v4 | 178/200 | 178/187 = 95.2% | [0.9111, 0.9745] |
| Five-seed 2M endpoint mean | 171.4/200 | 171.4/187 = 91.7% | not reported: optimizer-seed mean |
| Tuned rule | 160/200 | 160/187 = 85.6% | [0.7981, 0.8988] |
| Preparedness teacher | 151/200 | 151/187 = 80.7% | [0.7450, 0.8576] |
| Selected MPC | 153/200 | 153/187 = 81.8% | [0.7567, 0.8669] |
| Legacy fixture | 141/200 | 141/187 = 75.4% | [0.6876, 0.8102] |
| Reactive heuristic | 91/200 | 91/187 = 48.7% | [0.4160, 0.5578] |

The headline **178/187 = 95.2%** is an aggregate achieved-count ratio; casewise policy coverage is **177/187 = 94.7%** because one case is policy-only. The two methods jointly demonstrate solutions on **188/200** cases, and 10 oracle-only cases demonstrate remaining headroom. Ratios and intervals are descriptive and post-hoc.
<!-- END ACHIEVED-COUNT REPORTING OVERLAY -->

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
