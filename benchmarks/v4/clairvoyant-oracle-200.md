# Matched 200-case privileged clairvoyant-oracle study

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

On final, the same fixed CEM protocol solved **182/200** and the separately authorized frozen policy solved **163/200**. The **163/182 = 89.6%** headline is an aggregate achieved-count ratio; casewise coverage is **162/182 = 89.0%** because one case is policy-only, and the two methods jointly demonstrate solutions on **183/200** cases.
<!-- END ACHIEVED-COUNT REPORTING OVERLAY -->

The CEM oracle sees each case's complete future shock tape. It is a privileged anytime achieved lower bound, **not** a submission baseline, a proven mathematical optimum, or an infeasibility certificate.

## Matched split results

| Split | Oracle solved | Wilson 95% CI | Tuned rule | Selected MPC | Learned v4 policy |
|---|---:|---:|---:|---:|---:|
| Development | **187/200** | [0.8920, 0.9616] | 160/200 | 153/200 | 178/200 (accepted shipped-policy receipt) |
| Final | **182/200** | [0.8622, 0.9423] | 147/200 | 135/200 | 163/200 (later separate owner-authorized receipt) |

The learned v4 model was not run as part of this privileged oracle study. A later, separately owner-authorized evaluation of the already frozen shipped artifact solved 163/200 final cases; see the [canonical final report](final-results-200.md). That result did not feed back into the oracle study, training, or model selection.

## Development casewise comparison

The comparison joins the oracle rows to the already accepted shipped-ONNX development parity receipt; it does not rerun the model.

| Both solve | Policy only | Oracle only | Neither | Demonstrated union |
|---:|---:|---:|---:|---:|
| 177 | 1 | **10** | 12 | 188/200 |

Remaining directly demonstrated headroom is **10 cases**: the oracle solves those identical tapes while the shipped policy does not. Policy-only and oracle-only cases are reported separately, so an aggregate ratio is not presented as casewise ceiling coverage.

## Evidence and safety

Every tuned-rule, selected-MPC, and oracle rollout has zero hard violations and exactly `0.0` maximum conservation residual. The same holds across all evaluated oracle candidates.

The historical **37/40** result remains intact as the original 40-case development-subset diagnostic. This study neither overwrites nor reinterprets that receipt.

| Evidence | SHA-256 |
|---|---|
| [Portable development receipt](../../internal/developmental_runs/v4/clairvoyant-oracle-200-dev.json) raw receipt | `e52464ecbca8d9b4d2838980713ab34be873e9814113f082b76d5bfbdbf9e9fc` |
| [Portable final receipt](../../internal/developmental_runs/v4/clairvoyant-oracle-200-final.json) raw receipt | `dbed01b66a670573c2748c230817dcee292618d4af133a758ee160cab9bece31` |
| Raw study summary | `f42fb1754aa03d48bccc8eb5124bb0092762f2dfccb2030d2360e68f0e250ee2` |
| Raw study protocol | `a1725fb53f9b0ce3835d4da25d53e203b417ad0cb7de7a8c4c8434eed53e02ae` |
| Historical 37/40 receipt | `f037c98d8fec483dfa6b5c9c1691842597a4163c7d1ee6f3e72618f987d671b9` |
| Shipped policy parity receipt | `e3b487df8221db75d58dc68eccbc9df93af16cb0e9f17b5bc60cf50a5b42ba6c` |
| Shipped policy parity rows | `ca9320566b86dfb7a02d2cb9232c7a28c80f08dbbd700dffc6d2af9af1c22d6b` |
| Shipped v4 ONNX | `a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483` |
