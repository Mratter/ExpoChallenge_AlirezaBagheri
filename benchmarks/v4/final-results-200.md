# Final 200-case results

<!-- BEGIN ACHIEVED-COUNT REPORTING OVERLAY -->
## Demonstrated-achievable reference

The shipped v4 policy solved **163/182 = 89.6%** relative to the privileged CEM achieved-count reference (Wilson 95% **[84.3%, 93.2%]**), alongside its raw **163/200** held-out result.

**Demonstrated-achievable reference denominator = the 182 of 200 final cases solved by the privileged future-aware CEM run; its 18 search failures are not proofs of infeasibility.**

| Method | Raw solved / 200 | Achieved-count ratio (/182 reference) | Wilson 95% CI on /182 |
|---|---:|---:|---:|
| Privileged CEM | 182/200 | 182/182 = 100.0% | [0.9793, 1.0000] |
| Shipped v4 PPO | 163/200 | 163/182 = 89.6% | [0.8427, 0.9321] |
| Tuned rule | 147/200 | 147/182 = 80.8% | [0.7443, 0.8584] |
| Preparedness teacher | 139/200 | 139/182 = 76.4% | [0.6970, 0.8196] |
| Selected MPC | 135/200 | 135/182 = 74.2% | [0.6736, 0.7999] |
| Legacy fixture | 125/200 | 125/182 = 68.7% | [0.6162, 0.7497] |
| Reactive heuristic | 72/200 | 72/182 = 39.6% | [0.3274, 0.4681] |

The headline ratio compares aggregate solved counts, not a contained case set: casewise policy coverage is **162/182 = 89.0%** because one case is policy-only. The two recorded methods jointly demonstrate solutions on **183/200** cases; 20 oracle-only cases demonstrate remaining headroom.

Receipt audit of the oracle's 18 failed searches found **15/18** with nonnegative minimum tail margin (mean +0.03391065; range -0.01072877 to +0.11473788). Failed-check occurrences were 14 resilience-AUC, 9 critical-day-cap, and 3 assessment-tail checks, with overlaps. The portable receipt does not retain numeric day-cap excess, and these search failures do not prove infeasibility.

The /182 intervals are descriptive post-hoc Wilson intervals. They and the raw /200 interval treat cases as Bernoulli units and do not model clustering within five fixed scenario families.
<!-- END ACHIEVED-COUNT REPORTING OVERLAY -->

The shipped v4 row is the single owner-authorized learned-policy final evaluation. The final result was not used for model selection or training.

| Method | Solved | Rate | Wilson 95% CI | Scope |
|---|---:|---:|---:|---|
| Privileged clairvoyant CEM | 182/200 | 0.910 | [0.8622, 0.9423] | privileged anytime achieved lower bound |
| Shipped v4 PPO | **163/200** | **0.815** | **[0.7554, 0.8627]** | owner-authorized learned policy |
| Tuned constant rule | 147/200 | 0.735 | [0.6698, 0.7913] | deterministic oracle warm start |
| Preparedness teacher | 139/200 | 0.695 | [0.6280, 0.7546] | public deterministic regression |
| Selected causal MPC, k=5 | 135/200 | 0.675 | [0.6073, 0.7361] | causal receding-horizon diagnostic |
| Legacy ONNX regression fixture | 125/200 | 0.625 | [0.5561, 0.6891] | retired-policy regression fixture |
| Reactive heuristic | 72/200 | 0.360 | [0.2967, 0.4286] | public deterministic regression |

## Shipped v4 results by scenario family

| Family | Solved | Wilson 95% CI |
|---|---:|---:|
| `v3_final_coastal_isolation` | 34/40 | [0.7093, 0.9294] |
| `v3_final_grid_cascade` | 31/40 | [0.6250, 0.8768] |
| `v3_final_food_access` | 38/40 | [0.8350, 0.9862] |
| `v3_final_aftershock_corridor` | 26/40 | [0.4951, 0.7787] |
| `v3_final_public_health` | 34/40 | [0.7093, 0.9294] |

The 200 cases are clustered within five fixed scenario families. The overall Wilson interval treats case outcomes as Bernoulli observations and does not model within-family dependence, so its precision is slightly overstated; the 40-case family rows expose that heterogeneity directly.

The clairvoyant CEM sees the complete future shock tape. It is a privileged anytime achieved lower bound, not a causal baseline or a proven mathematical ceiling.

## Matched shipped-policy / oracle cases

| Both | Policy only | Oracle only | Neither | Demonstrated union |
|---:|---:|---:|---:|---:|
| 162 | 1 | 20 | 17 | 183/200 |

The aggregate count ratio is 163/182 = 89.6%; casewise policy coverage of oracle-achieved cases is 89.0%. They are reported separately because finite CEM solved sets need not nest.
The shipped policy is 16 solved cases ahead of the strongest hand-coded planner, the tuned constant rule at 147/200.

Every bound result has zero hard violations and exactly `0.0` conservation residual.

## Evidence

- Success receipt SHA-256: `6c21f3be7dc1af8c7bbc00e671210e315e42d6211bd276eccd45adc74421f373`
- Shipped-policy ordered rows SHA-256: `754607ff0b4ef29c42bd2f6fc6a45183744b9c20f1ad5bac3a48eb6b370df405`
- Oracle pairing rows SHA-256: `c7814a378d55a8ccf1a7919b9a907c80bd2bca95bbf88ba1fb4ef8badab9d0e6`
- Shipped v4 ONNX SHA-256: `a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483`
- Shipped artifact manifest SHA-256: `7ecc9948789163febf9cc9a455e20c0d5e5fb75c70919598169f21614e1a5a06`
- Development parity receipt SHA-256: `e3b487df8221db75d58dc68eccbc9df93af16cb0e9f17b5bc60cf50a5b42ba6c`
- Privileged oracle receipt SHA-256: `baf5aa6ec8e419a50f87e744eac7779f30a53b6aab60018ff1a7043126b0b5ec`
- Public/legacy regression gate SHA-256: `97bdeb13556a2fdb9b291c62e699da739441e593ad57f6a5adc014e7ece38638`
