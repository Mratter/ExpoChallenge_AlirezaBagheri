# Evidence and Results

This document separates measured synthetic results, privileged search diagnostics, the Hurricane Maria reconstruction, and historical evidence. It is the interpretation guide; the linked machine receipts remain the source records.

## Result vocabulary

- **Raw solved count** always uses the full roster denominator: 200 for current development and final evidence.
- **Oracle-solved reference** compares a planner's aggregate count with the number of cases solved by the fixed privileged future-aware CEM search. It is descriptive and does not replace the raw denominator.
- **Casewise coverage** counts how many oracle-solved rows the policy also solved. It differs from the aggregate ratio when the solved sets are not nested.
- **Anytime achieved lower bound** means CEM demonstrated a feasible trajectory for each solved row. A failed search is not an infeasibility certificate.
- **Submission planners** are causal and use only the public state. Privileged CEM sees the complete future shock tape and is never a submission baseline or model-selection input.

## Final synthetic benchmark

The shipped v4 artifact was frozen after development selection and publication. Exactly one owner-authorized learned-policy run was then completed on the disjoint 200-case final roster.

| Final method | Raw solved / 200 | Ratio to 182 oracle-solved cases | Descriptive Wilson 95% CI on /182 | Scope |
| --- | ---: | ---: | ---: | --- |
| Privileged future-aware CEM | **182 / 200** | **182 / 182 = 100.0%** | **[0.9793, 1.0000]** | Anytime oracle-solved reference; not a submission baseline |
| **Shipped v4 PPO** | **163 / 200** | **163 / 182 = 89.6%** | **[0.8427, 0.9321]** | Single owner-authorized learned-policy evaluation |
| Tuned constant rule | **147 / 200** | **147 / 182 = 80.8%** | **[0.7443, 0.8584]** | Public deterministic planner |
| Preparedness teacher | **139 / 200** | **139 / 182 = 76.4%** | **[0.6970, 0.8196]** | Public deterministic planner |
| Causal MPC, `k=5` | **135 / 200** | **135 / 182 = 74.2%** | **[0.6736, 0.7999]** | Receding-horizon diagnostic |
| Legacy ONNX fixture | **125 / 200** | **125 / 182 = 68.7%** | **[0.6162, 0.7497]** | Retired-policy regression fixture |
| Reactive heuristic | **72 / 200** | **72 / 182 = 39.6%** | **[0.3274, 0.4681]** | Runtime public baseline |

The policy's raw result is **163 / 200 (81.5%)**, with receipt-level Wilson 95% interval **[0.7554293724, 0.862698072]**. It is 16 cases ahead of the tuned rule. The raw interval treats the 200 cases as Bernoulli observation units; it does not model dependence inside the five fixed 40-case families, so it may slightly overstate precision. Family rows are reported below.

Privileged CEM solved 182/200 final cases. Its 18 search failures are not proofs of infeasibility. The exact policy/oracle partition is **162 both solved, 1 policy-only, 20 oracle-only, and 17 neither**. Therefore:

- the union is **183 / 200**;
- aggregate policy-to-oracle solved-count ratio is **163 / 182 = 89.6%**; and
- casewise policy coverage of oracle-solved cases is **162 / 182 = 89.0%**.

Those last two quantities are deliberately different because the one policy-only case proves the finite solved sets do not nest. The 20 oracle-only rows are demonstrated remaining headroom for this policy under the fixed search budget.

Every bound final row has zero hard violations and exactly `0.0` maximum conservation residual. The canonical report is [final-results-200.md](../benchmarks/v4/final-results-200.md); its complete machine record is [final-evaluation-200.success.json](../internal/evaluation_runs/v4/final-evaluation-200.success.json).

### Final scenario families

| Final family | Shipped v4 PPO | Tuned rule | Preparedness teacher | Daily budget center | Base shock probability | Severity ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Food access | **38 / 40** | **38 / 40** | 39 / 40 | 144 | 0.23 | 0.31 |
| Coastal isolation | **34 / 40** | 30 / 40 | 26 / 40 | 157 | 0.28 | 0.35 |
| Public health | **34 / 40** | 29 / 40 | 30 / 40 | 198 | 0.30 | 0.36 |
| Grid cascade | **31 / 40** | 30 / 40 | 28 / 40 | 168 | 0.26 | 0.34 |
| Aftershock corridor | **26 / 40** | 20 / 40 | 16 / 40 | **136** | **0.30** | **0.36** |

All three planners have their lowest solve count on aftershock corridor. It combines the lowest budget center with the joint-highest shock probability and severity ceiling. The learned policy's margin is widest there, but this is a descriptive pattern across authored families—not a causal estimate of any one parameter. See the [family-analysis supplement](../benchmarks/v4/final-family-analysis-200.md).

## Development selection and oracle diagnostic

The selected checkpoint was chosen only on the 200-case development roster.

| Development method | Raw solved / 200 | Ratio to 187 oracle-solved cases | Descriptive Wilson 95% CI on /187 |
| --- | ---: | ---: | ---: |
| **Selected v4 PPO, seed 67017 at 1M** | **178 / 200** | **178 / 187 = 95.2%** | **[0.9111, 0.9745]** |
| Privileged future-aware CEM | **187 / 200** | **187 / 187 = 100.0%** | **[0.9799, 1.0000]** |
| Tuned constant rule | **160 / 200** | **160 / 187 = 85.6%** | **[0.7981, 0.8988]** |
| Causal MPC, `k=5` | **153 / 200** | **153 / 187 = 81.8%** | **[0.7567, 0.8669]** |
| Preparedness teacher | **151 / 200** | **151 / 187 = 80.7%** | **[0.7450, 0.8576]** |
| Legacy ONNX fixture | **141 / 200** | **141 / 187 = 75.4%** | **[0.6876, 0.8102]** |
| Reactive heuristic | **91 / 200** | **91 / 187 = 48.7%** | **[0.4160, 0.5578]** |

CEM's 187 solved rows form the development oracle-solved reference; its 13 search failures do not certify impossibility. The matched policy/oracle partition is **177 both solved, 1 policy-only, 10 oracle-only, and 12 neither**. Their union is **188 / 200**. Aggregate ratio is **178 / 187 = 95.2%**, while casewise coverage is **177 / 187 = 94.7%**.

The five registered 2M PPO endpoints solved 172, 171, 171, 174, and 169 cases: mean **171.4 / 200**, or **171.4 / 187 = 91.7%** of the oracle-solved reference, population standard deviation **1.62**, and sample standard deviation **1.82**. No Wilson interval is reported for this optimizer-seed mean.

Selection ranked 20 checkpoints by solve count, then earlier transition count and lower seed. Seed `67017` at 1M won at 178/200; the runner-up solved 174/200. Full SB3-to-ONNX parity reproduced all 178 outcomes over 6,000 action vectors and 132,000 action elements. The FastAPI `POST -> persist -> GET` replay also reproduced all 178 development outcomes.

## What Solved means

A planner is Solved only if all six frozen checks pass:

| Check | Frozen requirement |
| --- | --- |
| Assessment-tail targets | Every service is at or above its public target on every one of days 28–30; canonical targets are `0.55` |
| Resilience AUC | Mean priority-weighted daily resilience is at least `0.44` |
| Critical service-days | No more than 12 of 150 service-days are below `0.30` |
| Hard constraints | Total hard-violation count is exactly `0` |
| Material conservation | Maximum absolute logistics conservation residual is at most `1e-6` |
| Terminal pipeline | Pending arrivals at day 30 do not exceed depot capacity |

The definition hash is `d033c42b43ade8fff3c3b2d11f92adcf7567b4221b3b16d798a8f0afc896df82`. The API returns each check, per-service tail values, reason codes, definition ID, and hash. The browser does not infer a different verdict.

The conjunction resists simple terminal-score gaming. A strong service cannot hide a weak one; a day-30 spike cannot hide failure earlier in the assessment tail; AUC rejects neglect followed by a sprint; the critical-day limit rejects prolonged collapse; and the terminal-pipeline check rejects end-game inventory dumping.

Calibration spans both sides. The retained legacy reactive benchmark solved 14/40, while privileged future-aware search failed 3/40 on the historical development subset. On expanded rosters, fixed-budget CEM achieved 187/200 on development and 182/200 on final. These results show the rule is neither automatic nor unreachable, while preserving the distinction between demonstrated solutions and proven optimality.

Resilience AUC remains useful as a secondary trajectory-quality measure. It cannot replace the complete tail-target, critical-day, feasibility, conservation, and pipeline definition. A planner's primary result is how many disasters it independently Solved.

## Worked example

Historical case `v3_dev_health_compound:820007` is a narrow legacy-model success. It contains 14 shocks in 30 days, 209 material per day, and 153 crew per day.

| Service | Day 0 | Day 30 | Target |
| --- | ---: | ---: | ---: |
| Transport | 34.5% | 62.7% | 55% |
| Housing | 42.6% | 59.2% | 55% |
| Food | 26.3% | 64.5% | 55% |
| Healthcare | 23.6% | 71.0% | 55% |
| Public services | 23.6% | 66.1% | 55% |

It passes three checks narrowly at once: housing tail margin **+0.75 percentage points**, resilience AUC **0.44171** against the `0.44` floor, and exactly **12 critical service-days** of the 12 allowed. Healthcare begins tied for lowest at 23.6% but finishes highest at 71.0%, consistent with its highest priority of 1.88.

## Why the synthetic task is hard

On the retained original 40-case development subset, a 30-day case contains a mean of **10.6 shocks**, ranging from 4 to 16. The last shock occurs on days 25–27 in **32 of 40 cases (80%)**, including day 27 in 13 cases. Shocks are blocked during days 28–30.

Recovery is concave and slow, so the last shock often cannot be repaired reactively before assessment. The policy must invest in preparedness using causal risk probabilities without seeing the future tape. On that historical subset, mean minimum tail margin rose **0.0288 -> 0.0329 -> 0.0397 -> 0.0497** across BC, 200k, 500k, and 1M transitions.

## Post-release development studies

These studies used development cases only and did not change the shipped artifact or retained final result:

| Study | Registered result | Decision |
| --- | --- | --- |
| Oracle-distilled actor + adopted PPO | 2M endpoints 178, 174, 170; mean 174.0 | Best tied 178 and missed the 183 promotion threshold |
| Large `[768,512,256]` network at `3e-5` | 2M endpoints 178, 176, 175; mean 176.33 | Positive large-plus-low-LR signal, but best tied 178 and smaller-network LR control was not run |
| 2× weak-family sampling | 2M endpoints 175, 170, 172; mean 172.33 | Best selectable checkpoint 176; treatment also increased imitation exposure by 33.3% |
| Combined large + oracle-distilled attempt | Two complete curves; third seed stopped at critic EV `0.4789480567` | Incomplete, nonfactorial, not promotable, and not retried |

Full reports: [oracle distillation](../benchmarks/v4/oracle-distilled-ppo-study-200.md), [network capacity](../benchmarks/v4/network-capacity-study-200.md), [family sampling](../benchmarks/v4/moderate-family-study-200.md), and [incomplete combined attempt](../benchmarks/v4/combined-distilled-large-study-200.incomplete.md).

## Historical original 40-case subset

These measurements used eight seeds per family and are not numerically interchangeable with the expanded 200-case studies.

| Historical method | Solved / 40 |
| --- | ---: |
| v4 PPO at 1M | **35 / 40** |
| Tuned constant rule | 33 / 40 |
| BC initialization | 32 / 40 |
| BC teacher | 31 / 40 |
| Legacy shipped-policy fixture | 31 / 40 |
| Causal MPC `k=1` / `k=3` / `k=5` | 18 / 29 / 30 |
| Reactive heuristic | 17 / 40 |
| Privileged clairvoyant oracle | 37 / 40 |

The 37/40 oracle result is a privileged anytime achieved lower bound on this historical subset—not a submission baseline or a 200-case ceiling. The retired release's final record remains [legacy-final-40.json](evidence/legacy-final-40.json).

## Hurricane Maria evidence boundary

The [Hurricane Maria retrospective](../benchmarks/v4/hurricane-maria-retrospective.md) is a **project reconstruction from official records**. Its [machine-readable record](../internal/retrospectives/hurricane-maria-30d.json) separates source anchors and transformations from generated scenario inputs and simulated outputs.

Official-record anchors do not make simulated day-by-day services, shocks, allocations, preparedness, or recovery trajectories observed facts. The reconstruction is not a causal estimate, a historical counterfactual claim, a validation cohort, or part of development/final model selection.

## Evidence index

| Claim | Human-readable report | Machine record |
| --- | --- | --- |
| Shipped final result | [final-results-200.md](../benchmarks/v4/final-results-200.md) | [final-evaluation-200.success.json](../internal/evaluation_runs/v4/final-evaluation-200.success.json) |
| Final family pattern | [final-family-analysis-200.md](../benchmarks/v4/final-family-analysis-200.md) | Final success receipt plus matched oracle receipt |
| Development training sweep | [training-study-200.md](../benchmarks/v4/training-study-200.md) | [training-study-200-summary.json](../internal/developmental_runs/v4/training-study-200-summary.json) |
| Selected checkpoint | Publication report above | [checkpoint-selection-200.json](../internal/developmental_runs/v4/checkpoint-selection-200.json) |
| ONNX parity | Artifact manifest | [city_recovery_ppo.v4.parity.json](../internal/developmental_runs/v4/city_recovery_ppo.v4.parity.json) |
| Privileged CEM diagnostic | [clairvoyant-oracle-200.md](../benchmarks/v4/clairvoyant-oracle-200.md) | [development](../internal/developmental_runs/v4/clairvoyant-oracle-200-dev.json), [final](../internal/developmental_runs/v4/clairvoyant-oracle-200-final.json) |
| Cheap development planners | [development-baselines-200.md](../benchmarks/v4/development-baselines-200.md) | [development-baselines-200.json](../internal/developmental_runs/v4/development-baselines-200.json) |
| Hurricane Maria reconstruction | [hurricane-maria-retrospective.md](../benchmarks/v4/hurricane-maria-retrospective.md) | [hurricane-maria-30d.json](../internal/retrospectives/hurricane-maria-30d.json) |

The artifact manifest at `artifacts/city_recovery_ppo.v4.manifest.json` binds the selected checkpoint, embedded normalization, ONNX identity, interface, selection receipt, and parity receipt. The final claim/success lifecycle binds the owner's one-run authorization and exact frozen artifact; it is not a runtime authorization mechanism.

All synthetic scenario families and shock tapes are authored or generated locally. The repository does not bundle an empirical disaster dataset. Further learned-policy final reruns remain unauthorized.
