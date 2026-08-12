# Moderate family reweighting: 200-case development result

This is **development-only post-release evidence**. It does not replace the
shipped policy, alter the application, or authorize another final evaluation.
The study measured the shipped ONNX policy on the fixed 192-case TRAIN roster,
ranked the six training families by solve count, and assigned 2× episode weight
to the two weakest families throughout BC, DAgger, critic warm-up, and PPO.

The study is **complete—not promoted**; the shipped policy is retained.

## TRAIN-only choice of weights

| Training family | Shipped policy | Contextual tuned rule | Applied weight |
| --- | ---: | ---: | ---: |
| Grid failure | 27/32 | 25/32 | 2× |
| Displacement | 31/32 | 31/32 | 2× |
| Health surge | 32/32 | 31/32 | 1× |
| Supply chain | 32/32 | 32/32 | 1× |
| Transit nexus | 32/32 | 32/32 | 1× |
| Weather isolation | 32/32 | 29/32 | 1× |

The shipped policy solved **186/192** training cases and alone selected grid
failure plus displacement. The tuned rule solved **180/192** and would have
ranked grid failure plus weather isolation as its two weakest families; it is
reported only to disclose that disagreement and did not select the weights.
Development and final evidence were not used in this decision.

The deterministic sampler expands the 192 unique cases to a 256-occurrence
cycle. Consequently, the four-round BC/DAgger sequence contains **30,720**
action-labeled observations and freezes an observation RMS with count
`30720.0001`; the matched incumbent used **23,040** observations and RMS count
`23040.0001`. This is a 33.3% increase in imitation exposure and updates. The
treatment therefore combines family reweighting with extra imitation work; it
is not pure fixed-volume importance weighting. PPO remains 2M active
transitions and all three critics completed the same 50k warm-up as the matched
incumbent seeds.

## Development result

| Seed | 500k | 1M | 2M endpoint | Matched incumbent 2M | Endpoint delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 37017 | 171 | **176** | 175 | 172 | +3 |
| 47017 | 165 | 168 | 170 | 171 | -1 |
| 57017 | 166 | 166 | 172 | 171 | +1 |

The 2M endpoints were **175, 170, and 172 / 200**, for a mean of **172.333**,
population standard deviation **2.055**, and sample standard deviation
**2.517**. Against the same three incumbent seeds the mean gain was **+1.0
case**; against the incumbent five-seed mean of 171.4 it was **+0.933**. The
best of the nine selectable candidates was seed `37017` at 1M with **176/200**,
two cases below the shipped checkpoint's 178/200. Selected performance and 2M
endpoint performance are deliberately reported separately.

The full six-stage solve curves—BC, post-warm-up, 200k, 500k, 1M, and 2M—were:

| Seed | BC | Warm-up | 200k | 500k | 1M | 2M |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 37017 | 156 | 156 | 161 | 171 | 176 | 175 |
| 47017 | 151 | 151 | 162 | 165 | 168 | 170 |
| 57017 | 153 | 153 | 158 | 166 | 166 | 172 |

At the selectable milestones, the three-seed challenger means were **167.333,
170.000, and 172.333** at 500k, 1M, and 2M. The same-seed incumbent means were
**166.667, 170.000, and 171.333**, so the matched mean deltas were **+0.667,
0.000, and +1.000**. The same-seed endpoint comparison is the fairest
scientific contrast because it holds optimizer seeds and transition milestones
fixed. By comparison, 176 versus 178 is selection-asymmetric: it compares the
best of nine challenger candidates with the best of 20 incumbent candidates.
The preregistered conjunctive promotion gate remains the decisive decision rule.

## Matched family and case movement

The family table compares 2M endpoints with the same three incumbent seeds.
Each family has 40 cases per seed.

| DEV family | Challenger counts | Incumbent counts | Mean delta | Challenger-only / incumbent-only cases |
| --- | --- | --- | ---: | ---: |
| River flood | 33 / 31 / 32 | 32 / 31 / 31 | +0.667 | 4 / 2 |
| Industrial outage | 38 / 36 / 38 | 38 / 37 / 36 | +0.333 | 3 / 2 |
| Logistics strike | 40 / 40 / 40 | 40 / 40 / 40 | 0.000 | 0 / 0 |
| Seismic cluster | 29 / 28 / 27 | 26 / 28 / 28 | +0.667 | 6 / 4 |
| Health compound | 35 / 35 / 35 | 36 / 35 / 36 | -0.667 | 0 / 2 |

Across the pooled 600 matched endpoint rows, **504 were solved by both, 13 by
the challenger only, 10 by the incumbent only, and 73 by neither**. The small
net gain therefore includes failure redistribution: seismic gains are
seed-inconsistent and health loses two cases. There is no sharp collapse on an
easy family—logistics remains 40/40 for every seed—but there is also no robust
targeted weak-family improvement.

The best registered challenger and shipped selected checkpoint similarly do
not dominate one another: **175 both, 1 challenger-only, 3 shipped-only, and 21
neither**. The challenger's one unique solve is health compound; the shipped
checkpoint's three are one river-flood and two seismic-cluster cases.

## Preregistered gate

| Conjunctive condition | Observed | Required | Passed |
| --- | ---: | ---: | ---: |
| Selected checkpoint | 176/200 | At least 183/200 | **No** |
| Three-seed 2M mean | 172.333/200 | Strictly above 171.4/200 | Yes |
| Endpoints at or above 172 | 2/3 | At least 2/3 | Yes |

All conditions were required, so the failed best-checkpoint threshold is
decisive. Resilience AUC was not used for selection or promotion. Every TRAIN
and DEV row has zero hard violations and exactly `0.0` maximum conservation
residual.

## Interpretation and limits

This is a narrow non-promotion result for the shipped-policy-ranked 2:1 sampler,
the adopted optimizer, these three seeds, and the 2M budget. The sampler
duplicates existing support; it adds no family or scenario, and the extra
imitation exposure prevents attributing any difference to reweighting alone.
Other weight ratios, a fixed-volume implementation, the tuned-rule-ranked
alternative, longer budgets, and new hard-family support remain untested.
Per-family changes are descriptive matched evidence, not causal effects.

No final case was constructed or evaluated for this experiment. The shipped
artifact and its one-time final receipt remain unchanged.

## Evidence

- [Portable machine receipt](../../internal/developmental_runs/v4/moderate-family-study-200.json) SHA-256: `8cfd77617f0fb5ddb11c035c93dac00af99495681c11ec83620463e3dbc5979a`
- External difficulty receipt SHA-256: `27d4b675273ebdfabc7ec5f6546a2d4c75ec5774e024c9d0c57484f800e4e5d4`
- External protocol SHA-256: `4cc902fdee9e090df0be6042ccb5f2953eadde9693867e553f26b61ca8c65ad7`
- External study summary SHA-256: `935a0069d3c1eb53885e4ff5843ec5545eef4277a4a73ff6376a9948ea64e8a0`
- The machine receipt retains both 192-row TRAIN difficulty evaluations, all
  nine selectable 200-row DEV candidates, six-stage curves, matched incumbent
  endpoint rows, finite-prefix sampler evidence, three trainer receipts, and
  all 12 durable checkpoint-bundle identities (including diagnostic 200k).
