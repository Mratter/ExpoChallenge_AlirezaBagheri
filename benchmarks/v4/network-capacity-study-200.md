# Network-capacity study: 200-case development result

This is **development-only post-release evidence**. It does not replace the
shipped policy, modify the application, or authorize another final evaluation.
The preregistered challenger enlarged both the actor and critic hidden layers
from `[384, 256, 128]` to `[768, 512, 256]`, kept the public `73 -> 22`
interface unchanged, and tested learning rates `7.5e-5` and `3e-5` on the same
three policy seeds. Each run used the adopted behavior-cloning warm start,
actor-frozen critic warm-up, frozen observation RMS, and 2M active PPO
transitions.

The study is **complete—not promoted**.

## Registered 2M endpoints

| Architecture arm | LR | Seed 37017 | Seed 47017 | Seed 57017 | Mean / 200 | Population SD | Sample SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Large | `7.5e-5` | 171 | 169 | 172 | 170.667 | 1.247 | 1.528 |
| Large | `3e-5` | 178 | 176 | 175 | **176.333** | **1.247** | **1.528** |
| Incumbent | `7.5e-5` | 172, 171, 171, 174, 169 across five seeds | — | — | 171.4 | 1.625 | 1.817 |

The fair seed-level comparison is the selected large-arm mean of **176.333 / 200**, or **+4.933 cases** over the incumbent five-seed 2M mean of 171.4. The best of the 18 registered selectable checkpoints was the large `3e-5` seed `37017` endpoint at **178 / 200**. It tied, rather than exceeded, the shipped checkpoint selected from 20 incumbent candidates.

The comparison is paired within the challenger: for each policy seed, the two
learning-rate arms began with byte-identical BC actor, policy, dataset, and
frozen observation RMS identities, and the registered configuration differed
only in learning rate. The matched 2M deltas for `3e-5` versus the paired large
`7.5e-5` arm were **+7, +7, and +3** cases for seeds `37017`, `47017`, and
`57017`. Against the same-seed incumbent endpoints of 172, 171, and 171, the
large `3e-5` endpoints improved by exactly **+6, +5, and +4** cases. Those
comparisons are distinct from the +4.933 difference between the challenger
three-seed and incumbent five-seed means.

## Development curves

| Arm / seed | BC | After critic warm-up | 200k | 500k | 1M | 2M | 1M → 2M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Large `7.5e-5` / 37017 | 152 | 152 | 160 | 166 | 175 | 171 | -4 |
| Large `7.5e-5` / 47017 | 153 | 153 | 161 | 167 | 171 | 169 | -2 |
| Large `7.5e-5` / 57017 | 155 | 155 | 161 | 170 | 171 | 172 | +1 |
| Large `3e-5` / 37017 | 152 | 152 | 154 | 164 | 173 | 178 | +5 |
| Large `3e-5` / 47017 | 153 | 153 | 157 | 164 | 173 | 176 | +3 |
| Large `3e-5` / 57017 | 155 | 155 | 157 | 170 | 174 | 175 | +1 |
| Incumbent five-seed mean | — | — | — | 168.2 | 171.4 | 171.4 | 0.0 |

The low-learning-rate large arm was still climbing at the registered boundary:
all three seeds improved from 1M to 2M by **+5, +3, and +1** cases, and its mean
rose from 173.333 to 176.333. The large `7.5e-5` mean fell by 1.667 cases over
the same interval, while the incumbent five-seed mean was flat at 171.4 from
1M to 2M. This is an interesting positive signal for the large-plus-low-LR
combination, not evidence that the larger network had converged by 2M.

## Per-family 2M endpoints

Each cell lists the three registered seed counts and their mean; each family
contains 40 development cases.

| Family | Large `7.5e-5` | Large `3e-5` |
| --- | ---: | ---: |
| River flood | 30 / 31 / 31 (30.667) | 34 / 33 / 32 (33.000) |
| Industrial outage | 36 / 35 / 36 (35.667) | 37 / 37 / 37 (37.000) |
| Logistics strike | 40 / 40 / 40 (40.000) | 40 / 40 / 40 (40.000) |
| Seismic cluster | 30 / 27 / 30 (29.000) | 31 / 30 / 31 (30.667) |
| Health compound | 35 / 36 / 35 (35.333) | 36 / 36 / 35 (35.667) |

## Architecture and gate

The large policy has **1,169,709** parameters: a 587,542-parameter deterministic
actor mean path, 22 learned log-standard-deviation parameters, and a
582,145-parameter critic. Thus the actor total is **587,564**. The incumbent
policy has 162,710 deterministic actor-mean parameters plus the same 22 learned
log-standard-deviation parameters, a 160,001-parameter critic, and **322,733**
parameters total. The optional smaller control arm was not run.

| Preregistered conjunctive condition | Observed | Required | Passed |
| --- | ---: | ---: | ---: |
| Selected checkpoint | 178/200 | At least 183/200 | **No** |
| Selected arm, three-seed 2M mean | 176.333/200 | Strictly above 171.4/200 | Yes |
| Selected-arm endpoints at or above 172 | 3/3 | At least 2/3 | Yes |

All three conditions were required, so the failed best-checkpoint threshold is
decisive. Resilience AUC was not a selection or promotion metric. All recorded
development rows have zero hard violations and exactly `0.0` maximum
conservation residual.

## Interpretation and limits

The result does not isolate capacity alone because learning rate and capacity
interact: the study has no incumbent-size `3e-5` arm. It therefore supports a
positive result for the registered large-plus-low-LR combination, alongside a
formal non-promotion decision—not a generic claim that capacity is irrelevant.
The non-promotion scope is exactly `[768, 512, 256]`, these two learning rates,
these three seeds, and the 2M-transition budget. The rising `3e-5` curves make
longer-budget behavior explicitly unresolved.

No final case was constructed or evaluated for this experiment. The shipped
artifact and its one-time final receipt remain unchanged.

## Evidence

- [Portable machine receipt](../../internal/developmental_runs/v4/network-capacity-study-200.json) SHA-256: `fd27e39b3b4868e43231b91f879e1830f1b2380f37bd03c3b23b9e5510564304`
- External protocol SHA-256: `b80b9d99e629109b88e92d8dc9d1d0ec1c754ce70b2477225f61d97e9496a49a`
- External summary SHA-256: `86e336a1cdd8c7584f7796030e121a8dbf110af8caa035c8fc65cc38778d3ddb`
- The receipt retains exact portable rows for all 18 selectable checkpoints,
  the six complete diagnostic curves, all six training-receipt hashes, all 18
  bundle identities, per-family aggregates, source hashes, and gate arithmetic.
