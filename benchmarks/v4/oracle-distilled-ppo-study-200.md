# Oracle-distilled PPO: 200-case development result

This is **development-only post-release evidence**. It does not replace the shipped policy, modify the application, or authorize another final evaluation. The experiment continued one fixed offline actor—trained once from privileged CEM trajectories on the 192-case training split—with fresh seeded critics and the adopted PPO configuration. The actor still consumed only the public 73-field observation at runtime.

## Result

| Policy seed | BC initialization | After critic warm-up | PPO 200k | PPO 500k | PPO 1M | PPO 2M |
|---:|---:|---:|---:|---:|---:|---:|
| 37017 | 157/200 | 157/200 | 169/200 | 168/200 | 173/200 | **178/200** |
| 47017 | 157/200 | 157/200 | 171/200 | 169/200 | 173/200 | **174/200** |
| 57017 | 157/200 | 157/200 | 170/200 | 171/200 | 171/200 | **170/200** |

The three 2M endpoints were **178, 174, and 170**, for a mean of **174.0 / 200**, population standard deviation **3.266**, and sample standard deviation **4.0**. The fair seed-level comparison is therefore **+2.6 cases** over the incumbent five-seed 2M mean of 171.4. The best of the nine registered selectable checkpoints was seed `37017` at 2M with **178 / 200**—a tie with, not an improvement over, the shipped checkpoint's 178 / 200 development result.

Every retained development row has zero hard violations and exactly `0.0` conservation residual.

## Family pattern at the 2M endpoints

| Development family | Seed 37017 | Seed 47017 | Seed 57017 | Three-seed mean |
|---|---:|---:|---:|---:|
| River flood | 34/40 | 34/40 | 33/40 | 33.67/40 |
| Industrial outage | 39/40 | 36/40 | 38/40 | 37.67/40 |
| Logistics strike | 40/40 | 40/40 | 40/40 | 40.00/40 |
| Seismic cluster | 29/40 | 29/40 | 23/40 | 27.00/40 |
| Health compound | 36/40 | 35/40 | 36/40 | 35.67/40 |

These are descriptive counts on the shared development roster. They expose where optimizer-seed dispersion appears, but they are not a causal family ablation.

## Preregistered promotion decision

Promotion required all three conditions:

| Condition | Observed | Required | Passed |
|---|---:|---:|:---:|
| Best registered checkpoint | 178/200 | At least 183/200 | No |
| Three-seed 2M mean | 174.0/200 | Strictly above 171.4/200 | Yes |
| 2M endpoints at or above 172 | 2 of 3 | At least 2 of 3 | Yes |

The conjunctive gate therefore returned **complete—not promoted**. The shipped ONNX artifact, manifest, application wiring, and retained final result remain unchanged.

## Scientific boundary

The upstream teacher was privileged: its fixed-budget CEM search saw each training case's full future shock tape. The student did not. It learned a public `73 → 22` actor from 5,040 fit observations, with 24 complete trajectories—720 action-labeled observations—held out from fitting; the approved BC actor solved 157/200 development cases before PPO. On that holdout, its oracle-action MSE/MAE were **0.0426693 / 0.148059**, while the separately matched hand-rule control's MSE/MAE were **0.0247503 / 0.0949414**. Each PPO run imported that byte-identical actor and frozen observation RMS, built a fresh critic from its registered policy seed, froze the actor for 50,000 critic-warm-up transitions, and then ran 2M active actor-critic transitions.

This was **single-pass offline behavior cloning with zero DAgger iterations**. The operative receipt fields record `dagger_iterations: 0`, `interactive_relabelling: false`, and no legacy demonstration collection inside the PPO workers. A generic legacy trainer-flow label still names “behavior cloning and DAgger”; it is nonoperative metadata and does not describe this treatment.

The null result is narrow: it applies to this single-pass oracle-BC actor, the adopted optimizer, and the 2M-transition budget. It does not resolve offline-policy distribution shift, and it does not show that privileged future information is the only possible advantage.

No final case was constructed or evaluated for this experiment.

## Evidence

- [Portable machine receipt](../../internal/developmental_runs/v4/oracle-distilled-ppo-study-200.json) SHA-256: `aee2df40263f892fb8d979ae190a483a91711564169bbac45336f32a24bb5e0d`
- External attempt-02 protocol SHA-256: `8bf4d9de7d734911d88b5a2ef90db437a56653e9e830320744d1d68e56e08596`
- External attempt-02 summary SHA-256: `e85eea9b60963ef655c640f6cb089f7f7832cd1b8ff4a9aa24195aca015dded4`
- Upstream training-oracle receipt SHA-256: `e7777e53f20b886bbb82b167e0303b20ee0de32dcf9b87f50d175a0b71c5dc89`
- Upstream oracle-BC student receipt SHA-256: `76025a6376db6905b1d96d08122a14bccc7639040921768a79e4c83debabec84`

The portable receipt contains all 1,800 per-case rows for the nine selectable checkpoints, the complete diagnostic curves, per-family aggregates, bundle hashes, upstream identities, exact ranking, and recomputed promotion arithmetic.
