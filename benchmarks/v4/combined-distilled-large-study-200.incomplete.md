# Combined large-network + oracle-distillation attempt — incomplete

This development-only attempt is **not a completed three-seed study and not a promotion candidate**. It stopped at the preregistered actor-frozen critic warm-up verification gate for seed `57017`; it was not retried or resumed, and no final case was imported or evaluated.

## What completed

| Seed | BC init | Post-warm-up | 200k | 500k | 1M | 2M | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 37017 | 153/200 | 153/200 | 158/200 | 162/200 | 173/200 | 170/200 | complete |
| 47017 | 153/200 | 153/200 | 160/200 | 167/200 | 170/200 | 174/200 | complete |

Only these two registered curves completed. The best observed registered selectable checkpoint was seed `47017` at 2M with `174/200`; because the attempt is incomplete, that observation is not a valid study summary or promotion candidate. Their 2M paired deltas were `-2` and `+3` versus the same-seed incumbent endpoints, and `-8` and `-2` versus the same-seed large-network-only endpoints. Because seed `57017` did not reach PPO, this report does **not** calculate a two-seed substitute for the preregistered three-seed mean, standard deviation, or promotion decision.

### Completed 2M endpoints by development family

| Seed | River flood | Industrial outage | Logistics strike | Seismic cluster | Health compound |
|---:|---:|---:|---:|---:|---:|
| 37017 | 33/40 | 35/40 | 40/40 | 27/40 | 35/40 |
| 47017 | 34/40 | 36/40 | 40/40 | 28/40 | 36/40 |

Across all 14 retained development evaluations, hard violations were `0` and maximum conservation residual was exactly `0.0`.

## Why the attempt stopped

Seed `57017` completed exactly 50,000 fixed critic-warm-up transitions. The gate requires the final warm-up rollout explained variance to be strictly above `0.5`; the final value was `0.47894805669784546`, so the gate failed. Intermediate iterations above `0.5` do not override the registered final-rollout check. Active PPO transitions remained `0`, the actor stayed byte-identical, the frozen observation RMS stayed unchanged, and its retained BC/post-warm-up development result was `153/200`.

This was an intentional verification-gate termination, not a process crash: the trainer returns code `3` when `training_complete` is false, and the orchestrator treats any nonzero trainer return code as a stop. The console therefore records the orchestrator's failure message after preserving the incomplete worker receipt.

## Large offline fit

The 768/512/256 actor used one offline BC stage over a fixed dataset: 5,040 fit observations, a 720-observation trajectory-level holdout, 15 epochs, zero DAgger or interactive relabeling, and the frozen source observation RMS. Only the privileged teacher labels saw future tape; the student consumed 73 causal public inputs and produced 22 actions.

| Target | Fit MSE / MAE | Held-out MSE / MAE | Held-out relative MSE improvement |
|---|---:|---:|---:|
| Oracle labels | 0.0382702537 / 0.1383763850 | 0.0410502814 / 0.1448733062 | 0.8704107290 |
| Matched hand-rule control | 0.0161156859 / 0.0766169503 | 0.0217285659 / 0.0857965201 | 0.9525681396 |

## Interpretation boundary

The treatment is nonfactorial: relative to large-network-only evidence, it changes initialization from preparedness-teacher BC plus four DAgger rounds to single-pass offline oracle BC, changes per-seed observation normalization to one shared frozen RMS, and differs in seed `57017` warm-up budget (60k in the historical large-only run versus 50k here). The partial curves cannot isolate a causal distillation effect. The stopped attempt was not retried, produced no three-seed summary or promotion result, used no final case, and did not alter the shipped artifact.

## Evidence

- Portable incomplete receipt SHA-256: `027069bc3ac5ecabaa0062a50ed3cc0f9d530d8ca80812cb0a890873809cd822`
- External 37-file root inventory SHA-256: `f3bcabc8e8ad84e77444a7dc2d34013c7615870a42586cd3f470028f268723c0`
- Base protocol SHA-256: `0709892d67a75cff6f46c4f46e7aa6b53c8f0e2155e18b8ae913ac14f851e9b9`
- PPO protocol SHA-256: `d6aa57f538e00054ad4077bf5fe5b98b60ff99c3bd7efde75273482a3726dc75`
- Console stdout SHA-256: `2ed71bb29c0338d9bf1c4dcf94ca45fd093421824b140c344b228a702651e125`
- Console stderr SHA-256: `f10910603d181dbefc89c2e2289be07b18576b7c45dd9e527aa9c747419ef6fe`
