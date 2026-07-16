# Data Audit

## Data Decision

Feature Complete uses no external dataset. Every state, priority, shock, dependency, transition coefficient, scenario-family center, policy training episode, and evaluation unit is authored synthetic data under CC0-1.0 for this repository. API/UI metadata permanently exposes `empirical: false` / `Synthetic model`. Results are not forecasts of real cities.

## Deterministic Sources

| Source | Revision / provenance | Runtime availability |
|---|---|---|
| Dynamics, constraints, shock catalogue | `synthetic-city-dynamics-v2` in `backend/app/simulator.py` | Tracked source, offline |
| Authored scenario families | Four training and five held-out families in `backend/app/scenarios.py` | Tracked source, offline |
| Random generation | NumPy 2.3.2 `Generator(PCG64)`; complete tape generated before either planner | Locked in `uv.lock`, offline |
| Learned candidate | Stable-Baselines3 2.7.0 PPO, 30,000 CPU steps, training seed `17017` | Tracked SB3 checkpoint and ONNX export, offline |
| Baseline | OR-Tools 9.14.6206 GLOP, single-thread solve | Locked in `uv.lock`, offline |
| Persistence | Canonical JSON keyed by SHA-256 of schema/seed/scenario/policy/baseline | `%LOCALAPPDATA%\Innoverse\ai17-city-recovery`, offline |

## Split And Leakage Controls

- Split unit: complete authored scenario-family member plus its seed, never an individual day.
- Training families: `train_transit_cascade`, `train_displacement`, `train_supply_interrupt`, `train_health_surge`.
- Training scenario seeds: `170100..170107` (32 complete units).
- Parity observations: training-family members at seeds `180100..180103`; parity is interface validation, not outcome evaluation.
- Held-out families: `holdout_coastal_weather`, `holdout_blackout`, `holdout_food_access`, `holdout_aftershock`, `holdout_public_health`.
- Held-out seeds: `271700..271707` (40 complete units), explicitly excluded from training.
- The protocol is `evaluation/protocol.v1.json`, SHA-256 `b36bba8dba6948b6b2a29170f6e5a9f7ebf012f95ce859edcece87bb5c9c5655`. The fixed artifact is loaded before evaluation and cannot adapt from evaluation requests.
- Candidate and baseline receive one identical precomputed shock object per day. Neither observes a future shock when proposing today's allocation.

## Artifact Provenance

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Accepted legacy linear JSON, non-PPO | 722 | `23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3` |
| SB3 PPO checkpoint | 80,181 | `f270bc720e7d2866d293feab27692d3ac9542d064d275b13c33f4d960dad4e33` |
| ONNX runtime policy | 10,469 | `983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8` |
| Policy metadata | 2,530 | `becc2eed1e552e9a503c3210d2ebae18eeccc593c9a7d716fae11e1e69b1c62e` |
| PyTorch/ONNX parity evidence | 631 | `20d87aafc638f3c6e7942a1578eea0710e0cd083c5a2054063f1813a76916a82` |

The manifest verifies exact project/schema, artifact set, role, path, source, license, bytes, and SHA-256. Metadata cross-links the SB3, ONNX, parity, and legacy hashes. Runtime also parses/checks ONNX and performs a real smoke action. Any inconsistency blocks readiness and all product routes.

## Authored Bounds

- Horizon 7-30 days; budget 50-500 units/day
- Five initial services `.05-.95`; five priorities `.5-2.0`
- Daily shock probability `0-.35`
- Severity minimum `.05-.25`; maximum `.10-.40`; minimum strictly below maximum
- Optional forced shock day inside the horizon and type in the five-shock catalogue

Pydantic rejects unknown fields, wrong vector lengths, invalid bounds, and inconsistent severity or forced-day values.

## Limitations

The 40-case holdout is a larger deterministic synthetic protocol, not evidence of population generalization. It establishes computation, artifact provenance, split discipline, constraint enforcement, repeatability, and measured comparison only. It does not establish causal benefit, calibration, fairness, cost effectiveness, or municipal suitability.
