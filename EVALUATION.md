# Evaluation

## Pre-Registered Feature Complete Protocol

- Protocol: `evaluation/protocol.v1.json`, SHA-256 `b36bba8dba6948b6b2a29170f6e5a9f7ebf012f95ce859edcece87bb5c9c5655`
- Fixed candidate: `city-recovery-sb3-ppo-v1`, checksum-pinned ONNX CPU inference
- Visible baseline: OR-Tools GLOP `ortools-glop-visible-v1`, current-day objective only
- Split unit: complete authored scenario-family member plus seed
- Holdout: five families x eight disjoint seeds = 40 complete cases
- Determinism: three canonical full-result executions per case, 120 executions total
- Primary metric: priority-weighted daily resilience AUC
- Recovery metrics: post-largest-shock recovery shortfall AUC, days to pre-shock recovery, and critical service-days
- Uncertainty: paired deterministic nonparametric bootstrap over complete scenario-seed units, seed `1717`, 5,000 samples
- Hard invariant: zero lower, upper, budget, and sum violations for both planners
- Outcome rule: report measured resilience improvement when mean candidate AUC is higher; otherwise report the measured resilience/recovery trade-off without changing the baseline or holdout

## Policy Export Parity

`evaluation/policy_parity.v1.json` contains 32 current-state observations from training-family scenarios at separate parity seeds.

| Measurement | Observed | Tolerance |
|---|---:|---:|
| Maximum PyTorch/ONNX action absolute error | `1.7881393432617188e-07` | `1e-05` |
| Maximum pre-projector proposal absolute error | `7.856886107049377e-06` | reported |
| Maximum post-projector allocation absolute error | `7.850000002918023e-06` | `1e-04` |

Parity passed. ONNX SHA-256 is `983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8`; SB3 checkpoint SHA-256 is `f270bc720e7d2866d293feab27692d3ac9542d064d275b13c33f4d960dad4e33`.

## Held-Out Results

Report: `evaluation/feature_complete_report.v1.json`, SHA-256 `fea00d1bf578c7d52cad816eed732a58ffb3f9b809c2788ba35c601e976f9351`.

| Measurement | SB3 PPO / ONNX | OR-Tools GLOP | Candidate minus baseline |
|---|---:|---:|---:|
| Weighted daily resilience AUC | `0.49148043` | `0.44418455` | `+0.04729588` |
| Post-shock recovery shortfall AUC, lower is better | `0.01455615` | `0.01476437` | `-0.00020822` |
| Days to pre-shock recovery, lower is better | `4.35` | `5.375` | `-1.025` |
| Critical service-days, lower is better | `3.35` | `5.25` | `-1.9` |

- Paired bootstrap 95% interval for the resilience AUC delta: `[0.04273770, 0.05176164]`.
- Candidate resilience AUC higher: 40 cases; baseline higher: 0; ties: 0.
- Exact-repeat determinism mismatches: 0/40.
- Candidate lower/upper/budget/sum violations: `0/0/0/0`.
- Baseline lower/upper/budget/sum violations: `0/0/0/0`.

The fixed candidate shows a measured improvement over this visible baseline on this authored synthetic protocol. The result is reported as measured simulator behavior, not statistical evidence about cities. The bootstrap interval describes only the finite authored protocol and is not causal or population uncertainty.

## Runtime Fixture

The production verifier submits an 11-day unseen authored scenario with seed `118773` five times. It requires byte-identical responses, checks 110 service-level lower/upper constraints plus daily sum/budget evidence, restores the content-addressed result byte-identically through the persistence API, confirms the deterministic index, and rejects an invalid 31-day scenario. Exact closing metrics are recorded by the final `scripts/verify.ps1` run.

## Failure Evidence Contract

- Pydantic-invalid scenarios return HTTP 422 `INVALID_SCENARIO` with field details.
- Missing, unreadable, byte-drifted, hash-drifted, schema-drifted, parity-inconsistent, or unparsable policy artifacts leave `/health/live` at 200 but make every other route return structured HTTP 503 `DEPENDENCY_NOT_READY`.
- Corrupt or identity-mismatched saved results return explicit `PERSISTENCE_FAILED`; they are not skipped or recomputed silently.
- The UI clears prior candidate/baseline evidence before showing invalid, dependency, persistence, computation, or network failure.
- Full preflight cross-checks artifact/model/dataset metadata, a real ONNX action, full default trajectories, persisted restore, and the evaluation report/model/protocol hashes.

## Limitations

All inputs and dynamics are synthetic and non-empirical. The evaluation does not establish real-world generalization, causal effectiveness, equity, safety, calibration, cost effectiveness, or fitness for municipal decisions. Presentation and Release evidence are outside this builder candidate.
