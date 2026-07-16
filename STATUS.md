# Status

- Phase: Feature Complete builder candidate verification
- Current gate: Builder verification complete; ready for independent AI17 Feature Complete tester review on the clean candidate HEAD
- Independent Feature Complete verdict: Not recorded
- Accepted earlier state: Evidence and Vertical Slice ledger anchor `db679c895aebb42027a4c6f0590b466dd0657e9b` remains closed; clean activation baseline was `eb9c1dfa8ab52c03a2ebf97f31a43ab28849715c`
- Runtime: CPU, Python 3.12, one loopback process at `127.0.0.1:4117`
- Default metadata seed: `20260714`
- Policy: real Stable-Baselines3 PPO checkpoint with checksum-pinned ONNX CPU inference
- Legacy policy: accepted linear candidate preserved byte-for-byte and permanently disclosed as non-PPO
- Data: authored synthetic and non-empirical only
- Last updated: 2026-07-16

## Implemented

- Deterministic Gymnasium environment with five resource classes, bounded authored scenarios, PCG64 shock tapes, common dynamics/projector, and full daily observations/actions/proposals/bounds/allocations/transitions.
- Visible OR-Tools 9.14.6206 GLOP baseline using the same shocks and hard constraints without future-shock access.
- SB3 2.7.0 PPO trained for 30,000 CPU steps over four training families and 32 complete training scenario-seed units; exported through PyTorch 2.8.0 to ONNX opset 17.
- 32-case PyTorch/ONNX parity: maximum action error `1.7881393432617188e-07`; maximum pre-projector proposal error `7.856886107049377e-06`; maximum post-projector allocation error `7.850000002918023e-06`.
- Preregistered evaluation over five held-out families x eight seeds x three canonical repeats. Candidate resilience AUC `0.49148043`; baseline `0.44418455`; delta `+0.04729588`; paired bootstrap 95% interval `[0.04273770, 0.05176164]`.
- Held-out recovery deltas also favor the candidate on average: post-shock shortfall `-0.00020822`, recovery days `-1.025`, and critical service-days `-1.9`. Candidate resilience was higher on 40/40 units. These are synthetic protocol measurements, not real-world claims.
- Determinism mismatches `0`; candidate/baseline lower, upper, budget, and sum violations all `0` across the held-out evaluation.
- Canonical local persistence keyed by scenario/seed/policy/baseline identity, deterministic index ordering, idempotent repeat saves, byte-identical restore, and explicit corrupt-result errors.
- Missing/corrupt/drifted policy files, invalid ONNX, parity inconsistency, legacy relabeling, and metadata drift block the product without fallback.
- UI labels the actual SB3/ONNX candidate and OR-Tools baseline, exposes measured recovery and constraint evidence, restores saved authored results, and retains the synthetic and legacy non-PPO disclosures.

## Builder Verification

- Dependency lock and CPU environment sync: PASS.
- PPO training/export and parity build: PASS.
- `scripts/evaluate_policy.py`: PASS, 40 held-out cases, 120 exact result executions, zero mismatches/violations.
- `scripts/preflight.ps1 -Profile cpu -Full`: PASS; 28 backend tests, Ruff, 6 frontend tests, strict TypeScript, and Vite production build.
- `scripts/verify.ps1 -Profile cpu`: PASS; five byte-identical unseen responses, 110 service-bound checks, byte-identical persisted restore after a full process stop/start, invalid-input rejection, and clean server shutdown.
- Live unseen seed `118773`: candidate resilience AUC `0.46665569`; OR-Tools `0.44759060`; response SHA-256 `07393b9265e2386c606be7c373f907842ad9fda589f2ce44c3e59ae1b044c741`; result id `ce270cd9d63a8b30c9f0b4a7adb75b0f6f649cc77f6997fb0f6553bd008b45df`.

## Artifact Hashes

- ONNX `983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8`
- SB3 `f270bc720e7d2866d293feab27692d3ac9542d064d275b13c33f4d960dad4e33`
- Metadata `becc2eed1e552e9a503c3210d2ebae18eeccc593c9a7d716fae11e1e69b1c62e`
- Parity `20d87aafc638f3c6e7942a1578eea0710e0cd083c5a2054063f1813a76916a82`
- Evaluation protocol `b36bba8dba6948b6b2a29170f6e5a9f7ebf012f95ce859edcece87bb5c9c5655`
- Evaluation report `fea00d1bf578c7d52cad816eed732a58ffb3f9b809c2788ba35c601e976f9351`
- Legacy linear candidate `23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3`

## Remaining At This Gate

- Route the exact clean candidate SHA to the independent AI17 Feature Complete tester. This builder does not grant the gate verdict.

Presentation, clean-machine/empty-cache setup, outbound-firewall enforcement, long-path clone, and the complete Release failure/process matrix were not run here.
