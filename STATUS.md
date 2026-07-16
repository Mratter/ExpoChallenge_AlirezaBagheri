# Status

- Phase: Independent graphic-designer pass complete
- Current gate: Designer candidate verified; ready for a separate independent Presentation tester on the exact clean pushed SHA
- Independent Feature Complete verdict: PASS for frozen candidate `3c16f0359cca93e494cc65f0a8850ef6e9c744da`, recorded by `9a7618469f5c5050e98732c00adcfe2059c1dadc`
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

## Independent Graphic-Designer Verification

- PASS with zero blocking visual or accessibility issues at `1440x900`, `1280x720`, and `390x844`; no document-level horizontal overflow and no action-dock/control overlap at either scroll boundary.
- Recast the accepted recovery desk as a shared-shock evidence folio: a paper run brief, explicit shared/forced shock marks, a selected-day link, and paired candidate/baseline service tracks in the daily ledger. The authored scenario editor remains on the left and computed evidence remains on the right.
- Preserved the accepted palette and typography, used no new raster/vector assets, and retained the real SB3 PPO / ONNX, visible OR-Tools GLOP, synthetic/non-empirical, and legacy non-PPO disclosures without changing any measured claim.
- Keyboard and assistive-state checks passed: high-contrast focus, roving tabs with linked panels, semantic ledger headers/cells, full mobile row names, focus-preserving busy actions, alert autofocus/scroll, deterministic Review routing, reduced motion, reachable empty state, and explicit stale-evidence status.
- A deliberately delayed mobile request exposes `Running` in the first-fold topbar; settled valid state returns to `Local`. The normal bounded comparison still exposes its complete in-panel status.
- Fresh valid browser flow had zero console errors/warnings and loopback-only `200` requests. Deliberate invalid scenarios produced only the expected HTTP `422`, focused the compact alert, and routed Review to the exact Days or Severity min control.
- Verification on the final source: normal `scripts/preflight.ps1 -Profile cpu` PASS; backend `28 passed`; Ruff PASS; frontend `11 passed`; strict TypeScript PASS; Vite production build PASS with 1,775 transformed modules.
- Model, training, evaluation, artifact, dataset, backend, and accepted evidence files were not changed or regenerated during this pass.

## Artifact Hashes

- ONNX `983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8`
- SB3 `f270bc720e7d2866d293feab27692d3ac9542d064d275b13c33f4d960dad4e33`
- Metadata `becc2eed1e552e9a503c3210d2ebae18eeccc593c9a7d716fae11e1e69b1c62e`
- Parity `20d87aafc638f3c6e7942a1578eea0710e0cd083c5a2054063f1813a76916a82`
- Evaluation protocol `b36bba8dba6948b6b2a29170f6e5a9f7ebf012f95ce859edcece87bb5c9c5655`
- Evaluation report `fea00d1bf578c7d52cad816eed732a58ffb3f9b809c2788ba35c601e976f9351`
- Legacy linear candidate `23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3`

## Remaining At This Gate

- Commit and push the designer candidate, then route that exact clean SHA to the independent Presentation tester for the full viewport/state/accessibility/demo matrix.
- Release verification may begin only after an independent Presentation PASS. This designer pass grants neither Presentation nor Release.

No retraining or evaluation rerun was performed. Independent Presentation and Release verdicts, clean-machine/empty-cache setup, outbound-firewall enforcement, long-path clone, and the complete Release failure/process matrix remain unclaimed.
