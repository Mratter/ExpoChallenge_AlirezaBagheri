# Project Brief

- Challenge: AI17
- Product: Civic Relay / Autonomous City Recovery Planner
- Audience: competition judges first; municipal resilience analysts as the design reference
- Primary job: author a bounded synthetic recovery scenario, compare two plans, inspect every daily allocation, and restore deterministic saved results
- Runtime: React/Vite compiled into a Python 3.12 FastAPI process
- Bind: `127.0.0.1:4117`
- Default metadata seed: `20260714`
- Runtime network: loopback only
- Production mocks, hidden fallback, and online services: forbidden

## Falsifiable Thesis

For a whole held-out authored scenario-family member and PCG64 seed, the frozen SB3 PPO policy exported to checksum-pinned ONNX can produce a different allocation trajectory than the visible OR-Tools GLOP planner while both consume the identical precomputed shock tape and satisfy the same lower, upper, budget, and sum constraints through the shared projector. The claim fails if identical inputs change canonical result bytes, PyTorch and ONNX actions exceed declared parity tolerances, either planner receives a different shock, any hard invariant is violated, or reported metrics cannot be regenerated from the preregistered protocol.

This is a synthetic simulator thesis, not a claim of real-world recovery effectiveness. All scenarios, coefficients, shocks, and policy training inputs are authored and non-empirical.

## Feature Complete Scope

- `CityRecoveryEnv` is a deterministic Gymnasium environment with five ordered resource classes, 23 observable current-state features, five allocation actions, seeded PCG64 shocks, and complete inspectable trajectories.
- The runtime candidate is a real Stable-Baselines3 PPO checkpoint exported to ONNX and executed locally with sequential, single-thread `CPUExecutionProvider` inference.
- The visible baseline is OR-Tools GLOP with a disclosed immediate priority/deficit/recovery objective. Neither planner sees future shocks.
- Both planners use the same daily context, bounds, budget, shock tape, state transition, and capped-simplex projector. Serialized measurements cover lower, upper, budget, and sum invariants.
- Every successful comparison is stored as canonical JSON under a content-derived result id. Repeating the same scenario is idempotent, and a new store/process can restore byte-identical content.
- A preregistered holdout uses five unseen authored families and eight disjoint seeds per family. It reports resilience, three recovery measures, deterministic repeats, paired uncertainty, and violations.
- Missing, corrupt, drifted, unparsable, or parity-inconsistent policy artifacts block every route except process liveness. No legacy heuristic or other fallback runs.

## Artifact Identity

- SB3 checkpoint: 80,181 bytes, SHA-256 `f270bc720e7d2866d293feab27692d3ac9542d064d275b13c33f4d960dad4e33`
- ONNX runtime policy: 10,469 bytes, SHA-256 `983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8`
- PyTorch/ONNX parity report: SHA-256 `20d87aafc638f3c6e7942a1578eea0710e0cd083c5a2054063f1813a76916a82`
- Accepted legacy linear candidate: unchanged 722 bytes, SHA-256 `23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3`; explicitly not PPO and not used for Feature Complete inference

## Gate Boundary

This repository contains a Feature Complete builder candidate. It does not self-approve the named gate. Independent project-specific tester and global judge verdicts remain required. Presentation and Release matrices are not part of this candidate turn.
