# AI17 Agent Context

Last refreshed: 2026-07-17. This file is the standalone resumption authority after agent-document cleanup. Live HEAD contains runtime/release-readiness hardening newer than the archived gate wording.

## Identity and goal

- Challenge AI17, **Civic Relay / Autonomous City Recovery Planner**.
- Goal: compare a frozen Stable-Baselines3 PPO policy exported to checksum-pinned ONNX CPU inference with a visible OR-Tools GLOP planner under identical authored shocks and hard constraints.
- All scenarios, dynamics, training inputs, and evaluation units are synthetic and non-empirical. This is local simulation evidence, not a real-city forecast or operational recommendation.

## Live repository state

- Root: `AI challanges/AI Autonomous City Recovery Planner (17)`; `main` aligned with `origin/main` and clean before this file was added.
- HEAD: `cece793` (`Harden runtime environment for long Windows paths`, 2026-07-17), after `4dfdb84` (isolate closed training toolchain) and `4ef5215` (cold dependency setup).
- This file is the only intended new worktree file from this task until committed.
- The archived status snapshot stops at the independent graphic-designer pass. Live HEAD has three later setup/runtime verification hardening commits, but no later accepted independent Presentation or Release PASS is captured in this lineage. Do not infer a gate PASS from hardening commits alone.

## Stack, architecture, and entrypoints

- Python 3.12, FastAPI/Uvicorn, Gymnasium, NumPy, ONNX Runtime, OR-Tools; training-only Stable-Baselines3/PyTorch group; React 19 + TypeScript + Vite.
- Backend entry: `backend/app/main.py`; simulator, scenarios, artifact guard, models, and persistence are in `backend/app/`.
- UI entry: `frontend/src/main.tsx`.
- `scripts/run.ps1` serves API and built UI from one loopback process at `http://127.0.0.1:4117`.
- `scripts/project_environment.ps1` chooses repository `.venv` for normal paths or a short root-hashed `%LOCALAPPDATA%\Innoverse\ai17-city-recovery\environments\...` environment when ONNX native paths would approach the Windows loader limit.

```powershell
.\scripts\setup.ps1 -Profile cpu
.\scripts\preflight.ps1 -Profile cpu -Full
.\scripts\run.ps1 -Profile cpu
.\scripts\verify.ps1 -Profile cpu
```

The normal setup/preflight/run/verify scripts select the loader-safe environment automatically. Direct training/evaluation commands require the same printed `UV_PROJECT_ENVIRONMENT` in long clones.

## Critical data, models, and assets

- Preserve tracked `artifacts/city_recovery_ppo.v1.onnx`, `.zip`, metadata, `frozen_policy.v1.json`, and `manifest.lock.json`.
- Preserve tracked `evaluation/protocol.v1.json`, `policy_parity.v1.json`, `gate2-evidence.json`, and `feature_complete_report.v1.json`.
- The legacy linear candidate remains tracked and explicitly disclosed as non-PPO; it is not a fallback.
- Canonical saved comparisons are keyed by scenario/seed/policy/baseline identity and must restore byte-identically. Do not clear external result state if it matters.

## Completed state

- Evidence, Vertical Slice, and Feature Complete are accepted.
- The frozen PPO beats the OR-Tools baseline on the preregistered synthetic evaluation while maintaining zero recorded constraint/determinism violations; this remains a synthetic protocol result only.
- Independent graphic-design verification passed at the three required viewports.
- Live HEAD hardens cold setup, excludes the closed training toolchain from runtime setup, supports long Windows paths, and extends restart/persistence verification.

## Blockers and limitations

- No tracked independent Presentation PASS or Release PASS is present after the designer candidate; external evidence must be reconciled to an exact repository SHA.
- Clean-machine/empty-cache, firewall, long-path clone, full failure/process, and Release matrices must not be claimed merely because their harness was hardened.
- The environment is authored synthetic; there is no geographic, municipal, disaster, or empirical validation.

## Exact next actions

1. Keep HEAD clean except for this context file; do not retrain or regenerate frozen artifacts.
2. Reconcile any external Presentation/Release evidence with exact SHA `cece793` (or freeze a newer clean candidate), then record the accepted outcome in this `ai17agent.md` and the appropriate durable product records.
3. Run `scripts/verify.ps1 -Profile cpu` on the exact candidate, including restart persistence and loader-safe native dependency checks.
4. Route that same pushed SHA to an independent Presentation tester if no accepted ledger exists; only after Presentation PASS open the complete Release matrix and neutral judgment.
5. Update this `ai17agent.md` plus durable product records such as `README.md`/`EVALUATION.md` when the gate is independently reconciled, not merely from builder hardening results.

## Preservation warnings

- Do not delete or regenerate tracked policy/evaluation bytes; training is a closed toolchain.
- Preserve loader-safe external environments if avoiding a costly reinstall matters.
- Generated candidates after review: `.playwright-cli/`, `.pytest_cache/`, `.ruff_cache/`, `.run/`, `.venv/`, `__pycache__/`, `frontend/node_modules/`, `frontend/dist/`, TypeScript build info, and ignored `output/` captures.

## Consolidated sources

This file consolidates the resume-relevant parts of `README.md`, `PROJECT_BRIEF.md`, the former status/tester/judge notes, `ARCHITECTURE.md`, `DATA_AUDIT.md`, `EVALUATION.md`, `RESEARCH.md`, `DESIGN.md`, `DEMO_SCRIPT.md`, `ASSET_MANIFEST.md`, plus live Git status/log and the latest runtime-hardening diffs. Removed agent evidence is archived in `agent-cleanup-backup-20260717.zip`; this file is the workspace resumption authority.
