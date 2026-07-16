# Civic Relay

AI17's Autonomous City Recovery Planner is a deterministic synthetic simulator for comparing a frozen Stable-Baselines3 PPO / ONNX policy with a visible OR-Tools GLOP planner across identical seeded shocks and hard daily constraints.

All scenarios, dynamics, coefficients, and policy training inputs are authored synthetic and non-empirical. This is local simulation evidence, not a forecast or operational recommendation. The accepted Gate 2 linear candidate remains tracked and disclosed as non-PPO; it is not used as a runtime fallback.

## Windows 11 CPU Run

```powershell
.\scripts\setup.ps1 -Profile cpu
.\scripts\preflight.ps1 -Profile cpu -Full
.\scripts\run.ps1 -Profile cpu
```

Setup requires `uv` 0.7.21 or newer and preserves the frozen lock and package hash verification. It installs the CPU runtime and test environment while explicitly excluding the closed training toolchain. For reliable wheel acquisition on Windows it supplies conservative, process-scoped defaults (four downloads, two installs, six HTTP retries, and a 120-second read timeout); explicit caller values take precedence, and the prior environment is restored on exit.

Open `http://127.0.0.1:4117`. `run.ps1` does not open a browser and makes no outbound runtime connection. A port collision, missing compiled UI, or invalid artifact bundle is a blocking error. If a required artifact is lost after startup, every route except `/health/live` returns structured `503 DEPENDENCY_NOT_READY`; the primary UI is not served in a degraded state.

Run the bounded live verification while port 4117 is free:

```powershell
.\scripts\verify.ps1 -Profile cpu
```

Verification runs the functional suites and production build, starts the compiled application, submits a new 11-day scenario five times, checks canonical bytes and every allocation invariant, restores the persisted result byte-identically, rejects invalid input, and stops the server.

Regenerate evaluation evidence from the frozen artifact:

```powershell
uv run --frozen python scripts/evaluate_policy.py
```

Model training/export is intentionally separate from normal setup/runtime:

```powershell
uv run --frozen --no-default-groups --group training python scripts/train_policy.py --timesteps 30000
```

Torch and Stable-Baselines3 remain exactly pinned in the non-default `training` group. The training command rewrites the SB3, ONNX, parity, metadata, and manifest bundle; it is not part of normal setup or runtime. A resulting change is a new model candidate and requires evaluation and independent review.

## API

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/meta`
- `GET /api/v1/simulations`
- `GET /api/v1/simulations/{result_id}`
- `POST /api/v1/simulations/compare`

The compare request contains a seed and bounded authored scenario. The canonical response contains the complete shared shock schedule; both full daily trajectories; raw actions/proposals; lower/upper bounds; projection and violation evidence; resilience/recovery metrics; artifact provenance; deterministic result identity; and limitations. Successful results are stored locally under `%LOCALAPPDATA%\Innoverse\ai17-city-recovery` unless `INNOVERSE_STATE_DIR` explicitly selects another directory.

See `EVALUATION.md` for the preregistered 40-case holdout and `ARCHITECTURE.md` for the environment, planner, projector, artifact, and persistence contracts.
