# Civic Relay

AI17's Autonomous City Recovery Planner is a deterministic synthetic simulator for comparing a visible urgency allocation planner with a frozen policy candidate across shared shocks and hard daily constraints.

The Gate 2 candidate is a checksum-verified linear heuristic selected on synthetic scenarios. It is not PPO, not empirically trained, and not operational guidance.

## Windows 11 CPU Run

```powershell
.\scripts\setup.ps1 -Profile cpu
.\scripts\preflight.ps1 -Profile cpu -Full
.\scripts\run.ps1 -Profile cpu
```

Open `http://127.0.0.1:4117`. `run.ps1` does not open a browser and uses no outbound runtime connection. A port collision, missing compiled UI, or corrupt artifact is a blocking error.

Run the complete live verification separately while port 4117 is free:

```powershell
.\scripts\verify.ps1 -Profile cpu
```

Verification runs backend/frontend checks, starts the compiled application, submits a new 11-day scenario five times, checks byte stability and every allocation constraint, rejects invalid input, then stops the server.

## API

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/meta`
- `POST /api/v1/simulations/compare`

The compare request contains a seed and bounded scenario. The response contains the entire shared shock schedule, both full daily trajectories, projector evidence, resilience AUC comparison, artifact provenance, and limitations. JSON keys are sorted and compact for stable byte comparison.

See `EVALUATION.md` for exact Gate 2 evidence and `ARCHITECTURE.md` for the authored synthetic equations.
