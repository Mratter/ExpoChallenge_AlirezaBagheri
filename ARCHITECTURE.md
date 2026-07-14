# Architecture

## Runtime Shape

`scripts/run.ps1` launches one uv-managed Python 3.12 process on `127.0.0.1:4117`. FastAPI owns `/health/*` and `/api/v1/*`, then serves the compiled React/Vite files at `/`. Runtime requires no outbound network. Missing frontend output, a corrupt policy checksum, or port collision blocks startup/readiness rather than selecting a fallback.

```text
React scenario editor
        |
POST /api/v1/simulations/compare
        |
strict Pydantic bounds -> one PCG64 shock tape
        |                         |
visible urgency planner     frozen candidate artifact
        |                         |
        +---- same capped-simplex projector ----+
                              |
                    full daily trajectories
```

## Synthetic Dynamics

Service order is transport, housing, food, healthcare, public services. For shock severity `s`, impact vector `v`, base budget `B`, dependency matrix `D`, and current service vector `q`:

```text
q'      = clip(q * (1 - s*v), 0, 1)
B_t     = B * (1 - s*budget_factor)
support = 0.55 + 0.45 * (D @ q')
gain    = eta * sqrt(x/200) * support * (1-q')
strain  = delta * max(0, 0.35-q') * (1-x/B_t)
q_next  = clip(q' + gain - strain, 0, 1)
```

`eta=[.18,.16,.20,.22,.17]`, `delta=[.010,.012,.015,.018,.010]`, all shock vectors, and `D` are authored synthetic coefficients in `backend/app/simulator.py`. They are not estimated from outcomes.

## Shared Constraint Boundary

Both planners emit a proposal `y`. The only allocation boundary applies lower `0.04*B_t` when a service is below `.30`, upper `0.50*B_t`, and sum `B_t`:

```text
x = clip(y - lambda, lower, upper)
```

`lambda` is found by 64 float64 bisections between `min(y-upper)` and `max(y-lower)`, then values are rounded to eight decimals with deterministic residual repair. The response exposes distance, each bound binding, projected sum, and measured sum/lower/upper violations per day and planner.

## Planner Boundary

The baseline score is `priority*(1-q')*(2.5 when q'<.30 else 1)` and is normalized to the available budget before projection. The candidate is a checksum-verified linear mix of normalized deficit, criticality, marginal-gain, and network-centrality features. `scripts/build_policy_artifact.py` exhaustively scores 56 weight vectors on five declared synthetic calibration scenarios. This is not reinforcement learning.

## API Contract

- `GET /health/live`: process liveness only
- `GET /health/ready`: verifies the required policy artifact checksum
- `GET /api/v1/meta`: commit, profile, seed, model, generator, and synthetic dataset metadata
- `POST /api/v1/simulations/compare`: strict scenario in; canonical sorted JSON comparison out

Validation, missing dependency, and computation failures use `{ "error": { "code", "message", "details" } }`. No degraded response is returned.
