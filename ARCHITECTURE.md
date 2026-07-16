# Architecture

## Runtime Shape

`scripts/run.ps1` launches one uv-managed Python 3.12 process on `127.0.0.1:4117`. FastAPI owns `/health/*` and `/api/v1/*`, then serves the compiled React/Vite files at `/`. Runtime requires no outbound network. Missing frontend output, port collision, or any invalid required policy artifact blocks the product rather than selecting a fallback. The request guard revalidates the complete artifact graph for every route except process liveness.

```text
React scenario editor / saved-result restore
                    |
        strict bounded Pydantic scenario
                    |
    complete PCG64 shock tape generated once
              /                 \
 SB3 PPO -> ONNX action     OR-Tools GLOP plan
              \                 /
       same capped-simplex projector
                    |
       same synthetic state transition
                    |
 full daily trajectories + resilience/recovery metrics
                    |
 canonical content-addressed local persistence
```

## Gymnasium Environment

`CityRecoveryEnv` has five ordered resource classes: transport, housing, food, healthcare, and public services. Its 23 float32 observations expose current post-shock service state, normalized priorities, dependency support, current shock impact, available-budget fraction, remaining-horizon fraction, and current severity. Its five bounded actions are converted with a stable softmax into a positive budget proposal. The action is not an allocation until the common projector runs.

`reset(seed=...)` constructs the entire PCG64 tape before a planner acts. `step` records the shock, state before/after shock, observation inputs, raw action, proposal, lower/upper bounds, projected allocation, bindings, measured violations, support, gain, strain, end state, resilience, and reward. A deterministic cycling wrapper trains on complete scenario/seed units.

## Synthetic Dynamics

For shock severity `s`, impact vector `v`, base budget `B`, dependency matrix `D`, current service vector `q`, and projected allocation `x`:

```text
q'      = clip(q * (1 - s*v), 0, 1)
B_t     = B * (1 - s*budget_factor)
support = 0.55 + 0.45 * (D @ q')
gain    = eta * sqrt(x/200) * support * (1-q')
strain  = delta * max(0, 0.35-q') * (1-x/B_t)
q_next  = clip(q' + gain - strain, 0, 1)
```

`eta=[.18,.16,.20,.22,.17]`, `delta=[.010,.012,.015,.018,.010]`, shock vectors, dependencies, scenario-family centers, and every other coefficient are authored synthetic values. They are not estimated from outcomes.

## Shared Constraint Boundary

The common boundary applies lower `0.04*B_t` when service state is below `.30`, upper `0.50*B_t`, total budget `B_t`, and exact sum `B_t`:

```text
x = clip(y - lambda, lower, upper)
```

`lambda` is found by 64 float64 bisections, followed by eight-decimal rounding and deterministic residual repair. Each day exposes the proposal distance, bindings, projected sum, explicit lower/upper arrays, and independently measured lower, upper, budget, and sum violation counts.

## Planner Boundary

The candidate is a real Stable-Baselines3 2.7.0 PPO `MlpPolicy`, trained for 30,000 CPU steps on four authored training families. The deterministic PyTorch action is exported at ONNX opset 17. Runtime uses only the checksum-pinned ONNX graph through sequential, single-thread `CPUExecutionProvider`; the shared action-to-proposal conversion and constraint projector remain outside the graph.

The default CPU setup installs the runtime and test dependency graph and explicitly excludes the non-default `training` group. PyTorch 2.8.0 and Stable-Baselines3 2.7.0 remain pinned in that group for deliberate model-build reproduction, but neither is imported by the shipped runtime, preflight, evaluation, or test paths.

The baseline is OR-Tools 9.14.6206 GLOP. Every day it maximizes:

```text
sum(priority * deficit * (eta * support + 0.04 * dependency_centrality) * allocation)
```

under the identical lower, upper, and sum constraints. Its solution still passes through the common projector. The response exposes solver, status, objective text, and coefficients. Neither planner sees future shocks.

The 722-byte accepted linear artifact is preserved with its original SHA-256 and non-PPO disclosure. Runtime never substitutes it for the SB3/ONNX candidate.

## Artifact Boundary

Manifest schema v2 requires exactly five CC0-1.0 records: accepted legacy linear JSON, SB3 checkpoint, ONNX graph, policy metadata, and PyTorch/ONNX parity evidence. Exact id, role, path, source, byte count, and SHA-256 are validated. Cross-document hashes must agree; the legacy artifact may not be relabeled; parity tolerances must pass; ONNX must parse, pass `onnx.checker`, match input/output schemas, and return a finite five-action smoke result.

Any failure produces `503 DEPENDENCY_NOT_READY` for readiness, metadata, simulation, persistence, and primary UI routes. `/health/live` remains process liveness only.

## Persistence Boundary

A successful result id is SHA-256 over comparison schema, seed, canonical scenario, ONNX policy hash, and baseline id. `RunStore` writes canonical sorted JSON atomically to `runs/{result_id}.json`. Repeating an identical comparison rewrites identical bytes at the same id. Listing is ordered by result id. Restore verifies filename/id, recomputed identity, JSON validity, and canonical bytes. Corrupt results fail explicitly; they are not skipped or recomputed.

## API Contract

- `GET /health/live`: process liveness only
- `GET /health/ready`: complete policy dependency and inference readiness
- `GET /api/v1/meta`: commit, profile, seed, SB3/ONNX/parity/legacy, OR-Tools, persistence, and synthetic dataset metadata
- `POST /api/v1/simulations/compare`: strict scenario in; canonical persisted comparison out
- `GET /api/v1/simulations`: deterministic saved-result summaries
- `GET /api/v1/simulations/{result_id}`: integrity-checked canonical result restore

Validation, dependency, persistence, and computation failures use `{ "error": { "code", "message", "details" } }`. No degraded response is returned. React clears prior evidence before displaying a failed state.
