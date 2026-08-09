# Civic Relay

Civic Relay is a deterministic recovery simulation presented as an interactive toy-brick city. RELAY, a frozen Stable-Baselines3 PPO policy running through checksum-pinned ONNX CPU inference, divides each day's available units across five city services. The player applies typed disasters and watches the resulting trajectory unfold in a procedural 3D diorama. A visible OR-Tools GLOP planner evaluates the identical shock schedule for the end-of-run counterfactual and the Analyst Toolbox.

Every scenario, coefficient, dynamic, and training input is authored synthetic and non-empirical. This is local simulation evidence, not a real-city forecast or operational recommendation. The tracked legacy linear candidate remains disclosed as non-PPO and is never used as a runtime fallback.

## Architecture

The application runs as two independent processes:

```text
Terminal 1 — Backend API   uvicorn on 127.0.0.1:4117   (FastAPI + ONNX + OR-Tools)
Terminal 2 — Frontend dev   vite on    127.0.0.1:4173   (React 19 + Three.js)
```

The Vite dev server proxies `/api` and `/health` requests to the backend, so the frontend talks to the API through a single origin without CORS friction in the browser. The backend is API-only and never serves frontend assets.

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

## Prerequisites

- Python 3.12
- Node.js with npm
- `uv` 0.7.21 or newer

## Run

Start the backend and the frontend in two separate terminals.

### Backend (terminal 1)

```powershell
uv sync --frozen --python 3.12 --no-group training
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 4117 --no-access-log
```

### Frontend (terminal 2)

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:4173`. The application opens at `http://127.0.0.1:4173/#/game`.

The backend makes no outbound runtime connection. A port collision, missing policy artifact, or invalid frozen bundle is a blocking error: every route except `/health/live` returns structured `503 DEPENDENCY_NOT_READY` and the API is not served in a degraded state.

### Environment variables (optional)

| Variable | Purpose |
| --- | --- |
| `INNOVERSE_STATE_DIR` | Override the persisted result storage location (default `%LOCALAPPDATA%\Innoverse\ai17-city-recovery`). |
| `UV_PROJECT_ENVIRONMENT` | Redirect the Python venv to a shorter path. Use this when ONNX native DLLs hit the Windows 240-char loader limit on long clones. |
| `INNOVERSE_COMMIT` | Shown in `/api/v1/meta` (default `development`). |
| `INNOVERSE_PROFILE` | Shown in `/api/v1/meta` (default `cpu`). |

## Play

1. Choose **Sandbox** for an unlimited, unscored run or **Stress Test** for a six-disaster arsenal and a measured end-of-run debrief.
2. Choose **Calm**, **Moderate**, or **Severe**. These presets change ambient shock probability, severity bounds, and daily capacity; they never change the Stress Test arsenal.
3. Start the run. The first deterministic comparison is normally ready within the five-second start target.
4. Drag the plate to orbit, scroll to zoom, and use pause, day scrubber, or `0.5x` / `1x` / `2x` playback controls. At `1x`, a day advances about every two seconds; the camera stays above the plate.
5. Set severity from `0.05` to `0.40`, then drag one of the five engine shock types onto the city: aftershock, supply, epidemic, utility, or weather. For keyboard or touch play, select a shock card, choose one of the five named districts, and confirm the strike. The highlighted district reports that type's real impact strength.
6. The event strikes at the next day boundary. RELAY re-evaluates the complete seeded comparison, playback continues from the current day, and the city shows the returned impact, allocations, service condition, and repairs.

| Difficulty | Ambient shock probability | Severity band | Daily units |
| --- | ---: | ---: | ---: |
| Calm | `0.10` | `0.05–0.16` | `220` |
| Moderate | `0.20` | `0.10–0.28` | `180` |
| Severe | `0.34` | `0.18–0.40` | `140` |

The game never advances the simulator one frame at a time. Each throw appends a strict `forced_shocks` entry and repeats `POST /api/v1/simulations/compare`. The same seed, scenario, and ordered throw list produce the same canonical result. Every comparison still runs both planners and is persisted under its content-derived identity.

## Reading the city

The entire view is procedural: geometry, materials, animation, impact effects, and audio are generated locally with no external asset files or runtime network requests.

- Five districts visualize transport, housing, food, healthcare, and public services. Roughly eight reusable building archetypes appear as intact, slight, moderate, or rubble states according to the returned service values.
- Scaffolds, cranes, repair vehicles, traffic, and convoys are derived from real day-to-day recovery and allocation records. The central silo shows the day's available flow, including reductions caused by a shock.
- RELAY's matte-black orb speaks only deterministic lines derived from the active trajectory, shock, service condition, and allocation data.
- A low impact rumble, quiet RELAY blips, and a restrained district-dark drone use procedural WebAudio. Sound begins only after a browser-approved user gesture, is on by default, and can be disabled with the visible sound control.
- The scenario's base capacity arrives anew each day. A shock can reduce that day's available amount, and the projector allocates the entire available amount. RELAY controls where units go, not whether they are used.

City stakes use service condition only:

- **Stumble:** any service is below `0.12` at the end of a day.
- **District dark:** one service remains below `0.12` for at least three consecutive days; the district stays gray and still until recovery.
- **Fall:** food or healthcare remains below `0.12` for at least four consecutive days, or two or more services are below `0.12` on each of two consecutive days. The services in that cascade do not need to be the same pair on both days.

A fall ends playback on the first qualifying day. The debrief reports disasters endured, the worst moment, critical-floor recoveries, terminal weighted wellbeing, resilience AUC, and survival or fall. It then evaluates the already-returned baseline trajectory under the same rules and labels it exactly **conventional rule-based planner**. The baseline is never rendered as a second city.

## Analyst Toolbox

Open the complete Analyst Toolbox at `http://127.0.0.1:4173/#/toolbox` or use the quiet switch in the application header. It retains the raw scenario controls, both-planner trajectory comparison, daily allocation and projection audit, constraint evidence, and byte-identical saved-result restoration. A Toolbox result can launch directly into the city view without changing its authored scenario or issuing a replacement comparison.

If WebGL is unavailable, the game displays a clear fallback and the Analyst Toolbox remains usable.

## Recommendations

The Analyst Toolbox exposes a **Recommendations** tab alongside the Trajectory and Daily audit tabs. It turns the deterministic comparison into explicit decision guidance for a city planner:

- **Strategy summary** — which planner is recommended for the scenario family and the measured resilience AUC margin.
- **Actionable recommendations** — a deterministic list of concrete next steps (which service to reinforce, which shock type to keep reserves for, constraint-feasibility confirmation, and a synthetic-evidence disclosure).
- **Critical moment** — the lowest-resilience day, the shock that caused it, and the most fragile service across the run.
- **Daily recommendations** — a per-day table of the priority service, allocation focus, rationale, and risk alerts (critical, strained, or district-dark streaks).

Every recommendation is a deterministic function of the returned trajectory and shock tape. No recommendation is generated from a model call, random source, or external service; repeating the same scenario and seed reproduces the same recommendations byte-for-byte.

## Tests

```powershell
uv run pytest          # backend
cd frontend; npm test  # frontend
```

Backend tests cover health/meta endpoints, canonical determinism, constraint invariants, strict scenario validation, artifact drift detection, and persistence integrity. Frontend tests cover the game model, scene effects, camera framing, audio, stakes, and the application shell.

## API

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/meta`
- `GET /api/v1/simulations`
- `GET /api/v1/simulations/{result_id}`
- `POST /api/v1/simulations/compare`

Comparison schema `2.2.0` adds a deterministic `recommendations` block (strategy summary, actionable recommendations, critical moment, and per-day priority/risk assessment) on top of the `2.1.0` ordered `Scenario.forced_shocks` list. Strict unknown-field rejection and the legacy singular shock field are retained. Previously persisted `2.0.0` and `2.1.0` results remain self-verifying and restore with their original canonical bytes; they are not migrated.

The canonical compare response contains the shared shock schedule, both daily trajectories, action proposals, exact projected allocations, bounds and violation evidence, resilience and recovery metrics, deterministic recommendations, artifact provenance, deterministic result identity, and limitations. See `ARCHITECTURE.md` for runtime and identity contracts and `EVALUATION.md` for the preregistered synthetic holdout.

## Repository layout

```text
backend/app/      FastAPI app, simulator, scenarios, models, persistence, artifact guard
backend/tests/    backend pytest suite
frontend/src/    React + TypeScript source (App, game, api, types, scenarios)
frontend/tests/   frontend vitest suite
artifacts/        frozen policy bundle (ONNX, SB3 checkpoint, metadata, manifest, legacy)
evaluation/       preregistered protocol, parity evidence, gate2 evidence
```

The accepted policy and evaluation records are frozen release inputs. Normal setup, runtime, and tests do not rewrite them. Training and artifact generation are intentionally outside this product workflow.
