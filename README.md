# Civic Relay

Civic Relay is a deterministic recovery simulation presented as an interactive toy-brick city. RELAY, a frozen Stable-Baselines3 PPO policy running through checksum-pinned ONNX CPU inference, divides each day's available units across five city services. The player applies typed disasters and watches the resulting trajectory unfold in a procedural 3D diorama. A visible OR-Tools GLOP planner evaluates the identical shock schedule for the end-of-run counterfactual and the Analyst Toolbox.

Every scenario, coefficient, dynamic, and training input is authored synthetic and non-empirical. This is local simulation evidence, not a real-city forecast or operational recommendation. The tracked legacy linear candidate remains disclosed as non-PPO and is never used as a runtime fallback.

## Play

The application opens at `http://127.0.0.1:4117/#/game`.

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

Open the complete Analyst Toolbox at `http://127.0.0.1:4117/#/toolbox` or use the quiet switch in the application header. It retains the raw scenario controls, both-planner trajectory comparison, daily allocation and projection audit, constraint evidence, and byte-identical saved-result restoration. A Toolbox result can launch directly into the city view without changing its authored scenario or issuing a replacement comparison.

If WebGL is unavailable, the game displays a clear fallback and the Analyst Toolbox remains usable.

## Windows 11 CPU run

Requirements are Python 3.12, Node.js with npm, and `uv` 0.7.21 or newer.

```powershell
.\scripts\setup.ps1 -Profile cpu
.\scripts\preflight.ps1 -Profile cpu -Full
.\scripts\run.ps1 -Profile cpu
```

Open `http://127.0.0.1:4117`. `run.ps1` serves the compiled frontend and API from one loopback process, does not open a browser, and makes no outbound runtime connection. A port collision, missing compiled UI, or invalid frozen bundle is a blocking error. If a required artifact becomes unavailable after startup, every route except `/health/live` returns structured `503 DEPENDENCY_NOT_READY`; the primary UI is not served in a degraded state.

Setup uses the frozen dependency lock, verifies package hashes, excludes the closed training toolchain, installs pinned frontend dependencies, and builds `frontend/dist`. Normal paths use the repository `.venv`. When a long clone approaches the Windows native-loader limit, the scripts select a short root-hashed environment under `%LOCALAPPDATA%\Innoverse\ai17-city-recovery\environments`. An absolute short `UV_PROJECT_ENVIRONMENT` can override it.

Successful comparison results are stored under `%LOCALAPPDATA%\Innoverse\ai17-city-recovery` unless `INNOVERSE_STATE_DIR` selects another directory.

## Verification

Run the bounded CPU verification while port `4117` is free:

```powershell
.\scripts\verify.ps1 -Profile cpu
```

The verifier checks the frozen artifacts and environment, runs backend and frontend tests, builds the production UI, starts the loopback application, submits the same 11-day fixture five times, checks canonical result bytes and every allocation invariant, restarts the server, restores the persisted result byte-identically, rejects invalid input, and shuts the server down.

The accepted policy and evaluation records are frozen release inputs. Normal setup, runtime, and verification do not rewrite them. Training and artifact generation are intentionally outside this product workflow.

## API

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/meta`
- `GET /api/v1/simulations`
- `GET /api/v1/simulations/{result_id}`
- `POST /api/v1/simulations/compare`

Comparison schema `2.1.0` adds the ordered `Scenario.forced_shocks` list while retaining strict unknown-field rejection and the legacy singular field. Previously persisted `2.0.0` results remain self-verifying and restore with their original canonical bytes; they are not migrated.

The canonical compare response contains the shared shock schedule, both daily trajectories, action proposals, exact projected allocations, bounds and violation evidence, resilience and recovery metrics, artifact provenance, deterministic result identity, and limitations. See `ARCHITECTURE.md` for runtime and identity contracts and `EVALUATION.md` for the preregistered synthetic holdout.
