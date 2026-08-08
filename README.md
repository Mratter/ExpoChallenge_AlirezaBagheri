# Autonomous City Recovery Planner

> **Presentation workbench branch.** The active interface in `frontend/` is a
> model-evidence workbench. The previous game/product interface is preserved,
> unchanged, in `defunct/legacy-frontend/` and is no longer built or served.
> Run `scripts/setup.ps1 -Profile cpu` once, then `scripts/run.ps1 -Profile cpu`
> and open `http://127.0.0.1:4117`. See `WORKBENCH_PRESENTATION.md` for the
> presenter route. The historical product documentation below is retained as
> implementation context for the archived interface.

“An AI-assisted city-recovery planning and simulation platform. The policy can plan autonomously inside the digital twin, while operational decisions remain subject to human approval.”

Civic Relay is the fictional Relay City interface for the Autonomous City Recovery Planner. The production game still presents a deterministic 180-building city around the checksum-pinned MLP PPO v2 runtime. The additive scientific track now also contains a district-level CityRecoveryEnv-v4, a trained relational GNN + PPO + feasibility model, matched learned ablations, common non-learning baselines, and a reproducible preliminary evaluation. The simulator may auto-execute and replan inside controlled experiments. An operational plan cannot execute without explicit human approval, and this software never actuates real infrastructure.

The primary scientific question is: **Does a learned sequential policy improve long-horizon recovery, equity, and tail-risk outcomes over strong heuristic and optimization-only baselines when every method receives the same information and passes through the same feasibility constraints?** Aggregate training reward is not the headline endpoint, and implementation alone is not evidence that PPO or a graph model is superior.

This project’s contribution is the design, implementation, and evaluation of an integrated constrained sequential-planning system for city recovery. It combines established graph learning, reinforcement learning, optimization, simulation, and risk-analysis methods.

That contribution statement describes an integrated research program, not a superiority claim. V4 trained seven registered learned configurations with seeds `47017`, `47018`, and `47019`, each for `4,096` transitions, producing 21 checksum-indexed checkpoints. The completed v4 evaluation is **preliminary**, uses one held-out topology family, and is explicitly `claim_eligible=false`; none of its seven final scientific claim gates pass.

## Scientific evidence status

| Capability | Status | Evidence boundary |
| --- | --- | --- |
| Frozen manifest, redacted training view, checkpoint provenance, and training/test access guard | Implemented and tested | Preregistered manifest `d651…1084`; executed plan `34a9…a987`; v4 evidence remains preliminary. |
| Equal, random, needs-weighted heuristic, short-horizon MPC, MLP PPO, relational GNN, and ablations through one feasibility layer | Trained/evaluated in v4 | `1,980` held-out policy episodes; `2,748` total raw episodes including validation selection and constructed second-hit pairs. |
| Per-action solver intervention and rescue diagnostics | Implemented and evaluated | Random/equal/heuristic/MLP/GNN proposal diagnostics quantify feasibility, intervention, rescue, projection distance, and reranking; all eleven held-out policy aggregates record zero post-projection hard violations. |
| Relational GNN, matched MLP/GNN capacity, relation, topology, risk, equity, and reserve ablations | Evaluated / mixed preliminary evidence | GNN-vs-MLP, shuffled-edge, and risk-CVaR results are inconclusive; typed relational loses to homogeneous on the critical-service endpoint. |
| Joplin public case | Initial-state grounding is partial | No parameter calibration or quantitative recovery-trajectory validation; externally unvalidated. |
| V4 reward-gaming audit | Implemented and executed | 13/13 modeled adversarial cases pass under simulator hash `23b6…a2aa`; this is not universal or operational safety proof. |
| V4 uncertainty audit | Preliminary MC repeatability only | Three matched Judge tails have stable one-/two-/three-tail branch rankings and reported MCSE; the policies emit no probabilities or intervals, so forecast calibration is not evaluated. |
| V4 simulator sensitivity | Executed / preliminary baseline-favored stability | Nine named sets and 64 fixed-seed Latin-hypercube samples produce `26,280` policy rows. The selected heuristic remains the top policy in every tested configuration; no directional reversal occurs. This is not PPO-superiority or external-validity evidence. |
| Operator review, approval, rejection, override, re-projection, and simulation-only execution guard | Implemented and tested | This is workflow enforcement, not operational validation. |
| Judge Demo Mode | Executed / preliminary synthetic bundle | Uses held-out seed `472100`, learned seed `47017`, and matched rollout seeds; its own claim boundary remains false. |

The canonical v4 result uses the needs-weighted heuristic selected on the validation split before held-out construction. Relational PPO is worse on the registered critical-service primary endpoint: mean improvement `-0.65647900`, hierarchical paired 95% interval `[-0.69506321, -0.61297461]`, with `0/0/180` wins/ties/losses across 60 scenario/risk instances and three training seeds. Weighted unmet need and worst-district metrics point the other way, so the trade-off is reported rather than hidden. The comparison is preliminary and fails the final gate.

Architecture and ablation results are also mixed. Relational GNN versus parameter-matched MLP is inconclusive (`0.011717607`, 95% CI `[-0.057446737, 0.060656028]`). Relational versus homogeneous GNN is contradicted on critical-service loss (`-0.024740565`, 95% CI `[-0.040824202, -0.0062594357]`). Intact versus shuffled edges is inconclusive. Risk conditioning is now evaluated on the declared `cvar_10_weighted_unmet_need` endpoint and is inconclusive: mean improvement `-0.0040641`, 95% CI `[-0.012487927, 0.00828783]`. The learned reserve head improves ordinary held-out critical-service and unmet-need metrics, but its matched second-hit weighted-unmet-need difference-in-differences is `-0.0034944` with 95% CI `[-0.0039511232, -0.0030400853]`; positive would favor the learned head, so the specific second-hit claim is contradicted in this preliminary run.

The optimizer diagnostic is descriptive, not a causal decomposition. Random priorities + shared QP average `109.45815932` critical-service days lost, while relational-GNN PPO priorities + the same QP average `108.27764913`, a random-minus-learned gap of `1.18051019`. The validation-selected needs-weighted heuristic + QP is better still at `107.62117013`. Random proposals are feasible only `3.36%` of the time and are rescued by the QP in `96.64%` of actions; learned proposals are feasible `39.70%` of the time and rescued in `60.30%`. Because this aggregate contrast has no paired confidence interval or claim gate, it does not establish statistically supported PPO value beyond the optimizer.

The completed v4 sensitivity bundle under `artifacts/validation/sensitivity-v4-preliminary/` is `preliminary_non_claim`. Learned advantage is defined as comparator cost minus learned cost, so positive would favor PPO. It is negative at default (`-0.656479`), across all nine named sets (`-0.92556189` to `-0.51979908`), and across all 64 Latin-hypercube samples (`-2.44894005` to `-0.11665901`; mean `-1.05390051`). Top-policy stability is `1.0` and there are zero conclusion reversals: the stable conclusion is **baseline-favored**, not PPO-favored. All `26,280` sensitivity rows record zero modeled post-projection hard violations. See [VALIDATION.md](VALIDATION.md).

Generated claim status belongs in [CLAIMS.md](CLAIMS.md); simulator and historical evidence belongs in [VALIDATION.md](VALIDATION.md); model scope belongs in [MODEL_CARD.md](MODEL_CARD.md). From PowerShell, regenerate the table with `.\.venv\Scripts\python.exe scripts\generate_claims_v4.py`. Claim gates fail closed when evidence is absent, mismatched, or preliminary.

structurally realistic, authored-synthetic, not empirically calibrated to real disasters

Relay City is fictional. Every scenario, coefficient, dynamic, and training input is authored and non-empirical. This is local simulation evidence, not a real-city forecast or operational recommendation. The tracked legacy linear candidate remains disclosed as non-PPO and is never used as a runtime fallback.

## Play

The application opens at `http://127.0.0.1:4117/#/game`.

1. Choose an authored setting: **Fault-line city**, **Coastal storm season**, or **Fragile supply corridor**. The start screen discloses its authored type mix and raw severities but intentionally leaves the incident days for the debrief, where the complete returned tape is shown with source labels. Difficulty does not move or rewrite those authored incidents.
2. Choose **Sandbox** for an unlimited, unscored run or **Stress Test** for a six-disaster arsenal and a measured end-of-run debrief.
3. Choose **Calm**, **Moderate**, or **Severe**. The visible explanation gives the exact ambient shock probability, ambient severity envelope, and daily supply. These controls do not change shock-type weights, authored-event severity, the player severity control, the authored schedule, 20-day horizon, or Stress Test allowance.
4. Start the run. Standard game launches use a 20-day horizon, and the first deterministic comparison is normally ready within the five-second start target. Analyst Toolbox custom horizons remain exactly as authored.
5. Drag the plate to orbit, scroll to zoom, and use pause, day scrubber, or `0.5x` / `1x` / `2x` playback controls. At `1x`, the single presentation clock advances one day every `7,000ms`; the camera stays above the enlarged plate.
6. Set severity from `0.05` to `0.40`, then drag one of the five presented disasters onto the city: Earthquake, Supply, Epidemic, Utility, or Weather. Earthquake is the game-facing name for the unchanged engine key `aftershock`; no sixth shock type is introduced. For keyboard or touch play, select a shock card, choose one of the five named districts, and confirm the strike. The highlighted district reports that type's returned service-impact strength. Drop position is presentation only: the strict forced-shock record contains day, type, and severity, and the unchanged type vector determines the five citywide service impacts.
7. The event telegraphs over the current day and strikes at the next day boundary. RELAY re-evaluates the complete seeded comparison, playback continues from the current day, and Relay City shows the returned footprint, allocations, service condition, and multi-day repairs.

| Difficulty | Ambient shock probability | Severity band | Daily units |
| --- | ---: | ---: | ---: |
| Calm | `0.10` | `0.05–0.16` | `220` |
| Moderate | `0.20` | `0.10–0.28` | `180` |
| Severe | `0.34` | `0.18–0.40` | `140` |

### Authored scenario presets

These are game-content presets over the unchanged five shock types. They are not training data, held-out evaluation families, empirical forecasts, or extra engine mechanics.

| Preset | Setup disclosure | Authored incident days disclosed in the debrief |
| --- | --- | --- |
| Fault-line city | Earthquake ×2 at raw `0.24` and `0.31`; Utility ×1 at raw `0.22` | Days `3`, `9`, and `15` |
| Coastal storm season | Weather ×2 at raw `0.24` and `0.34`; Supply ×1 at raw `0.18` | Days `4`, `10`, and `16` |
| Fragile supply corridor | Supply ×2 at raw `0.22` and `0.32`; Utility ×1 at raw `0.20`; Epidemic ×1 at raw `0.18` | Days `3`, `8`, `13`, and `17` |

The terminal debrief lists every actual returned incident, including ambient and player-added events, and labels its provenance as **authored preset**, **player**, or **seeded ambient draw**. It also reconciles the authored plan: an entry displaced by a later forced event on the same day is marked overridden, and an entry after an early fall remains visible as unreached rather than being silently omitted. A restored or Toolbox result has no persisted game-session provenance, so its forced rows say **Stored forced event — origin unavailable** instead of guessing. The debrief is the disclosure boundary. The simulator precomputes the tape before rollout, but each planner receives only the identical current-day shock/context and neither observes future shocks.

### Guided eight-day tutorial

**Guided incident** is a fixed eight-day teaching run: seed `17008`, `180` units/day, ambient shock probability `0`, severity envelope `0.10–0.28`, and one authored Weather incident on day `2` at raw severity `0.24`. Its strongest typed footprint is Transport at `0.75`. No player disaster is appended during the lesson.

Day 1 telegraphs that real returned day-2 record. Day 2 walks through **IMPACT → ASSESSMENT → RESPONSE**; day 3 holds the heavy-response handoff. Days 4–8 explain **RECOVERY** only from returned service changes, allocations, depot stock, landed freight, scheduled next-day freight, capacity-constrained held/overflow freight, effective throughput, repair dispatch/supply, transfers, and spoilage. The tutorial never promises a recovery milestone that the eight-day result does not contain, and its debrief exposes the authored day/source exactly like a standard run.

The game never advances the simulator one frame at a time. Each throw appends a strict `forced_shocks` entry and repeats `POST /api/v1/simulations/compare`. The same seed, scenario, and ordered throw list produce the same canonical result. Every comparison still runs both planners and is persisted under its content-derived identity.

The game view uses one deterministic `requestAnimationFrame` presentation clock to remove visible day-boundary snapping. It samples only between values already returned by the candidate trajectory: normal days move from `services_before` to `services_end`; on a shock day the first 18% moves to the exact `services_after_shock` vector, ASSESSMENT holds that measured floor through 36%, and RESPONSE then moves toward `services_end`. Available arrival, depot stock/damage/throughput, road capacity, and dock/repair activity dressing ease between adjacent returned daily values. A cursor-derived 650ms-equivalent crew handoff holds yesterday's work-site pose at each boundary. Quintic easing and stable per-building offsets make the view continuous, but add no simulator tick, forecast, resource, or policy decision. Game-view service/stock readouts are labeled **visual interpolation**; exact allocations, manifests, shocks, daily logistics fields, and every Analyst Toolbox day remain discrete returned records.

Uninterrupted 1× playback is exactly `20 × 7s = 140s`. A complete six-kick Stress Test crosses roughly three minutes when the finite disasters are deliberately aimed under the documented slow-motion control and the debrief is read; `output/playwright/r3-stress-pacing/r3-stress-pacing-index.json` records the actual desktop start-to-debrief interaction. Playback speed changes presentation time only; it does not alter the engine trajectory or deterministic result.

## Reading the city

The entire view is procedural: geometry, materials, animation, impact effects, and audio are generated locally with no external asset files or runtime network requests.

- Five districts visualize transport, housing, food, healthcare, and public services across an enlarged, dense plate. Each district contains 36 buildings, for 180 total, arranged around a denser road and lane network. Roughly eight reusable archetypes appear as intact, slight, moderate, or rubble states according to returned service values.
- A central intake hub receives the day's exact available supply from the plate edge. In schema-v3 runs, heavy line-haul cargo is the recorded `pending_arrivals_landed + same_day_delivery_landed`; typed last-mile cargo is recorded `repair_supply`; and mutual-aid vehicles require a transfer event that agrees with its signed net vector. Routine `pending_next_day` is an inbound schedule, not a physical truck queue. A queue appears only for capacity-constrained `capacity_overflow`, equal to the recorded held-arrival vectors. An archived v1 result instead uses the exact allocation only as a labeled presentation fallback and states `legacy v1 result — depot state was not recorded`; the interface never invents missing depot state.
- A rubbled district depot uses a disclosed longer presentation route from the nearest non-rubble point of distribution only when one exists. This route is separate from mutual aid and never creates cargo or guesses a donor.
- Scaffolds, lifts, cranes, repair vehicles, traffic, and convoys are derived from real day-to-day recovery and logistics quantities. The complete Toolbox manifest partitions those quantities into fixed 12-unit heavy and 4-unit last-mile load equivalents. The diorama shows a deterministic bounded subset—at most 17 road mission slots, including no more than two civilian cars and one commuter vehicle—without merging or inflating a selected load. This is a view cap, not a fabricated engine fleet resource.
- Every selected vehicle starts at its semantic origin (plate ingress, hub yard, donor depot, or district depot), follows the same orthogonal centerline graph that renders the roads, docks at the named depot or a visible curb beside the named rebuilding site, dwells, and retraces the road to its origin. Motion is distance-based at roughly five world units per presented day before real weather/throughput modifiers, so a route three times longer takes three times longer rather than crossing the plate in the same few seconds. Ordinary traffic remains still through IMPACT/ASSESSMENT and resumes from the identical road position at RESPONSE, while the cargo-free assessment wave continues.
- Repairs follow the trajectory's true recovery arc. A building joins a stable staggered cohort only on a positive realized service tick and, for schema-v3 results, positive effective repair supply. Its work site and crew then remain through the multi-day arc; progress pauses whenever the returned service trajectory pauses, and damage tier controls how much of that arc the individual site requires.
- Service bars, wellbeing, city condition, lighting, repair dressing, depot state/activity, and mission travel share the same frame clock. Intermediate game-view numbers are labeled as visual interpolation; exact vehicle manifests and operational inspector rows still reconcile to their current returned day. Allocations, shocks, stakes, narration facts, debrief values, persistence bytes, and all Toolbox day values remain exact returned records.
- Disaster presentation follows the returned typed footprint. Weather gathers clouds before a rain-and-wind burst; Earthquake adds a restrained camera tremor and crumble; Supply interruption thins inbound freight and disrupts hub/road traffic; Epidemic presents response logistics around healthcare; Utility failure produces an infrastructure flicker weighted by real service impact.
- RELAY's matte-black orb speaks only deterministic lines derived from the active trajectory, shock, phase-correct service condition, allocation, and optional recorded logistics. Its v3 copy distinguishes scheduled next-day freight from capacity-constrained held/overflow freight.
- A low impact rumble, quiet RELAY blips, and a restrained district-dark drone use procedural WebAudio. Sound begins only after a browser-approved user gesture, is on by default, and can be disabled with the visible sound control.
- The scenario's base capacity arrives anew each day. A shock can reduce that day's available amount, and the projector allocates the entire available amount. RELAY controls where units go, not whether they are used.

City stakes use service condition only:

- **Stumble:** any service is below `0.12` at the end of a day.
- **District dark:** one service remains below `0.12` for at least three consecutive days; the district stays gray and still until recovery.
- **Fall:** food or healthcare remains below `0.12` for at least four consecutive days, or two or more services are below `0.12` on each of two consecutive days. The services in that cascade do not need to be the same pair on both days.

A fall ends playback on the first qualifying day. The debrief reports disasters endured, the worst moment, critical-floor recoveries, terminal weighted wellbeing, resilience AUC, and survival or fall. It then evaluates the already-returned baseline trajectory under the same rules and labels it exactly **conventional rule-based planner**. The baseline is never rendered as a second city.

## Analyst Toolbox

Open the complete Analyst Toolbox at `http://127.0.0.1:4117/#/toolbox` or use the quiet switch in the application header. It retains the raw scenario controls, including custom horizons up to the engine limit, both-planner trajectory comparison, daily allocation and projection audit, constraint evidence, and byte-identical saved-result restoration. A Toolbox result can launch directly into the city view without changing its authored scenario or issuing a replacement comparison.

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

The verifier checks the frozen artifacts and environment, runs backend and frontend tests, builds the production UI, starts the loopback application, submits the same 11-day fixture five times, checks canonical result bytes plus every allocation and v2 logistics-ledger invariant, restarts the server, restores the persisted result byte-identically, rejects invalid input, and shuts the server down.

Final Realism R3 presentation acceptance is desktop-only at 1440×900. Earlier narrow-viewport captures are retained as historical evidence and are not part of the R3 gate.

The accepted v1 and v2 policy/evaluation records are frozen release inputs. Normal setup, runtime, and verification do not rewrite them. Training and artifact generation are intentionally outside this product workflow.

## CityRecoveryEnv v2 evidence

The additive v2 evidence chain is pinned before runtime activation:

| Evidence | SHA-256 |
| --- | --- |
| Registered protocol `evaluation/protocol.v2.json` | `a67d3bfc4842639e6d976c311ed9b7b35b83e59dbcbb130643ed16dba1bdc0e1` |
| Registered engine specification | `e6d9fed7346fd3e12b2837c6e31ea26179d340d6cc89b671162ba70624c8c89c` |
| SB3 PPO v2 checkpoint | `5da60929411320ca30cf50a8716cfa6965394fb01adfd72d9aa02346e673cc3b` |
| ONNX v2 runtime policy | `0c40a585b0cddc5d4564e4a1ae6af2ed651a85a08b09c5e374be07278aa0ed20` |
| V2 metadata | `27b185dd30101d2291cda2ab4c3b25d2c75e0d46c26403a9e5e509f9fb6421ab` |
| PyTorch/ONNX parity report | `0ff46cd2b12de0e3aae68db7ba819b393b8c779d7425ab848bf360f1303047e1` |
| Manifest v3 | `6cadcdc7490714ebcd839322ae3067b9425e3dd04ad1d3e2998f3c006277411d` |
| Held-out report `evaluation/feature_complete_report.v2.json` | `cd7780ebcdfcf81035697560f5e4726f2228394bfcea238e0194951c1ac7bfc6` |

The preregistered holdout contains five unseen families and eight disjoint seeds: 40 complete units, each executed five times, for 200 canonical comparisons. There were zero repeat mismatches, zero candidate or baseline allocation violations, and zero registered logistics-ledger violations. Mean resilience AUC was `0.44342203` for PPO/ONNX and `0.43489304` for GLOP, a measured delta of `+0.00852899` with paired 95% interval `[0.00598309, 0.01099349]`; the candidate was higher in 32 of 40 units.

The result is a trade-off, not a blanket superiority claim. Candidate-minus-baseline critical service-days were `+1.325`, recovery days `+1.1`, post-shock shortfall AUC `+0.00034038`, mean depot stock fraction `-0.02150718`, and food spoilage `+3.17991933` units. The candidate also recorded `-1048.07529207` queued-delivery unit-days, `-225.38716724` throughput-constrained unit-days, and `-50.4` mutual-aid transfer units. These measurements describe only the fixed authored-synthetic protocol.

## API

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/meta`
- `GET /api/v1/simulations`
- `GET /api/v1/simulations/{result_id}`
- `POST /api/v1/simulations/compare`
- `POST /api/v1/planning-sessions`
- `POST /api/v1/plans`
- `GET /api/v1/plans/{plan_id}`
- `POST /api/v1/plans/{plan_id}/review`
- `POST /api/v1/plans/{plan_id}/approve`
- `POST /api/v1/plans/{plan_id}/reject`
- `POST /api/v1/plans/{plan_id}/override`
- `POST /api/v1/plans/{plan_id}/reproject`
- `POST /api/v1/plans/{plan_id}/execute`

Comparison schema `3.0.0` records CityRecoveryEnv-v2 identity and the complete daily logistics ledger while retaining strict unknown-field rejection, the ordered `Scenario.forced_shocks` list, and the legacy singular forced-shock field. Previously persisted `2.0.0` and `2.1.0` results remain self-verifying and restore with their original canonical bytes; they are not migrated or default-filled.

Manifest v3 lists the immutable v1 bundle and the additive v2 bundle together and marks `city-recovery-sb3-ppo-v2` active. The exact former five-record manifest is archived as `artifacts/manifest.v2.lock.json`, SHA-256 `1958b99ec0a52bc651ebea07f9923af2be6684ce37e73d406656acba27377205`, so v1 provenance remains independently auditable.

The canonical compare response contains the shared shock schedule, both daily trajectories, action proposals, exact projected allocations, bounds and violation evidence, resilience and recovery metrics, artifact provenance, deterministic result identity, and limitations. See `ARCHITECTURE.md` for runtime and identity contracts and `EVALUATION.md` for the preregistered synthetic holdout.
