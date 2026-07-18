# Civic Relay Redesign Ship Report

Date: 2026-07-18
Status: Shipped; the M4 candidate passed the complete CPU profile and compiled-app review.

## Scope delivered

The AI17 experience now opens as a game-first, procedural 3D recovery diorama while retaining the complete Analyst Toolbox. The frozen PPO, shared simulator, planner math, projector, policy identity, evaluation records, and persistence rules remain release inputs rather than redesign targets.

The shipped interaction model is deliberately narrow: the player controls disasters, camera, and playback; RELAY controls the five-way allocation. Map activity and speech are views over comparison data. Every kick adds an ordered forced shock, repeats the deterministic comparison, and resumes from the active day. The conventional baseline is evaluated in the debrief and Toolbox but is not rendered as a second city.

## Milestones

| Milestone | Commit | Delivered evidence |
| --- | --- | --- |
| M1 — The diorama lives | `44dec0f` | Game route, procedural city and baseplate, bounded camera, timed playback, data-driven damage/rebuilding, silo, RELAY, and preserved Toolbox switch. |
| M2 — The kick | `21c3d36` | Strict additive `forced_shocks`, schema `2.1.0`, deterministic re-comparison, drag-to-throw disasters, typed impact effects, allocation convoys, repair traffic, and legacy restore coverage. |
| M3 — Stakes | `369092e` | Start screen, Sandbox and Stress Test, three difficulty presets, six-disaster arsenal, exact stumble/dark/fall rules, collapse screen, debrief, and identical-shock conventional-planner counterfactual. |
| M4 — Polish | This M4 implementation commit | Procedural audio and mute control, Toolbox visual integration, final camera/impact polish, updated product documentation and tests, and final screenshot review. |

## Contract checks

- Frozen policy and evaluation files: `git diff -- artifacts evaluation` was empty, and the full preflight hash guard passed on the final candidate.
- Loopback-only runtime and bundled dependencies: every request recorded during the compiled-app review used `http://127.0.0.1:4117`; no external origin appeared.
- No downloaded 3D model, texture, audio, font, CDN, LLM, or TTS runtime dependency: confirmed by the final source scan and compiled-app network observation; visuals and sound are procedural or bundled code.
- Deterministic kick path: additive forced shock followed by `POST /api/v1/simulations/compare`; returned schedule is validated before consuming a Stress Test disaster.
- Strict input model: Pydantic unknown-field rejection retained; `forced_shocks` days and entries remain bounded and typed.
- Legacy persistence: schema `2.0.0` comparisons without the additive list restore their original canonical bytes without migration.
- Full allocation invariant: all shock-adjusted daily available units are projected across the five services; game copy does not describe unused funds.
- City-condition stakes: critical floor `< 0.12`; dark after three consecutive days; essential fall after four; cascade fall after two.
- Baseline presentation: no second rendered city; end-of-run counterfactual is labeled **conventional rule-based planner**.
- Protected wording and symbols: the final scoped, case-insensitive scan was clean outside the redesign contract itself and frozen release inputs.

## Verification record

Run all commands against the exact final M4 commit with port `4117` free.

| Check | Command or method | Final result |
| --- | --- | --- |
| Frontend tests | `npm test --prefix frontend -- --run` | PASS — 9 files, 65 tests. |
| Production build | `npm run build --prefix frontend` | PASS — 2,343 modules; Vite reports one non-blocking chunk-size advisory for the bundled 3D application. |
| Full CPU verifier | `.\scripts\verify.ps1 -Profile cpu` | PASS — 37 backend tests, 65 frontend tests, build and preflight, five deterministic repeats, 110 constraint checks, restart persistence, and byte-identical restore. |
| Live game flow | `.\scripts\run.ps1 -Profile cpu`, browser at `127.0.0.1:4117` | PASS — start-to-city measured 24 ms locally; playback, accessible and drag kicks, impact, collapse/debrief, saved restore, and Toolbox round-trip reviewed. |
| Browser console | Real compiled application | PASS — 0 errors and 0 warnings after the complete review flow. |
| Runtime requests | Browser network observation | PASS — all document, bundle, API, list, and restore requests remained on `127.0.0.1:4117`. |
| Repository state | `git status --short` | Clean after the verified M4 changes and this handoff record were committed. |

The full verifier is expected to cover artifact readiness, backend and frontend suites, production build, five identical 11-day submissions, canonical byte equality, daily allocation bounds and sums, invalid-input rejection, restart persistence, and byte-identical restoration.

## Visual review

Generated browser captures are intentionally ignored working evidence under `output/playwright/`:

- M1: `m1/m1-accepted-1440x900.png`
- M2: `m2/m2-impact-final.png`
- M3: `m3/m3-start.png`, `m3/m3-playing.png`, `m3/m3-collapse.png`, `m3/m3-debrief-final.png`
- M4: `m4/m4-start-1440x900.png`, `m4/m4-start-portrait-390x844.png`, `m4/m4-game-contained-1440x900.png`, `m4/m4-game-contained-390x844.png`, `m4/m4-aim-accessible.png`, `m4/m4-impact-food-final.png`, `m4/m4-toolbox-trajectory.png`, `m4/m4-toolbox-audit.png`, `m4/m4-collapse-focus.png`, and `m4/m4-debrief-focus.png`. Final self-review: **8.5/10** overall.

Final review notes: the loop corrected an over-distant camera fit, fog-muted geometry, plate-edge cropping, impact effects disappearing into the terrain, small low-contrast labels, narrow-screen overlay collisions, and modal focus. The accepted final composition keeps the city dominant at `1440×900`, retains the complete plate at `390×844`, and gives restrained impacts enough contrast to read. The only accepted build advisory is Vite's chunk-size warning for the bundled 3D application; it does not affect build or runtime correctness.

## Known boundaries

- The environment and evidence are authored synthetic and non-empirical. They are not geographic, municipal, disaster-response, or operational validation.
- WebGL is required for the 3D city; the application provides an explicit fallback to the fully functional Analyst Toolbox.
- Training and artifact generation are closed workflows and are not part of setup, runtime, or this redesign verification.
- Independent Presentation and Release acceptance must be tied to the exact final commit; builder verification alone does not establish those gates.

## Final handoff

- Release implementation commit: this M4 commit; its immutable SHA is reported in the final delivery summary.
- Verification completed: `2026-07-18T06:16:02+03:30`
- Verification state directory: ignored isolated state under `.run/verification-m4-final-camera`.
- Remaining product gaps: None. WebGL fallback and the non-blocking bundle-size advisory remain documented boundaries.
