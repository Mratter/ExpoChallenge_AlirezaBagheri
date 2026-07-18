# AI17 Redesign Directive — "The City You Can't Knock Over"

Agreed with the user on 2026-07-18 after a full /grill-me session. This document is the
contract for the redesign build. It overrides DESIGN.md for presentation decisions; it does
NOT override the engine thesis, artifact identity, or determinism rules in PROJECT_BRIEF.md
and ARCHITECTURE.md.

## Vision (north star)

A serious simulation presented as an approachable interactive toy — neal.fun energy, not
arcade. A full-3D toy-brick diorama city sits on a baseplate you can grab and rotate. RELAY,
the frozen PPO policy, keeps the city balanced like a self-balancing robot; the player is
the kick. You throw disasters at the miniature city, RELAY visibly reroutes resources,
repairs buildings, and announces its counter-moves. It is genuinely difficult — but
possible — to make it fall. The current dashboard is demoted to an "Analyst Toolbox"
second view. The 3D world and the AI character exist to break the data down visually
without pretension. No arcade juice: no taunt spam, no letter grades, no point fanfares.

## Hard constraints (violating any of these fails the build)

1. **Frozen policy.** Never retrain, regenerate, or substitute the SB3/ONNX artifacts. The
   PPO observes the same 23 features. The map/game is a VIEW over the 5 service scalars;
   every on-map event (repairs, convoys) is DERIVED from real trajectory/allocation data.
2. **Loopback only, no online services.** Three.js and all new deps are npm-bundled into
   `frontend/dist`. No CDN, no fonts from network, no LLM narration (lines are procedural
   templates over run data — deterministic and provably true).
3. **Determinism preserved.** The game never steps the sim live. Each kick appends a forced
   shock and re-runs `POST /api/v1/simulations/compare` (sub-second), then playback resumes
   from the current day. Same seed + same scenario + same kick list = same canonical result.
4. **Additive backend change only.** `Scenario.forced_shock: ForcedShock | None` gains a
   sibling `forced_shocks: list[ForcedShock]` (or the singular is superseded — pick one,
   keep `extra="forbid"` strictness, bump the comparison schema version string used in
   result-id derivation, and keep old persisted results restorable — they are
   content-addressed and self-verifying, so they remain valid; do not migrate them).
5. **Do not destabilize accepted evidence.** No changes to simulator math, projector,
   planners, artifact guard, persistence identity rules beyond the schema-version bump.
6. **Trademark.** The style is a "toy-brick diorama." The word LEGO never appears in UI,
   README, code identifiers, or demo script. No minifigure-shaped people (the figure is
   trademarked). Hospital signage is a white "H", never a red cross (protected emblem).
7. **Both planners still run on every compare** (they already do). The baseline is never
   rendered as a second city. It appears only in the end-of-run debrief.

## The game

### Roles
- **Player = the disaster.** The only player powers: choose/aim/throw disasters, control
  the camera, control time (pause/speed). The player never allocates anything.
- **RELAY = the AI** (the frozen PPO). It allocates, repairs, defers, and narrates.

### Modes (picked on the start screen)
- **Sandbox** — unlimited disasters, unscored free play.
- **Stress Test** — finite disaster arsenal, scored, debrief at the end. (Professional
  rename of "budget mode"; do not call it chaos/points/arcade anything.)

### Difficulty presets (start screen; raw knobs live in the Toolbox only)
| Preset  | Maps to |
|---------|---------|
| Calm     | low `shock_probability` (~0.10), narrow severity band, generous `daily_budget` |
| Moderate | engine defaults (0.20, default band, 180/day) |
| Severe   | high `shock_probability` (~0.30-0.35), wide/high severity band, tight budget |

Stress Test initial arsenal (tune during build): 6 disasters per 14-day run; per-disaster
severity chosen by the player within the schema band (0.05–0.40). Difficulty does not
change the arsenal; it changes the world.

### Time
- Auto-play, ~2s per day, pause + speed toggle always visible.
- Grabbing a disaster enters slow motion while aiming; the kick lands at the **next day
  boundary** (shocks are per-day in the engine — frame it as "strikes overnight").
- Under the hood each kick = re-POST compare with the appended `ForcedShock`, then resume
  playback at the current day index. The re-run is invisible to the player.

### Disasters (the 5 engine shock types, presented plainly)
aftershock, supply, epidemic, utility, weather — each with its authored multi-service
impact footprint. Aiming at a district highlights which disaster hits it hardest, but the
kick is TYPED (engine truth), visualized as an impact wave whose per-district strength
matches the real impact vector.

### The economy (engine truth — build the HUD and silo visuals from this)
- **Income is exogenous and daily.** The scenario's `daily_budget` B arrives fresh each
  day. No savings, no carryover, no revenue actions — the PPO's action space is the
  five-way split and nothing else. The AI cannot "earn" more funding.
- **Disasters tax income, not just services.** The day's budget is
  `B_t = B x (1 - severity x budget_factor)`, and the factor differs by disaster type
  (utility 0.30 > supply/weather 0.25 > aftershock 0.15 > epidemic 0.10). Visualize at
  the silo: the day's inbound shipment is visibly smaller after a kick, and the HUD reads
  e.g. "Budget today: 152 / 180 (shock tax)".
- **Full spend is forced.** The projector's exact-sum constraint allocates all of B_t
  every day — the AI's skill is WHERE, never whether. Strategy comes from sqrt
  diminishing returns per service, dependency support (public services is the strongest
  supporter of every other service in the D matrix), and neglect strain (services under
  0.35 decay when underfunded).
- **The player's arsenal is a separate game-layer currency.** It never touches the AI's
  budget pool. The duel is asymmetric by design: you spend disasters, RELAY divides income.

### Stumble vs. fall (game-layer rules read off the trajectory; engine untouched)
- Collapse is measured on **city condition only — never on economics.** (There is no
  treasury to drain anyway — budget is a daily flow, fully allocated each day. A thriving
  city held together on a small or heavily shock-taxed daily budget is a success.)
- **Stumble:** any service dips below its critical floor (initial: 0.12) — visible wobble,
  smoke, urgent narration; recoverable.
- **Fall (run ends early, somber collapse screen):**
  - an essential service (healthcare or food) below floor for 4+ consecutive days, or
  - 2+ services below floor simultaneously for 2+ consecutive days (cascade failure).
- **District dark:** a service below floor 3+ consecutive days renders its district gray,
  still, desaturated until it recovers.
- Sandbox uses the same rules (falling is allowed and easier — you have unlimited kicks).

### End of run — the Debrief (no letter grades, plain language)
- What happened: disasters endured (ambient + player), worst moment, recovery count,
  final weighted wellbeing, resilience AUC, survived/fell and on what day.
- **Counterfactual comparison (the thesis, on stage):** the same run's baseline trajectory
  (already returned by the engine) evaluated under the same collapse rules:
  "A conventional rule-based planner ends this same run at 58% wellbeing with Housing
  dark from day 9." Label it exactly "conventional rule-based planner" — never "human
  leader." A quiet "inspect this run in the Analyst Toolbox" link.

## RELAY (the character)

- Matte-black sphere floating above the silo, in-scene, with animated horizontal waveform
  lines (per the user's sketch) that move when it speaks/thinks. Speech is an HTML overlay
  bubble anchored to the orb — text always crisp.
- Tone: overloaded mission-AI. Clipped, dramatic-but-true machine announcements generated
  from real step records:
  - Kick lands: "SHOCK DETECTED — HOUSING. SEVERITY 0.31."
  - Response: "REROUTING 38 UNITS TO HOUSING. STABILIZATION ESTIMATE: 3 DAYS."
  - Derived repair: "REBUILDING RESIDENTIAL BLOCK 4."
  - Deliberate deferral: "HOUSING REPAIRS DEFERRED — UNITS REROUTED TO HEALTHCARE."
    (Engine truth: the projector's exact-sum constraint spends the FULL daily budget every
    day. Narration may speak of rerouting/deprioritizing, never of "saving/banking" funds —
    no treasury exists.)
  - Stumble: "CRITICAL FLOOR BREACHED — FOOD. PRIORITY OVERRIDE."
- **No taunt system.** At most one dry closing line in the debrief when the player threw a
  lot and the city stood (e.g., "The city stands."). Nothing mid-run, nothing sassy.
- Every line must be derivable from trajectory data (no invented facts). Line templates are
  deterministic given the run.

## The world (visual spec)

- **Full 3D toy-brick diorama** on a studded baseplate. Drag to orbit/pitch ("holding the
  plate"), scroll to zoom; camera clamped above the plate, city never off-frame.
- **Daylight tabletop — FINAL** (exhibit-glow/dark-room direction was considered and
  rejected 2026-07-18; no window-glow mechanic, no emissive/bloom pipeline). Realistic-
  miniature palette: brick reds, asphalt gray, tree green, white trim; each district
  carries a subtle service accent color (trim/roofs) for legibility.
- **Brick-built read, faked smart:** studs instanced on roofs and baseplate, stepped blocky
  silhouettes, seam lines only near camera. Real individual bricks appear only where they
  matter: rubble piles and the rebuild animation (one reusable instanced brick pool;
  buildings restack course by course as RELAY funds recovery).
- **Procedural low-poly only.** No model files, no textures. Flat-shaded box construction.
  **~8 unique building archetypes, reused and recolored freely across districts** (user
  decision: repetition is fine — "office one and office two" share a mesh; adjust count
  slightly for balance). 30–40 buildings total across 5 districts (housing blocks, hospital
  campus, market/food, transit hub, civic center) + central resource silo + road tiles +
  vehicle prefabs + trees.
- **Damage = discrete per-building variants (user-specified):** each archetype has 4
  states — intact / slightly damaged / moderately damaged / rubble — plus a
  scaffold-and-crane overlay on buildings actively under reconstruction. A district's
  service level maps to the DISTRIBUTION of states across its buildings (e.g. 0.40 →
  mostly cracked, one in rubble). Aggregate wellbeing also reads through traffic density,
  smoke plumes, saturation, and the slim HUD strip. Collapsed district = all-rubble, gray,
  still, no vehicles.
- **Vehicles do all physical work:** convoys leave the silo carrying each day's real
  allocations (count/frequency proportional); repair trucks/cranes dispatch to buildings
  being rebuilt. This is the "AI can actually move units on the map" requirement — always
  derived from real allocation numbers. No people/figures in v1 (minifigure shapes are
  trademarked; vehicles, cranes, and traffic carry the life).
- **Institutional capacity is real, not invented:** the simulator's dependency matrix
  already makes every service's recovery gain depend on supporting services — including
  public services (the civic center). Surface this: when the civic district is badly
  damaged, RELAY narrates degraded recovery efficiency citywide ("CIVIC CAPACITY DEGRADED —
  RECOVERY EFFICIENCY REDUCED"), and the numbers genuinely reflect it. **Never fabricate
  legislative/policy actions (taxes, laws): the frozen PPO's action space is five budget
  allocations and nothing else.** Trucks = allocations (real); open offices = dependency
  support (real); both honest.
- Kicks: player drags a disaster card from a small tray onto the 3D city (raycast to
  district), severity set during slow-mo aim; impact = brief camera-shake-lite + wave.
  Restraint: effects communicate, never celebrate.
- Recommended stack: `three` + `@react-three/fiber` + `@react-three/drei` (OrbitControls),
  React 19 compatible versions, all bundled. WebGL-unavailable fallback: a clear message
  pointing to the Analyst Toolbox (which must remain fully functional without WebGL).

## Sound (subtle, procedural WebAudio, no asset files)

- Disaster impacts: low muffled rumble. RELAY speaking: soft synth blips. District dark:
  a quiet low drone. Nothing for scores/menus; no melodies; no arcade stingers.
- Starts after first user gesture (browser policy); prominent mute toggle; default on.

## Information architecture

- App boots into the **game start screen**: mode (Sandbox / Stress Test) + difficulty
  (Calm / Moderate / Severe) + Start. Five seconds to playing.
- **Analyst Toolbox** = the entire current Recovery Desk (scenario authoring with all raw
  knobs, trajectory chart, daily audit, saved results restore), reached by a quiet top-bar
  switch, restyled to match the new art but functionally intact. The Toolbox can launch a
  custom scenario INTO the game view.
- Every game compare is persisted as usual (idempotent, content-addressed). The final run
  of a game session is what the debrief links to. If intermediate kick re-runs clutter the
  saved list, group or tag them by session in the Toolbox list — do not change persistence
  semantics.

## Milestone plan (one-day sprint; cut line marked)

- **M1 — The diorama lives.** New game route; 3D city renders from a completed compare
  result; baseplate orbit; day playback with pause/speed; damage states track service
  levels; RELAY orb + narration bubble from trajectory data; Toolbox switch works (old UI
  intact, restyle later).
- **M2 — The kick.** Backend `forced_shocks` list (+ schema version bump + tests); disaster
  tray, slow-mo aim, drag-to-throw, re-run-and-resume; impact visuals; convoys.
- **M3 — Stakes.** Stumble/dark/fall rules + collapse screen; Sandbox vs Stress Test +
  arsenal; difficulty presets; start screen; Debrief with conventional-planner
  counterfactual.
- ——— cut line: M1–M3 must ship ———
- **M4 — Polish.** Sound; Toolbox restyle; camera/impact refinement; README + DEMO_SCRIPT
  updates; screenshot self-review loop per UI_DIRECTIVE_V2 (target ≥8/10, beat WEB2's 7.5
  without copying it); frontend tests updated for new routes while keeping Toolbox
  coverage; SHIP_REPORT per WORKFLOW_V2.

Per-milestone verification: `scripts/run.ps1`, exercise the flow in the browser, check
console clean, `scripts/verify.ps1 -Profile cpu` after backend changes.

## Out of scope / do not touch

- No retraining, no artifact regeneration, no simulator/projector/planner math changes.
- No second rendered city, no live baseline presence during a run.
- No LLM/TTS, no network calls, no external asset files.
- No spatial state added to the engine or observation space.
- Old saved results must continue to restore byte-identically.
