# Code Tour

The repository has one architectural center: deterministic city mechanics. HTTP, policy inference, training, evaluation, evidence publication, and presentation all consume that domain rather than maintaining competing implementations.

Use [Development](DEVELOPMENT.md) for commands, [Evidence and Results](EVIDENCE.md) for measured claims, [Troubleshooting](TROUBLESHOOTING.md) for operational failures, and the [Training Deployment Plan](TRAINING_DEPLOYMENT_PLAN.md) for publication gates.

## Choose an entry point

| Goal | Start here | Follow with |
| --- | --- | --- |
| Trace a runtime comparison | `backend/app/main.py` | `model/policy.py`, `backend/app/city/environment.py`, `backend/app/persistence.py` |
| Understand city mechanics | `backend/app/city/physics.py` | `scenarios.py`, `outcome.py`, `environment.py` |
| Understand the learned policy | `model/policy.py` | `artifacts/city_recovery_ppo.v4.manifest.json`, `scripts/train_policy.py` |
| Understand training | `scripts/train_policy.py` | `backend/app/city/planners.py`, `tests/test_train_policy.py` |
| Reproduce a development comparison | `scripts/evaluate.py` | `backend/app/city/scenarios.py`, `backend/app/city/environment.py` |
| Review final evidence | `benchmarks/v4/final-results-200.md` | `internal/evaluation_runs/v4/final-evaluation-200.success.json` |
| Review privileged search | `benchmarks/v4/clairvoyant-oracle-200.md` | `scripts/run_oracle_study.py`, `scripts/headroom.py` |
| Review the Hurricane Maria reconstruction | `benchmarks/v4/hurricane-maria-retrospective.md` | `internal/retrospectives/hurricane-maria-30d.json` |
| Work on the Toolbox | `frontend/src/App.tsx` | `frontend/src/api.ts`, `frontend/src/DecisionAnalysis.tsx` |
| Work on the 3D city | `frontend/src/game/CityGame.tsx` | `frontend/src/game/CityScene.tsx` |

## Dependency shape

```text
models + shared evidence        physics
          |                  /     |     \
          +----> scenarios       outcome  planners + optimizer
                    \              |       /
                     +------> environment <------ model/policy
                                   |
                     persistence + analysis + exports
                                   |
                                 main.py
                                   |
                         generated TypeScript contract
                                   |
                          API clients -> browser
```

The direction is intentional:

- `backend/app/city` owns domain behavior and does not depend on FastAPI, React, persistence, or training scripts.
- `backend/app/main.py` is the application composition root; it does not redefine physics or scoring.
- `scripts` import production domain modules. Runtime modules never import training or evaluation scripts.
- `scripts/generate_frontend_contract.py` projects canonical Python values into TypeScript; the browser does not hand-maintain another contract.
- Tests cover each layer at its public seam and then cover assembled request, replay, evidence, and frontend flows.

## Read the system from the center outward

### 1. Public schemas and evidence primitives

Start with `backend/app/models.py`. Its strict Pydantic models define scenario, forced-shock, and comparison request shapes shared by runtime construction and frontend contract generation.

Then read `backend/app/shared_evidence.py`. It owns canonical JSON bytes and hashes, strict JSON loading, Wilson intervals, split contracts, and durable writes. Persisted comparisons and scientific receipts use the same primitives.

### 2. City mechanics

Read these modules in order:

1. `backend/app/city/physics.py` — service/hazard order, allocation projection, depot and road mechanics, transfers, repair, shocks, and conservation measurements.
2. `backend/app/city/scenarios.py` — disjoint 192-case training, 200-case development, and 200-case final rosters plus deterministic shock-tape generation.
3. `backend/app/city/outcome.py` — the frozen six-check Solved conjunction and trajectory summaries.
4. `backend/app/city/planners.py` — causal reactive, preparedness-teacher, and tuned-rule proposals plus shared proposal conversion.
5. `backend/app/city/optimizer.py` — OR-Tools allocation proposals used by diagnostic planning and headroom work.
6. `backend/app/city/environment.py` — the 73-field observation, 22-field action, proposal decoding, feasibility projection, daily transition, rollout, and paired comparison composition.

`CityRecoveryEnv` composes the pieces; it does not duplicate them. `CyclingScenarioEnv` provides training rotation over canonical cases without creating another simulator.

### 3. One policy through the runtime

`model/policy.py` names `artifacts/city_recovery_ppo.v4.onnx` as the bundled default. It validates the artifact identity when requested, the raw `observation[batch,73] -> action[batch,22]` contract, CPU provider, finite output, and action bounds.

PowerShell launchers use `scripts/runtime_policy.ps1` for the same fail-closed precedence: explicit `-PolicyPath`, then `INNOVERSE_POLICY_PATH`, then the bundle. `scripts/preflight_check.py` performs a complete deterministic smoke comparison before launch.

The application path is:

1. `backend/app/main.py` validates a request and loads the selected policy.
2. `backend/app/city/scenarios.py` constructs one deterministic shock tape.
3. `backend/app/city/environment.py` runs candidate and heuristic independently on that shared tape.
4. `backend/app/city/outcome.py` evaluates each planner's six checks.
5. `backend/app/persistence.py` stores the canonical result by content identity.
6. `backend/app/recovery_analysis.py` and `recovery_exports.py` replay or render only verified persisted results.

The 200-case served-path gate exercised this exact FastAPI `POST -> persist -> GET` route and reproduced all 178 accepted development outcomes for the bundled policy.

### 4. Training and publication

`scripts/train_policy.py` implements:

```text
BC/DAgger -> actor-frozen critic warm-up -> PPO -> development milestones -> receipt
```

The trainer records optimizer movement, critic quality, actor hashes, normalization state, and complete checkpoint bundles. Development-only selection chose seed `67017` at 1M with 178/200 solves. `scripts/export_policy.py` baked its frozen observation normalization into a self-contained opset-17 graph; full SB3-to-ONNX parity covered 6,000 action vectors and 132,000 action elements.

The deployed actor's single owner-authorized final run solved raw **163 / 200**. Privileged future-aware CEM solved 182/200 final cases under its fixed search budget, establishing an oracle-solved reference rather than a ceiling. The exact pairing—162 both, 1 policy-only, 20 oracle-only, 17 neither—shows the solved sets are not nested; their union is 183/200. See [Evidence and Results](EVIDENCE.md) for interpretation and receipts.

Scientific tools remain clients of the production domain:

- `scripts/evaluate.py` builds named split comparisons from canonical cases.
- `scripts/build_development_baselines.py` rebuilds cheap-planner development evidence.
- `scripts/run_oracle_study.py` and `publish_oracle_study.py` run and publish the privileged CEM diagnostic.
- `scripts/publish_final_evaluation_v4.py` implements the narrow one-run claim/success/failure lifecycle for the frozen final artifact.
- post-release study runners and publishers retain development-only distillation, capacity, family-sampling, and stopped-attempt evidence without changing deployment.

### 5. Browser flow

Start with `frontend/src/generated/backendContract.ts`. It is generated from canonical Python values and anchors service order, hazard impacts, observation/action order, request bounds, and the default scenario.

Then follow:

- `frontend/src/api.ts` for metadata, comparisons, and saved-run validation;
- `frontend/src/analysisApi.ts` for explanation/counterfactual validation and recovery-plan URLs;
- `frontend/src/App.tsx` for routing, scenario editing, paired runs, trajectory, audit, dispatch, and decision-log views;
- `frontend/src/DecisionAnalysis.tsx` for local action sensitivity, one-day replay, exports, and preparedness evidence;
- `frontend/src/game/CityGame.tsx` and `CityScene.tsx` for 3D playback of the same backend comparison.

The frontend contains zero uses of `any` or `@ts-ignore` across roughly 6,000 TypeScript lines. Its accessibility surface includes 30 `aria-label` attributes, 29 `role=` assignments, an `aria-live` region, and an `aria-modal` dialog.

## Tests as executable maps

- `tests/test_city_*.py` covers physics, scenarios, outcome, planners, optimizer, and complete environment transitions.
- `tests/test_policy.py`, `test_api.py`, `test_runtime_policy_resolution.ps1`, and `test_served_policy_replay.py` cover the bundled artifact and runtime path.
- `tests/test_train_policy.py`, `test_evaluate.py`, and evidence-specific tests cover scientific tools and retained receipts.
- `tests/test_recovery_analysis.py`, `test_recovery_exports.py`, and `test_shared_evidence.py` cover durable runtime evidence.
- `tests/test_frontend_contract_generation.py` binds canonical Python values to generated TypeScript.
- `frontend/src/*.test.ts` covers parsing, view models, generated contracts, and decision support.

For a cross-layer change, test from the domain outward: invariant, Python consumer, generated contract, TypeScript parser, rendered workflow.

## Reconstruction boundary

The Hurricane Maria material is a **project reconstruction from official records**. Source anchors and transformations are documented in its report, while simulated service, shock, allocation, and recovery lines remain generated outputs—not observations or causal estimates. The retrospective consumes the same public scenario/runtime concepts but is not a training split, validation cohort, final benchmark, or new simulator implementation.
