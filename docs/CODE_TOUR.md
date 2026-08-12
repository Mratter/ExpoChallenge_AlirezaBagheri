# Code tour

This repository is organized around one dependency rule: city mechanics are the center, while HTTP, training, evaluation, and presentation are consumers of those mechanics. Start with the path that matches the work you plan to do, then follow the guided reading order below when you need the complete picture.

## Choose an entry point

| Goal | Start here | Follow with |
| --- | --- | --- |
| Trace a runtime request | `backend/app/main.py` | `backend/app/models.py`, `model/policy.py`, `backend/app/city/environment.py`, `backend/app/persistence.py` |
| Understand or change training | `scripts/train_policy.py` | `backend/app/city/environment.py`, `backend/app/city/planners.py`, `backend/app/city/outcome.py` |
| Reproduce a policy comparison | `scripts/evaluate.py` | `model/policy.py`, `backend/app/city/scenarios.py`, `backend/app/city/environment.py` |
| Rebuild current development evidence | `scripts/build_development_baselines.py` | `scripts/evaluate.py`, `benchmarks/v4/development-baselines-200.md`, `internal/developmental_runs/v4/development-baselines-200.json` |
| Study achievable headroom | `scripts/run_oracle_study.py` | `scripts/headroom.py`, `backend/app/city/planners.py`, `backend/app/city/optimizer.py`, `backend/app/city/outcome.py` |
| Review the held-out result | `benchmarks/v4/final-results-200.md` | `internal/evaluation_runs/v4/final-evaluation-200.success.json`, `scripts/publish_final_evaluation_v4.py` |
| Compare final scenario families | `benchmarks/v4/final-family-analysis-200.md` | final success receipt, matched oracle receipt, `backend/app/city/scenarios.py` |
| Work on the browser application | `frontend/src/App.tsx` | `frontend/src/generated/backendContract.ts`, `frontend/src/api.ts`, `frontend/src/analysisApi.ts`, `frontend/src/DecisionAnalysis.tsx` |

## The dependency shape

The principal flow is:

```text
models + shared_evidence       physics
          |                 /    |     \
          +----> scenarios      outcome  planners + optimizer
                    \             |       /
                     +------> environment <------ model/policy
                                  |
                    persistence + analysis + exports
                                  |
                                main.py
                                  |
                         generated TypeScript contract
                                  |
                       API clients -> App -> 3D city
```

The diagram is intentionally one-directional:

- `backend/app/city` owns deterministic domain behavior and does not depend on FastAPI, persistence, React, or training scripts.
- `backend/app/main.py` is the composition root. It wires policy loading, simulation, persistence, analysis, exports, and static frontend delivery without moving domain rules into the route layer.
- `scripts` import production domain and policy modules. Production runtime modules never import training or evaluation scripts.
- `scripts/generate_frontend_contract.py` projects canonical Python values into TypeScript. The browser consumes the generated file; it does not maintain a competing copy of the simulator contract.
- Tests exercise each layer at its public seam, then cover the assembled request, replay, evidence, and frontend flows.

## Guided reading order

### 1. Establish the public schemas and evidence primitives

Read `backend/app/models.py` first for the validated scenario, forced-shock, and comparison request shapes. These Pydantic models define what may enter the runtime and are also inputs to scenario construction and frontend contract generation.

Then read `backend/app/shared_evidence.py`. It provides canonical JSON bytes and hashes, strict JSON loading, Wilson intervals, split contracts, function-source hashes, and durable parent-directory synchronization. Outcome identities, persisted results, development receipts, and generated evidence all use these shared primitives.

### 2. Read the city mechanics from the bottom up

Begin with `backend/app/city/physics.py`. It is the numerical foundation: service and hazard order, allocation projection, action proposals, depot damage, throughput, transfers, capped landing, and conservation measurements. The remaining city modules use these functions instead of reimplementing allocation or logistics rules.

Continue with `backend/app/city/scenarios.py`. It defines the disjoint canonical rosters—192 training cases (6 families × 32 seeds), 200 development cases (5 × 40), and 200 held-out final cases (5 × 40)—and turns a validated scenario plus seed into a deterministic disaster tape. This is where split membership and shock realization become concrete; exactly one learned-policy evaluation of the final roster is retained.

Read `backend/app/city/outcome.py` beside it. Outcome calculation is deliberately separate from state transition code. `absolute_outcome` evaluates the canonical solved conjunction; `summarize_trajectory` derives the aggregate evidence used by the API, evaluation tools, and analysis views.

Next inspect the proposal producers:

- `backend/app/city/planners.py` contains causal public-state policies: the reactive heuristic, preparedness teacher, tuned rule, and the shared weight-to-logit conversion.
- `backend/app/city/optimizer.py` contains the OR-Tools allocation proposal used by headroom analysis. It consumes a public allocation context and the same physics constants as every other planner.

Finish the domain pass in `backend/app/city/environment.py`. `CityRecoveryEnv` composes the scenario tape, 73-field observation contract, 22-field action contract, proposal decoding, feasibility projection, daily transition, and outcome summary. The rollout and comparison helpers near the bottom are the main bridge used by the runtime and scripts. `CyclingScenarioEnv` provides the training-facing rotation over canonical cases without creating a second simulator.

### 3. Follow one policy through the runtime

Read `model/policy.py` for the deployment boundary. It names `artifacts/city_recovery_ppo.v4.onnx` as the bundled zero-configuration default, validates the selected artifact's identity and raw `observation[batch,73]` to `action[batch,22]` tensor contract, creates an ONNX Runtime session, and validates every predicted action before returning it. `scripts/runtime_policy.ps1` gives setup, preflight, and launchers the same fail-closed precedence: explicit `-PolicyPath`, then `INNOVERSE_POLICY_PATH`, then the bundle.

Then move through the application modules in this order:

1. `backend/app/persistence.py` assigns canonical result identities and stores complete comparison results atomically.
2. `backend/app/recovery_analysis.py` verifies a persisted replay before producing per-day local action sensitivity or a one-day allocation counterfactual. Counterfactual replay changes only the selected intervention and follows the same policy thereafter.
3. `backend/app/recovery_exports.py` renders persisted candidate or baseline trajectories as deterministic CSV and PDF recovery plans.
4. `backend/app/main.py` wires these capabilities into health, metadata, comparison, saved-run, explanation, counterfactual, and export endpoints. It also serves the built frontend.

A comparison request therefore follows a compact path: validate the request, load the selected policy, generate one disaster tape, run independent candidate and baseline environments, summarize both outcomes, persist the canonical result, and return that stored structure to the browser.

### 4. Read the scientific tools as clients of the domain

`scripts/train_policy.py` is the full training pipeline. Its main flow is behavior cloning and DAgger, actor-frozen critic warm-up, PPO optimization, development evaluation at fixed milestones, and receipt writing. The instrumented PPO class and diagnostic helpers make optimizer movement, actor freezing, normalization state, and critic quality visible in the resulting evidence. The five registered 2M development endpoints averaged **171.4 / 200**, with population standard deviation **1.62** and sample standard deviation **1.82**; they are indexed by `internal/developmental_runs/v4/training-study-200-summary.json` and presented with the matched ablations in `benchmarks/v4/training-study-200.md`. Selection evaluated 20 checkpoints; `internal/developmental_runs/v4/checkpoint-selection-200.json` chose seed `67017` at 1M with **178 / 200 development solves**, and `internal/developmental_runs/v4/city_recovery_ppo.v4.parity.json` proves full SB3-to-ONNX parity for the bundled artifact. These are development statistics, distinct from the later held-out result. The gated sequence is recorded in the [training deployment plan](TRAINING_DEPLOYMENT_PLAN.md).

`scripts/evaluate.py` is the compact comparison runner. Read `build_cases`, `resolve_policy`, `rollout`, and `aggregate` in that order to see how a named 200-case development or final split and policy set become matched per-case rows and paired statistics. The shipped policy's final roster was evaluated exactly once after selection and artifact identity were frozen, using the claim-gated `scripts/publish_final_evaluation_v4.py` path. That owner-authorized run solved **163 / 200** and is retained in the [canonical final report](../benchmarks/v4/final-results-200.md) and machine [success receipt](../internal/evaluation_runs/v4/final-evaluation-200.success.json). Further learned-policy final reruns remain unauthorized.

`scripts/build_development_baselines.py` assembles the cheap-planner evidence over all 200 development cases. Its receipt-bound Markdown snapshot at `benchmarks/v4/development-baselines-200.md` predates the matched oracle rerun and remains byte-identical; its machine receipt at `internal/developmental_runs/v4/development-baselines-200.json` preserves the complete ordered rows, paired comparisons, invariants, and source identity. Current oracle interpretation lives in the separate matched report below.

`scripts/headroom.py` retains the original 40-case analysis and supplies the registered MPC and clairvoyant CEM mechanics. Its historical **37 / 40** result remains an original-subset achieved lower bound and has not been overwritten or pooled with the expanded evidence.

`scripts/run_oracle_study.py` applies the same registered population and iteration budget to all 200 development and 200 final cases and writes resumable per-case shards outside the repository. Both completed studies used eight spawned workers without fallback; if a future eight-or-more-worker pool fails, the runner resumes missing shards with four rather than serial execution. `scripts/publish_oracle_study.py` validates those raw receipts and produces the portable [200-case report](../benchmarks/v4/clairvoyant-oracle-200.md) plus complete [development](../internal/developmental_runs/v4/clairvoyant-oracle-200-dev.json) and [final](../internal/developmental_runs/v4/clairvoyant-oracle-200-final.json) receipts.

The oracle achieved **187 / 200** development solves and **182 / 200** final solves. Its matched development comparison with the shipped policy records **177 both solved, 1 policy-only, 10 oracle-only, and 12 neither**; the 10 oracle-only rows are the remaining provable headroom. The oracle sees the complete future shock tape, is not a submission baseline or model-selection input, and provides an anytime achieved lower bound rather than a proof of optimality. All oracle rows have zero hard violations and exactly `0.0` conservation residual. Normal runtime policies remain causal public-state planners.

On the final roster, the causal shipped policy solved **163 / 200**, **16 cases ahead** of the tuned constant rule at 147, while the privileged oracle solved **182 / 200**. Their exact pairing is **162 both, 1 policy-only, 20 oracle-only, and 17 neither**. The aggregate policy/oracle solved-count ratio is **89.6%**; casewise policy coverage of oracle-achieved cases is **89.0%**. The distinction matters because the finite anytime oracle has one policy-only case and is not a proven ceiling. The receipt-level Wilson interval for the shipped policy is **[0.7554293724, 0.862698072]**; it treats cases as Bernoulli units and does not model dependence within the five fixed 40-case scenario families, so the final report also exposes family-level results.

The [final family-analysis supplement](../benchmarks/v4/final-family-analysis-200.md) joins the retained shipped-policy and tuned-rule rows with the deterministic teacher breakdown and registered family construction. All three planners have their lowest solve count on aftershock corridor (**26 / 40**, **20 / 40**, and **16 / 40**). That family has the lowest budget center, 136, alongside the joint-highest base shock probability, 0.30, and severity ceiling, 0.36. The learned policy's margins are widest there—**+6** over the tuned rule and **+10** over the teacher—whereas the tuned rule ties it at **38 / 40** on food access. This is descriptive evidence across five designed families, not a causal parameter ablation.

`scripts/generate_frontend_contract.py` is the cross-language boundary. It imports canonical service, hazard, observation, action, request, and outcome values and renders `frontend/src/generated/backendContract.ts`. Run it with `--check` after changing a value it projects.

### 5. Follow the browser data flow

Start with `frontend/src/generated/backendContract.ts`. It is generated, not hand-maintained, and anchors service order, hazard impacts, observation and action order, request limits, and the default scenario.

Then follow the two API paths:

- `frontend/src/api.ts` validates metadata, comparison, and saved-run responses before exposing them to the application.
- `frontend/src/analysisApi.ts` validates explanation and counterfactual identities, dimensions, ordering, hashes, and normalized allocation shares, and constructs recovery-plan export URLs.

`frontend/src/App.tsx` is the browser composition root. It owns the scenario editor, planner comparison, trajectory, audit, dispatch, decision-log views, saved runs, and the route boundary between the Analyst Toolbox and the city.

`frontend/src/DecisionAnalysis.tsx` implements the decision-support workflow: per-day policy sensitivity across all 73 observation channels, a replayed one-day allocation counterfactual, deterministic recovery-plan downloads, and preparedness-versus-shock-absorption evidence.

For the 3D path, begin with `frontend/src/game/CityGame.tsx`, which owns setup, playback, day selection, disaster injection, audio, quality monitoring, and the debrief transition. Continue into `frontend/src/game/CityScene.tsx`, which assembles districts, infrastructure, depots, vehicles, disaster effects, recovery states, camera behavior, and rendering quality from the same comparison trajectory shown in the Toolbox. The supporting modules under `frontend/src/game` each own one visual or interaction concern.

### 6. Use tests as executable maps

The Python suite follows the production boundaries:

- `tests/test_city_physics.py`, `test_city_scenarios.py`, `test_city_outcome.py`, `test_city_planners.py`, `test_city_optimizer.py`, and `test_city_environment.py` cover the domain from primitives through complete transitions.
- `tests/test_policy.py` covers the bundled artifact identity plus ONNX loading and inference contracts; `tests/test_runtime_policy_resolution.ps1` covers launcher precedence and fail-closed resolution.
- `tests/test_train_policy.py`, `test_evaluate.py`, `test_build_development_baselines.py`, `test_development_evidence.py`, `test_headroom.py`, `test_run_oracle_study.py`, `test_publish_oracle_study.py`, and `test_publish_final_evaluation_v4.py` cover the scientific tools and their current plus historical evidence.
- `tests/test_api.py`, `test_recovery_analysis.py`, `test_recovery_exports.py`, and `test_shared_evidence.py` cover the runtime shell and durable results.
- `tests/test_frontend_contract_generation.py` proves that the generated TypeScript contract matches the canonical Python values.
- Tests beside the frontend source (`*.test.ts`) cover response parsing, generated contracts, view-model calculations, and decision-support behavior.

For a change that crosses layers, test from the center outward: domain invariant, Python consumer, generated contract if applicable, TypeScript parser, and finally the rendered workflow.
