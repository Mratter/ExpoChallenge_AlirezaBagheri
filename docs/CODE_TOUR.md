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
| Review oracle-distillation evidence | `benchmarks/v4/oracle-distilled-ppo-study-200.md` | `internal/developmental_runs/v4/oracle-distilled-ppo-study-200.json`, `scripts/publish_oracle_distilled_ppo_evidence.py` |
| Review network-capacity evidence | `benchmarks/v4/network-capacity-study-200.md` | `internal/developmental_runs/v4/network-capacity-study-200.json`, `scripts/run_large_architecture_study.py`, `scripts/publish_network_capacity_evidence.py` |
| Review the stopped combined attempt | `benchmarks/v4/combined-distilled-large-study-200.incomplete.md` | `internal/developmental_runs/v4/combined-distilled-large-study-200.incomplete.json`, `scripts/publish_combined_distilled_large_failure_evidence.py` |
| Review family-reweighting evidence | `benchmarks/v4/moderate-family-study-200.md` | `internal/developmental_runs/v4/moderate-family-study-200.json`, `scripts/moderate_family_training.py`, `scripts/publish_moderate_family_evidence.py` |
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

`scripts/train_policy.py` is the full training pipeline. Its main flow is behavior cloning and DAgger, actor-frozen critic warm-up, PPO optimization, development evaluation at fixed milestones, and receipt writing. The instrumented PPO class and diagnostic helpers make optimizer movement, actor freezing, normalization state, and critic quality visible in the resulting evidence. **Demonstrated-achievable reference denominator = the 187 of 200 development cases solved by the privileged future-aware CEM run; its 13 search failures are not proofs of infeasibility.** The five registered 2M development endpoints averaged **171.4 / 200**, or **171.4 / 187 = 91.7%** of that achieved-count reference, with population standard deviation **1.62** and sample standard deviation **1.82**; no Wilson interval is reported for an optimizer-seed mean. They are indexed by `internal/developmental_runs/v4/training-study-200-summary.json` and presented with the matched ablations in `benchmarks/v4/training-study-200.md`. Selection evaluated 20 checkpoints; `internal/developmental_runs/v4/checkpoint-selection-200.json` chose seed `67017` at 1M with **178 / 200 development solves**, or **178 / 187 = 95.2%** (descriptive post-hoc Wilson 95% **[0.9111, 0.9745]**), and `internal/developmental_runs/v4/city_recovery_ppo.v4.parity.json` proves full SB3-to-ONNX parity for the bundled artifact. These are development statistics, distinct from the later held-out result. The gated sequence is recorded in the [training deployment plan](TRAINING_DEPLOYMENT_PLAN.md).

`scripts/evaluate.py` is the compact comparison runner. Read `build_cases`, `resolve_policy`, `rollout`, and `aggregate` in that order to see how a named 200-case development or final split and policy set become matched per-case rows and paired statistics. The shipped policy's final roster was evaluated exactly once after selection and artifact identity were frozen, using the claim-gated `scripts/publish_final_evaluation_v4.py` path. That owner-authorized run solved **163 / 200** and is retained in the [canonical final report](../benchmarks/v4/final-results-200.md) and machine [success receipt](../internal/evaluation_runs/v4/final-evaluation-200.success.json). Further learned-policy final reruns remain unauthorized.

`scripts/build_development_baselines.py` assembles the cheap-planner evidence over all 200 development cases. Its receipt-bound Markdown snapshot at `benchmarks/v4/development-baselines-200.md` predates the matched oracle rerun and remains byte-identical; its machine receipt at `internal/developmental_runs/v4/development-baselines-200.json` preserves the complete ordered rows, paired comparisons, invariants, and source identity. Current oracle interpretation lives in the separate matched report below.

`scripts/headroom.py` retains the original 40-case analysis and supplies the registered MPC and clairvoyant CEM mechanics. Its historical **37 / 40** result remains an original-subset achieved lower bound and has not been overwritten or pooled with the expanded evidence.

`scripts/run_oracle_study.py` applies the same registered population and iteration budget to all 200 development and 200 final cases and writes resumable per-case shards outside the repository. Both completed studies used eight spawned workers without fallback; if a future eight-or-more-worker pool fails, the runner resumes missing shards with four rather than serial execution. `scripts/publish_oracle_study.py` validates those raw receipts and produces the portable [200-case report](../benchmarks/v4/clairvoyant-oracle-200.md) plus complete [development](../internal/developmental_runs/v4/clairvoyant-oracle-200-dev.json) and [final](../internal/developmental_runs/v4/clairvoyant-oracle-200-final.json) receipts.

The post-release distillation path is deliberately separate. `scripts/run_training_oracle_trajectories.py` records privileged CEM actions only on the 192-case training roster; `scripts/train_oracle_bc_student.py` performs one fixed offline, zero-DAgger fit with a trajectory holdout; and `scripts/run_distilled_ppo_study.py` imports that public-state actor and frozen observation RMS into three fresh seeded critics before the adopted warm-up and PPO flow. `scripts/publish_oracle_distilled_ppo_evidence.py` then validates the external protocol, upstream receipts, three trainer receipts, nine selectable bundles, and every development row before producing the portable [machine receipt](../internal/developmental_runs/v4/oracle-distilled-ppo-study-200.json) and [study report](../benchmarks/v4/oracle-distilled-ppo-study-200.md). The 2M endpoints were **178, 174, and 170 / 200** (mean **174.0**, population SD **3.27**, sample SD **4.0**); the best tied 178 rather than meeting the preregistered 183 threshold, so the challenger was not promoted. This evidence is development-only and leaves the shipped artifact and final result untouched.

The post-release capacity path is also isolated from deployment. `scripts/run_large_architecture_study.py` changes only the actor/critic hidden layers to `[768, 512, 256]` and pairs `7.5e-5` with `3e-5` on seeds `37017`, `47017`, and `57017`; the runtime interface remains 73 observations to 22 actions. `scripts/publish_network_capacity_evidence.py` rebuilds the external summary, verifies six training receipts and all 18 selectable bundles, and retains every DEV row, family aggregate, and diagnostic curve in the portable [machine receipt](../internal/developmental_runs/v4/network-capacity-study-200.json) and [study report](../benchmarks/v4/network-capacity-study-200.md). The `3e-5` endpoints were **178, 176, and 175 / 200** (mean **176.33**, population SD **1.25**, sample SD **1.53**), exactly **+6, +5, and +4** against the same-seed incumbent endpoints. The best tied 178 and failed the preregistered 183 gate, so it was not promoted. All three low-LR curves still climbed from 1M to 2M by **+5, +3, and +1**; without the optional incumbent-size `3e-5` arm, this is a positive large-plus-low-LR signal rather than a capacity-only conclusion. The scope is these two LRs, three seeds, and 2M budget; no final case was used and the shipped artifact is untouched.

The combined large-network plus oracle-distillation attempt is retained separately as incomplete evidence. `scripts/publish_combined_distilled_large_failure_evidence.py` validates the frozen 37-file external inventory, both console logs, protocols, offline-fit chain, all 14 retained DEV evaluations, eight PPO bundles, and the stopped worker receipt before producing the portable [incomplete receipt](../internal/developmental_runs/v4/combined-distilled-large-study-200.incomplete.json) and [report](../benchmarks/v4/combined-distilled-large-study-200.incomplete.md). Seeds `37017` and `47017` completed curves of **153, 153, 158, 162, 173, 170 / 200** and **153, 153, 160, 167, 170, 174 / 200**. Seed `57017` stopped intentionally after exactly 50k actor-frozen critic transitions because its final explained variance, **0.4789480567**, did not satisfy the strict `> 0.5` gate; active PPO stayed at zero and the actor and RMS were unchanged. The observed 174/200 best is not promotable, and no two-seed mean, SD, or promotion decision is reported. The comparison is nonfactorial because initialization, normalization, and one historical warm-up budget differ. The attempt was not retried or resumed, used no final case, and left the shipped artifact untouched.

The post-release family-sampling path is likewise isolated. `scripts/moderate_family_training.py` measures the shipped policy only on the 192-case TRAIN roster, assigns 2× weight to its weakest two families, and applies the resulting deterministic 256-occurrence cycle throughout BC, DAgger, warm-up, and PPO. `scripts/publish_moderate_family_evidence.py` validates that TRAIN-only choice, all three training receipts, all 12 durable bundles, the nine selectable candidates, and the matched incumbent and shipped-checkpoint rows before producing the portable [machine receipt](../internal/developmental_runs/v4/moderate-family-study-200.json) and [study report](../benchmarks/v4/moderate-family-study-200.md). The 2M endpoints were **175, 170, and 172 / 200** (mean **172.33**, population SD **2.05**, sample SD **2.52**); the same-seed endpoint mean moved by **+1.0 case**, while the registered best was the 1M checkpoint at **176 / 200** and failed the preregistered 183 gate. The cycle also increased BC/DAgger observations from **23,040 to 30,720 (+33.3%)**, so this treatment combines family reweighting with extra imitation exposure. Matched cases show a small redistribution rather than robust targeted-family improvement. This result is limited to one 2:1 sampler, three seeds, and 2M budget; no final case was used and the shipped artifact is untouched.

The oracle achieved **187 / 200** development solves and **182 / 200** final solves. The development denominator definition above therefore gives the selected policy an aggregate **178 / 187 = 95.2%** achieved-count ratio (descriptive post-hoc Wilson 95% **[0.9111, 0.9745]**). Its matched development comparison records **177 both solved, 1 policy-only, 10 oracle-only, and 12 neither**: casewise coverage is **177 / 187 = 94.7%**, the two methods jointly demonstrate solutions on **188 / 200** cases, and the 10 oracle-only rows are the remaining provable headroom. The oracle sees the complete future shock tape, is not a submission baseline or model-selection input, and provides an anytime achieved lower bound rather than a proof of optimality. All oracle rows have zero hard violations and exactly `0.0` conservation residual. Normal runtime policies remain causal public-state planners.

On the final roster, the achieved-count framing is explicit. **Demonstrated-achievable reference denominator = the 182 of 200 final cases solved by the privileged future-aware CEM run; its 18 search failures are not proofs of infeasibility.** The causal shipped policy solved **163 / 182 = 89.6%** of that achieved-count reference (descriptive post-hoc Wilson 95% **[0.8427, 0.9321]**), alongside its raw **163 / 200**, and finished **16 cases ahead** of the tuned constant rule at 147 / 200. Their exact pairing is **162 both, 1 policy-only, 20 oracle-only, and 17 neither**: casewise coverage is **162 / 182 = 89.0%**, and the two methods jointly demonstrate solutions on **183 / 200** cases. The distinction matters because the finite anytime oracle has one policy-only case and is not a proven ceiling. The raw receipt-level Wilson interval is **[0.7554293724, 0.862698072]**; it treats cases as Bernoulli units and does not model dependence within the five fixed 40-case scenario families, so the final report also exposes family-level results.

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
- `tests/test_train_policy.py`, `test_evaluate.py`, `test_build_development_baselines.py`, `test_development_evidence.py`, `test_headroom.py`, `test_run_oracle_study.py`, `test_publish_oracle_study.py`, `test_oracle_distilled_ppo_evidence.py`, `test_network_capacity_evidence.py`, `test_combined_distilled_large_failure_evidence.py`, `test_moderate_family_training.py`, `test_moderate_family_evidence.py`, and `test_publish_final_evaluation_v4.py` cover the scientific tools and their current, stopped-attempt, and historical evidence.
- `tests/test_api.py`, `test_recovery_analysis.py`, `test_recovery_exports.py`, and `test_shared_evidence.py` cover the runtime shell and durable results.
- `tests/test_frontend_contract_generation.py` proves that the generated TypeScript contract matches the canonical Python values.
- Tests beside the frontend source (`*.test.ts`) cover response parsing, generated contracts, view-model calculations, and decision-support behavior.

For a change that crosses layers, test from the center outward: domain invariant, Python consumer, generated contract if applicable, TypeScript parser, and finally the rendered workflow.
