# Tester Feedback

- Project: AI17 - Civic Relay / Autonomous City Recovery Planner
- Tester canonical ID: `/root/ai17_fc_tester`
- Named gate: Feature Complete only
- Frozen candidate: `3c16f0359cca93e494cc65f0a8850ef6e9c744da`
- Accepted Evidence/Vertical Slice ledger: `db679c895aebb42027a4c6f0590b466dd0657e9b` (closed and unmodified)
- Test environment: Windows 11 x64, PowerShell 7.6.3; CPU profile
- Recommendation: PASS

## Group 1 - Identity And Accepted Lineage Preservation

- Result: PASS
- Commands:
  - `git status --short --branch`
  - `git rev-parse HEAD; git rev-parse main; git rev-parse origin/main; git branch --show-current`
  - `git remote -v`
  - `gh auth status`
  - `gh repo view Mratter/innoverse-ai17-city-recovery --json nameWithOwner,visibility,isPrivate,defaultBranchRef,url`
  - `git log -8 --date=iso-strict --format='%H%x09%ad%x09%an%x09%s'`
  - `git diff --name-status eb9c1dfa8ab52c03a2ebf97f31a43ab28849715c..3c16f0359cca93e494cc65f0a8850ef6e9c744da`
- Evidence: worktree began clean on branch `main`; `HEAD`, local `main`, and `origin/main` all resolved exactly to the frozen candidate. Origin is `https://github.com/Mratter/innoverse-ai17-city-recovery.git`; authenticated GitHub metadata reports `Mratter/innoverse-ai17-city-recovery`, default branch `main`, visibility `PRIVATE`, `isPrivate=true`. The prior accepted project baseline `eb9c1dfa8ab52c03a2ebf97f31a43ab28849715c` remains in history; the candidate is one implementation commit above it. The program-ledger SHA belongs to the separate program repository and was not modified.
- Cleanup: no process started and no artifact changed. Only this tester-owned feedback file is now modified.
- Remaining Feature Complete work: artifact/SB3/ONNX provenance and parity; simulator/scenario/baseline/constraint workflows; held-out evaluation; persistence/restart/metadata/jobs/failures; focused tests/build and bounded API/browser smoke.

## Group 2 - SB3/ONNX Provenance, Load, Parity, And No Fallback

- Result: PASS
- Commands:
  - `uv --version; node --version; npm --version; ./.venv/Scripts/python.exe --version`
  - Python imports reporting Stable-Baselines3 `2.7.0`, ONNX Runtime `1.23.0`, OR-Tools `9.14.6206`, Gymnasium `1.2.1`, PyTorch `2.8.0+cpu`, and ONNX `1.19.0`.
  - `Get-FileHash` and `Get-Item` over the SB3 checkpoint, ONNX, metadata, legacy candidate, parity report, protocol, and evaluation report.
  - Python `zipfile` member inspection of `artifacts/city_recovery_ppo.v1.zip`.
  - Independent inline Python probe loading `PPO.load(...)`, validating the ONNX model/opset/session contract, regenerating `scripts.train_policy.parity_cases`, independently comparing PyTorch and ONNX actions/proposals/projected allocations, inspecting production session options, and overriding bundle paths with missing/corrupt members.
- Evidence: exact SHA-256 values matched the frozen declarations: SB3 `f270bc720e7d2866d293feab27692d3ac9542d064d275b13c33f4d960dad4e33` (80,181 bytes), ONNX `983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8` (10,469 bytes), metadata `becc2eed1e552e9a503c3210d2ebae18eeccc593c9a7d716fae11e1e69b1c62e`, parity `20d87aafc638f3c6e7942a1578eea0710e0cd083c5a2054063f1813a76916a82`, and legacy linear candidate `23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3`. The SB3 archive contains real policy, optimizer, variables, version, and system-info members; `PPO.load` returned a `PPO` with observation/action shapes `23/5`. The ONNX checker passed opset 17. Both the independent and production sessions reported only `CPUExecutionProvider`, sequential execution, and `1/1` intra/inter-op threads.
- Parity: regenerated 32 cases; maximum action error `1.7881393432617188e-07` <= `1e-05`; maximum pre-projector proposal error `7.856886107049377e-06`; maximum post-projector allocation error `7.850000002918023e-06` <= `1e-04`. These exactly reproduce the frozen report.
- Fail closed: missing ONNX and checkpoint overrides raised `required policy artifact is missing or unreadable`; a corrupt ONNX override raised artifact byte-count drift. No call returned a policy bundle or legacy inference result.
- Non-blocking backlog: the SB3 checkpoint records `num_timesteps=30208` because PPO completes its 256-step rollout, while metadata/status say `30000` requested training steps. This does not invalidate the real PPO identity or parity, but provenance wording should distinguish requested from actual executed steps.
- Cleanup: the corrupt probe lived only in an automatically removed temporary directory; committed artifacts remained byte-identical.
- Remaining Feature Complete work: simulator/scenario/baseline/constraint workflows; held-out evaluation; persistence/restart/metadata/jobs/failures; focused tests/build and bounded API/browser smoke.

## Group 3 - Simulator, Authored Scenario, Identical Shocks, Baseline, And Constraints

- Result: PASS
- Command: independent inline Python functional probe using `CityRecoveryEnv`, Gymnasium `check_env`, a new user-authored 17-day scenario at seed `3981762211`, `compare`, direct constraint measurements, changed-seed sensitivity, bounded validation failures, and a 250-case feasible capped-simplex sweep.
- Environment evidence: `CityRecoveryEnv` is a Gymnasium environment with exactly the ordered services `transport`, `housing`, `food`, `healthcare`, `public_services`; observation shape is 23 and action shape is 5; Gymnasium environment checking passed. The response identifies `numpy.PCG64`.
- Authored workflow: the scenario used budget `233`, initial services `[0.11,0.29,0.52,0.18,0.43]`, priorities `[1.8,1.35,0.75,1.95,1.1]`, shock probability `0.35`, severity range `[0.07,0.37]`, and a forced day-9 epidemic. Six out-of-bound/schema-invalid scenario variants were rejected.
- Determinism and sensitivity: same-seed result hashes were identical at `4473154883b4abf72d183d76a5cab1a2de18607e5b357249a871d84dfd7414a8`; the changed seed produced result `9b9f3918c5b6890ae3e8489ca36cac86aca8285b389cd33e34d1253fa51b8952` and a different shock-tape hash. Candidate and baseline trajectory shock objects exactly matched the single top-level precomputed tape on every day.
- Inspectability and baseline: both trajectories contained all 17 daily records with before/shocked/end services, raw proposal, bounds, allocation, projection measurements, gain/strain, resilience, and reward. Candidate records exposed five ONNX actions. Every baseline record exposed `OR-Tools`, `GLOP`, `OPTIMAL`, objective text, and objective coefficients; top-level metadata states future shocks are not visible.
- Constraints: 34 daily candidate/baseline allocations were remeasured independently. Candidate and baseline lower/upper/budget/sum counts were all zero, serialized counts agreed, and allocation sums matched each daily available budget within `1e-7`. All 250 additional randomized feasible projector cases also had zero lower, upper, budget, and sum violations.
- Measured authored result: candidate resilience AUC `0.4782597`, visible baseline `0.46312637`, outcome `candidate_higher_rauc`.
- Cleanup: no server or persistent store was created; artifact bytes remained unchanged.
- Remaining Feature Complete work: held-out evaluation/trade-offs/leakage; persistence/restart/metadata/jobs/failures; focused tests/build and bounded API/browser smoke.

## Group 4 - Held-Out Evaluation, Trade-Offs, Reproducibility, And Leakage Boundary

- Result: PASS
- Commands:
  - `./.venv/Scripts/python.exe scripts/evaluate_policy.py`
  - Independent inline Python audit rebuilding all 40 family/seed cases once, comparing canonical result/scenario/shock hashes and saved metrics, remeasuring every daily constraint, recalculating all aggregate means and the 5,000-sample paired bootstrap interval, and checking training/holdout set intersections.
  - `git diff --exit-code -- evaluation/feature_complete_report.v1.json`
  - `Get-FileHash evaluation/feature_complete_report.v1.json -Algorithm SHA256`
- Frozen evaluation reproduction: the preregistered five held-out families x eight seeds x three canonical executions completed 40 cases / 120 executions with determinism mismatches `0`, total violations `0`, and report SHA-256 `fea00d1bf578c7d52cad816eed732a58ffb3f9b809c2788ba35c601e976f9351`. Rerunning produced no report diff.
- Independent case audit: all 40 canonical case hashes, scenario hashes, shock-tape hashes, and planner metrics matched the frozen report. The audit independently checked 1,462 candidate/baseline daily allocations and found zero lower, upper, budget, or sum violations.
- Leakage boundary: training families are `train_transit_cascade`, `train_displacement`, `train_supply_interrupt`, and `train_health_surge` at seeds `170100..170107`; held-out families are the five `holdout_*` families at seeds `271700..271707`. Family and seed intersections are both empty. Metadata names only the four training families, the protocol explicitly excludes the held-out seeds/prefix, and training code constructs units only from `TRAINING_FAMILIES` x `TRAINING_SEEDS`.
- Recomputed measurements: candidate resilience AUC `0.49148043`, baseline `0.44418455`, delta `+0.04729588`; post-shock shortfall delta `-0.00020822`; recovery-day delta `-1.025`; critical-service-day delta `-1.9`. Candidate resilience was higher on `40/40` cases. The independent paired PCG64 bootstrap reproduced `[0.04273770, 0.05176164]` with seed `1717` and 5,000 samples.
- Claims: the frozen outcome remains `measured_resilience_improvement`; the report separately exposes every recovery measure and retains synthetic/non-empirical, finite-protocol, non-causal, and non-municipal limitations.
- Cleanup: evaluation content remained byte-identical; no runtime process was started.
- Remaining Feature Complete work: persistence/restart/metadata/jobs/failures; focused tests/build and bounded API/browser smoke.

## Group 5 - Persistence, Restart, Metadata, Progress Applicability, And Structured Failures

- Result: PASS
- Commands:
  - Bounded PowerShell orchestration of two hidden Uvicorn processes on fixed loopback port `4117` with an isolated tester state directory, two identical comparison requests, same-process restore, stop/restart restore, corrupt-state probe/restoration, maximum 30-day timing, OpenAPI inspection, and deterministic cleanup.
  - Independent inline Python `TestClient` dependency/computation-failure probe plus three-result temporary `RunStore` content-identity/index-order probe.
- Runtime identity and metadata: `/health/live`, `/health/ready`, and `/api/v1/meta` returned 200. Metadata exposed exact candidate `3c16f0359cca93e494cc65f0a8850ef6e9c744da`, profile `cpu`, default seed `20260714`, API/dataset `2.0.0`, model `city-recovery-sb3-ppo-v1`, ONNX SHA-256 `983b7090e9cfc761b7b2118a24cff907abfc9caa74036cfb16bd9218346b11d8`, visible GLOP baseline, and synthetic dataset status.
- Persistence/restart: authored result id `68e955c9a30ddcbacd13f6ed4cc14d90a57590290446b6fc8dc3f35e27c38345` had response SHA-256 `0235da6aa164385509d4a288becac96041b99ed705fac5d25ecf74f5dcc12591`. Two identical posts, same-process GET, and GET after full Uvicorn stop/restart were byte-identical. A separate three-result probe independently recomputed each content-derived id, proved repeated saves identical, restored through new `RunStore` instances, and returned ids in stable lexical order regardless of insertion order.
- Structured failures: invalid scenario returned 422 `INVALID_SCENARIO` with field details; absent saved result returned 404 `PERSISTENCE_FAILED`; malformed id and corrupt saved bytes returned 500 `PERSISTENCE_FAILED`; injected computation failure returned 500 `COMPUTATION_FAILED`. With policy loading forced to fail, liveness alone stayed 200 while ready, metadata, saved-result index, comparison, and root all returned JSON 503 `DEPENDENCY_NOT_READY`; no route returned candidate content or fallback output.
- Job/progress applicability: OpenAPI exposes no job, progress, event, or SSE route. The only production computation is the strictly bounded 7-30 day synchronous comparison; the maximum 30-day case completed in `49 ms` here with complete 30-day trajectories for both planners. The written runtime condition requires job status/SSE for **long-running work**, so it is not triggered by the current production surface. If training or full evaluation is later exposed through the API, job status and SSE become required.
- Cleanup: both test servers stopped, port `4117` was released, the isolated state directory and temporary store were removed, and only ignored `.run/tester-feature-complete` server logs remain.
- Remaining Feature Complete work: focused backend/frontend/type/build checks and bounded functional browser smoke.

## Group 6 - Focused Tests, Type/Build, API, And Bounded Browser Smoke

- Result: PASS
- Commands:
  - `./scripts/setup.ps1 -Profile cpu`
  - `./scripts/preflight.ps1 -Profile cpu`
  - `./.venv/Scripts/python.exe -m pytest -q`
  - `./.venv/Scripts/python.exe -m ruff check backend scripts`
  - `npm test --prefix frontend`
  - `npm run typecheck --prefix frontend`
  - `npm run build --prefix frontend`
  - Playwright CLI `0.1.17`, named system-Chrome session `ai17fc`, one bounded desktop workflow against the compiled app at `127.0.0.1:4117`.
- Automated results: locked setup/build PASS with npm audit `0` vulnerabilities; normal preflight artifact/smoke check PASS; backend `28 passed`; Ruff PASS; frontend `6 passed`; strict TypeScript PASS; Vite production build PASS with 1,775 transformed modules and unchanged tracked `frontend/dist` content.
- Browser workflow: edited a genuinely new scenario to name `Tester browser corridor`, seed `8675309`, horizon `10`, daily units `211`, and shock chance `29`; ran the real comparison; observed candidate resilience `47.2%`, visible OR-Tools baseline `42.9%`, delta `+4.25` percentage points, recovery days `1 / 2`, three shocks, candidate/baseline measured violations `0 / 0`, shock hash prefix `a1233d160c8e5a2a`, and ONNX hash prefix `983b7090e9cfc761`.
- Workflow completeness: Trajectory exposed per-service daily candidate/baseline allocations. Daily audit exposed exactly 10 rows with day, shock, budget, both resilience values, and delta. After an unsaved name edit, selecting the persisted result restored the authored scenario and evidence. Submitting Days=`31` replaced all prior result evidence with the sole `Scenario invalid` alert and actionable field message; selecting the saved result recovered the successful view.
- Browser evidence: `.playwright-cli/traces/trace-1784201043246.trace`, `.playwright-cli/traces/trace-1784201043246.network`, and `.playwright-cli/page-2026-07-16T11-29-00-430Z.png`. Requests were loopback-only. The sole console entry was Chromium's expected failed-resource message for the deliberate HTTP 422 invalid request; no JavaScript exception or warning occurred.
- Non-blocking environment backlog: `./scripts/preflight.ps1 -Profile cpu -Full` reached and passed artifact smoke, then the existing ignored `.venv/Scripts/pytest.exe` launcher failed with `uv trampoline failed to canonicalize script path`, even after ordinary locked setup. Direct `python -m pytest` ran the exact 28-test suite successfully, `ruff.exe` works, the normal required preflight passes, and this gate does not authorize a clean/empty-cache Release setup. Making the optional Full branch call `uv run --frozen python -m pytest` would avoid stale relocated Windows entry-point launchers.
- Cleanup: Playwright session closed, Uvicorn stopped, port `4117` released, isolated browser state removed, and the final tracked diff remains tester feedback only.
- Checks intentionally not run: Presentation viewport/accessibility/focus/reduced-motion/demo matrix; clean or long-path clone; empty-cache setup; exhaustive offline/browser matrix; occupied-port or child-death tests; Repeat-5 torture; complete Release process/residue matrix; `verify.ps1` because it deliberately performs the excluded five-repeat workflow.

## Final Recommendation

- Feature Complete recommendation: **PASS** for frozen candidate `3c16f0359cca93e494cc65f0a8850ef6e9c744da`.
- Blocking issues (maximum five): None.
- Non-blocking backlog:
  - Distinguish requested PPO steps (`30000`) from SB3 checkpoint rollout-complete `num_timesteps` (`30208`) in provenance wording.
  - Prefer module invocation for pytest in the optional Full preflight branch so an ignored relocated Windows `.venv` cannot strand the console-script trampoline.
- Residual boundary: training was not rerun from scratch, because empty-cache setup/retraining and Release reproducibility are outside this assignment. The frozen checkpoint loaded as real PPO, its ONNX export/parity were independently reproduced, and the entire held-out report was regenerated byte-identically.
- Named reviews not claimed: Presentation and Release.
- This tester changed no implementation, configuration, lock, builder-status, evaluation, or accepted-ledger file and does not issue the global judge verdict.
