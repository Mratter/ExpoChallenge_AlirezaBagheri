# Development

This guide covers local setup, runtime configuration, verification, and the training/deployment workflow. For architecture, use the [code tour](CODE_TOUR.md). For measured claims and receipts, use [Evidence and Results](EVIDENCE.md).

## Runtime setup

From the repository root on 64-bit Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

Setup resolves or installs Python 3.12 and a supported Node.js LTS release, creates the project Python environment, installs `requirements.txt`, installs the locked frontend packages with `npm ci`, builds `frontend/dist`, and runs runtime preflight. The launcher serves FastAPI at `127.0.0.1:4117` and opens the landing page at `#/`.

The normal path is zero-configuration. `artifacts/city_recovery_ppo.v4.onnx` is bundled and preflighted before startup. A GPU, CUDA, Git, and `uv` are not required.

To prevent setup from invoking `winget`, install 64-bit Python 3.12 and Node.js LTS first, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 `
    -SkipToolBootstrap
```

## Policy selection

The launchers use one fail-closed precedence rule:

1. an explicit `-PolicyPath`;
2. a nonblank `INNOVERSE_POLICY_PATH`; and
3. the bundled `artifacts/city_recovery_ppo.v4.onnx`.

An invalid higher-priority choice is an error; it never falls through to a fixture or lower-priority path. To override the bundle:

```powershell
$env:INNOVERSE_POLICY_PATH = 'C:\path\to\environment-policy.onnx'
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 `
    -PolicyPath 'C:\path\to\explicit-policy.onnx'
```

An optional digest makes the choice content-specific:

```powershell
$env:INNOVERSE_POLICY_PATH = 'C:\path\to\selected-policy.onnx'
$env:INNOVERSE_POLICY_SHA256 = (Get-FileHash `
    $env:INNOVERSE_POLICY_PATH `
    -Algorithm SHA256).Hash.ToLowerInvariant()
powershell -ExecutionPolicy Bypass -File .\scripts\preflight.ps1
```

`scripts/runtime_policy.ps1` implements PowerShell precedence. `model/policy.py`, `backend/app/main.py`, and `scripts/preflight_check.py` share the bundled default and enforce the runtime contract.

## Runtime contract and readiness

The selected policy must provide exactly:

| Contract | Required value |
| --- | --- |
| Input | `observation`, `tensor(float)[batch,73]` |
| Output | `action`, `tensor(float)[batch,22]` |
| Actions | Finite and inside `[-1,1]` |
| Provider | ONNX Runtime `CPUExecutionProvider` |
| Normalization | Embedded in the ONNX graph |

Run preflight separately with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\preflight.ps1
```

Preflight checks the Python runtime, built frontend, selected graph, raw shapes, CPU session, bounded smoke inference, and one deterministic comparison with zero hard violations and exact conservation.

`GET /health/live` reports process liveness only. `GET /health/ready` loads the selected artifact and reports its identity and 73/22 interface. `GET /api/v1/meta` adds the environment, order, outcome, baseline, persistence, and determinism contracts.

## Developer environment and checks

Setup may move the Python environment under `%LOCALAPPDATA%\Innoverse\city-recovery` when the repository path is too long for native libraries. Resolve it instead of assuming `.venv`:

```powershell
. .\scripts\project_environment.ps1
$ctx = Get-CityRecoveryEnvironmentContext -Root (Get-Location).Path
```

Install development-only packages if needed:

```powershell
& $ctx.PythonPath -m pip install `
    httpx==0.28.1 `
    pytest==8.4.1 `
    ruff==0.12.4 `
    stable-baselines3==2.7.0 `
    torch==2.8.0
```

Run Python verification:

```powershell
& $ctx.PythonPath -m pytest -q tests
& $ctx.PythonPath -m ruff check backend model scripts tests
```

Run browser verification:

```powershell
npm ci --prefix frontend
npm test --prefix frontend
npm run typecheck --prefix frontend
npm run build --prefix frontend
```

Check the generated Python-to-TypeScript contract:

```powershell
& $ctx.PythonPath .\scripts\generate_frontend_contract.py --check
```

CI runs Ruff and the full Python suite, plus frontend tests, type checking, and the production build.

## System boundaries

The learned policy, public planners, simulator, and feasibility projector are separate:

- `model/policy.py` loads and runs one ONNX actor.
- `backend/app/city/planners.py` contains the public reactive, teacher, tuned-rule, and MPC proposal logic.
- `backend/app/city/environment.py` owns observations, action decoding, transition composition, and rollouts.
- `backend/app/city/physics.py` owns allocation, logistics, shock, and conservation mechanics.
- `backend/app/city/outcome.py` owns the frozen six-check Solved conjunction.
- `backend/app/main.py` composes policy loading, simulation, persistence, analysis, exports, and static frontend delivery.

Both candidate and comparison planner receive the same exogenous shock tape but run in independent environment states. Their later observations may differ because their earlier actions changed services, stock, deliveries, and preparedness.

## Scenario roles

| Split | Families × seeds | Cases | Role |
| --- | --- | ---: | --- |
| Training | 6 × 32 (`810000–810031`) | 192 | Behavior cloning, DAgger, critic warm-up, and PPO interaction |
| Development | 5 × 40 (`820000–820039`) | 200 | Learning curves, comparisons, checkpoint selection, parity, and served replay |
| Final | 5 × 40 (`830000–830039`) | 200 | One retained owner-authorized evaluation of the frozen artifact |

The family sets and seed ranges are disjoint. Training tools do not import or evaluate final cases. The one retained learned-policy final run happened only after selection, export, parity, and artifact identity were frozen; further reruns remain unauthorized.

## Training flow

`scripts/train_policy.py` keeps the main flow linear:

```text
BC/DAgger -> actor-frozen critic warm-up -> PPO -> development milestones -> receipt
```

The publication study used four public-state DAgger iterations, then froze the actor while training the critic for at least 50,000 transitions and until explained variance exceeded `0.5` (subject to the registered maximum). PPO used 20 lanes, rollout size 5,000, batch size 500, five epochs, learning rate `7.5e-5`, clip range `0.15`, entropy coefficient `0.003`, and target KL `0.02` with Stable-Baselines3 early stopping.

The trainer records actor hashes, observation and return moments, explained variance, approximate KL, clip fraction, entropy loss, value loss, policy-gradient loss, and action standard deviation. Complete milestone bundles include model, optimizer, counters, normalization state, configuration, and a hash-bound manifest.

The registered five-seed study evaluated checkpoints at 200k, 500k, 1M, and 2M active actor-critic transitions. Selection ranked all 20 complete candidates by development solve count, then earlier transition count and lower seed. Seed `67017` at 1M won with 178/200; the runner-up solved 174/200.

Normal demo users do not need to run training. A maintainer-authorized new run must use new output paths because scientific tools refuse to overwrite receipts:

```powershell
& $ctx.PythonPath .\scripts\train_policy.py `
    --json-output .\internal\developmental_runs\v4\new-training-receipt.json

& $ctx.PythonPath .\scripts\evaluate.py `
    --split dev `
    --policy tuned `
    --policy 'onnx:C:\path\to\selected-policy.onnx'
```

## Deployment sequence

The shipped policy passed each stage documented in the [training deployment plan](TRAINING_DEPLOYMENT_PLAN.md):

1. durable checkpoint publication;
2. development-only checkpoint selection;
3. self-contained opset-17 CPU ONNX export with embedded observation normalization;
4. full 200-case SB3-to-ONNX parity across 6,000 action vectors and 132,000 action elements;
5. a portable descriptive manifest; and
6. a 200-case FastAPI `POST -> persist -> GET` served replay matching all 178 accepted development outcomes.

The manifest is descriptive. Runtime readiness validates the selected ONNX bytes directly; it does not depend on a source seal, authorization file, or manifest signature.

## Change checklist

When a change crosses layers, verify from the center outward:

1. preserve physics and outcome invariants;
2. test the Python consumer;
3. regenerate or check the TypeScript contract when a public value changes;
4. test the TypeScript parser and view model;
5. build the frontend; and
6. run preflight against the selected artifact.

Do not present a development improvement as a final result, a CEM search failure as infeasibility, a simulated retrospective trajectory as an observation, or a feasibility projection as a second planner.
