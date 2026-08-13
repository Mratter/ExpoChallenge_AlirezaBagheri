# Troubleshooting

Start with runtime preflight. It reports setup, policy, interface, inference, and smoke-comparison failures before the application starts:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\preflight.ps1
```

## Python or Node.js is missing

Run setup without `-SkipToolBootstrap` so it can use `winget`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

Or install 64-bit Python 3.12 and a supported Node.js LTS release manually. Check discovery with:

```powershell
py -3.12 --version
node --version
npm --version
```

If an external installer is still open, let it finish and rerun setup. The environment resolver refreshes tool discovery, so a new shell usually is not required.

## PowerShell blocks a script

Use the documented process-local policy override:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

This does not change the machine-wide execution policy.

## `DEPENDENCY_NOT_READY`

Run preflight and read the first failing check. Without an override, confirm that `artifacts/city_recovery_ppo.v4.onnx` exists. With an override, confirm that `-PolicyPath` or `INNOVERSE_POLICY_PATH` points to a readable ONNX file.

The selected graph must expose exactly:

- `observation: tensor(float)[batch,73]`;
- `action: tensor(float)[batch,22]`; and
- finite smoke-inference output inside `[-1,1]` on `CPUExecutionProvider`.

If `INNOVERSE_POLICY_SHA256` is set, it must match the winning file. An invalid explicit path or environment choice fails closed and never falls back to the bundle or legacy fixture.

`GET /health/live` may pass while `/health/ready` correctly returns 503: liveness means the process is running, not that the policy is ready.

## Port 4117 is already in use

Choose another port:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 `
    -Port 4120
```

Open `http://127.0.0.1:4120/#/toolbox`. Use the same alternate port for standalone preflight when needed.

## The project path is very long

Native Python libraries can exceed Windows path limits. Setup automatically places the environment under `%LOCALAPPDATA%\Innoverse\city-recovery\py312-<root-hash>` when needed.

Resolve the environment instead of assuming `.venv`:

```powershell
. .\scripts\project_environment.ps1
$ctx = Get-CityRecoveryEnvironmentContext -Root (Get-Location).Path
& $ctx.PythonPath --version
```

## The frontend is blank or stale

Stop the server, rebuild, and restart:

```powershell
npm ci --prefix frontend
npm run build --prefix frontend
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 -NoBrowser
```

Then open the Toolbox and hard-refresh with `Ctrl+F5`. Inspect `/health/live` and `/health/ready` separately.

## Setup installed packages but preflight fails

Use the exact Python environment chosen by setup:

```powershell
. .\scripts\project_environment.ps1
$ctx = Get-CityRecoveryEnvironmentContext -Root (Get-Location).Path
& $ctx.PythonPath -c "import fastapi, numpy, onnx, onnxruntime, uvicorn"
```

Rerun setup if the import check fails. Do not mix a global Python installation with the project environment.

## Saved runs are not where expected

The default state directory is:

```text
%LOCALAPPDATA%\Innoverse\ai17-city-recovery\runs
```

Check whether `INNOVERSE_STATE_DIR` was set in the PowerShell session that launched the app. To use another location:

```powershell
$env:INNOVERSE_STATE_DIR = 'D:\InnoverseRuns'
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

## A forced shock is rejected

Forced shocks are accepted only on days 1–27. Days 28–30 are the frozen assessment tail. This is part of the outcome contract, not a browser-only restriction.

## The 3D view is slow

The 3D route loads a larger rendering bundle than the Toolbox. Close graphics-heavy tabs, update the browser, reduce browser zoom, or use `#/toolbox`. Every numerical result and decision-support view remains available without the 3D scene.

## Policy results differ from the retained evidence

Confirm `/api/v1/meta` reports the expected policy identity and SHA-256. The bundled artifact SHA-256 is:

```text
a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483
```

Also confirm the scenario, seed, baseline identity, environment specification, and outcome-definition hash match. A different policy override, scenario value, or source identity legitimately produces a different content-addressed result.

For the retained claims and exact evidence paths, see [Evidence and Results](EVIDENCE.md). For test commands and runtime internals, see [Development](DEVELOPMENT.md).
