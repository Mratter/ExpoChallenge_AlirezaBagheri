# Model Workbench presenter guide

## Start the workbench

First setup on this worktree:

```powershell
Set-Location 'C:\Users\Alireza\Desktop\city-model-workbench'
.\scripts\setup.ps1 -Profile cpu
```

For the presentation:

```powershell
Set-Location 'C:\Users\Alireza\Desktop\city-model-workbench'
.\scripts\run.ps1 -Profile cpu
```

Open `http://127.0.0.1:4117`. The application is local and CPU-only.

## Two-minute walkthrough

1. Start with the landing screen. Say: “This is the trained Adaptive Cascade MLP v2:
   a real 300,113-parameter ONNX policy evaluated against a fixed visible-need
   heuristic on sealed synthetic scenarios.”
2. Point to **38 / 40 model passes**, **20 / 40 heuristic passes**, and the direct
   **38–0–2** scoreline. Explain that the first two are independent objective counts;
   the last is the complementary matched comparison.
3. Click **Open Model Toolbox**. Choose a preset or change the forecast, visible-need,
   public-regime, service-health, and phase controls, then click **Run real model**.
4. Show the result panel. It executes the sealed ONNX artifact and displays the
   learned action, the heuristic action, model confidence, logits, action
   probabilities, and the exact 21-value input vector.
5. Explain the model: “The network receives 21 public values: forecast signals,
   visible need, public regime telemetry, current service health, and phase context.
   It chooses one of five service interventions. It never receives the scenario seed,
   family name, hidden target, or future tape.”
6. Explain why learning helps: “The heuristic always serves the largest visible need.
   The learned model can recognize how several public signals interact and act before
   a visible problem turns into a larger cascade.”
7. Scroll to Architecture to show the exact **21 → 534 → 534 → 5** trained model and
   its training receipt. The older research tracks are intentionally omitted from the
   presentation surface.
8. Finish with the sealed benchmark and Evidence sections. The model contained an average of
   **2.25 more cascade windows per scenario**, with a paired bootstrap 95% interval
   of **[1.775, 2.750]**.
9. Show the secondary metric: critical-service deficit AUC was **0.303831090705**
   for the learned model versus **0.33655319428** for the heuristic. Lower is better.
10. Finish at Evidence. Say: “The result is sealed and independently replayed. ONNX
   action parity passed, every matched pair used the same tape, and there were zero
   hard violations.”

## What was trained

The active showcase model is a genuine MLP with **300,113 trainable parameters**,
**21 inputs**, and
**5 actions**. Supervised training used **9,600 labeled windows** from **800 synthetic
scenarios** for **120 epochs**. The measured CPU training time was **66.73 seconds**.
The exported ONNX model selects the same actions as the source PyTorch model within
the registered parity tolerance.

The archived 5,893-parameter v1 model passed 39/40 on a different sealed holdout.
Do not say v2 is more accurate than v1: v2 is **50.9× larger** and produced a stronger
38–0–2 direct margin on its own matched holdout, but cross-holdout score differences
cannot be attributed solely to model width.

Keep the claim narrow: this result demonstrates learnable pattern recognition inside
a purpose-built synthetic benchmark. It does not replace the production v2 result,
repair the preliminary v4 or R9 no-go findings, train R22/V10, or establish real-world
disaster performance.

## Evidence and model files

Synthetic showcase repository:

- `C:\Users\Alireza\Desktop\city-showcase-benchmark\artifacts\final\adaptive-cascades-showcase-v2\terminal.json`
  — verified-complete terminal marker.
- `C:\Users\Alireza\Desktop\city-showcase-benchmark\artifacts\final\adaptive-cascades-showcase-v2\result.json`
  — sealed 40-scenario result and all headline comparisons.
- `C:\Users\Alireza\Desktop\city-showcase-benchmark\artifacts\final\adaptive-cascades-showcase-v2\manifest.json`
  — hashes for the complete final evidence package.
- `C:\Users\Alireza\Desktop\city-showcase-benchmark\artifacts\final\adaptive-cascades-showcase-v2\replay-report.json`
  — independent exact replay receipts.
- `C:\Users\Alireza\Desktop\city-showcase-benchmark\artifacts\final\adaptive-cascades-showcase-v2\anti-gaming-report.json`
  — information-boundary, shared-tape, split, and policy-blindness checks.
- `C:\Users\Alireza\Desktop\city-showcase-benchmark\artifacts\candidate\adaptive-cascades-showcase-v2\adaptive-cascade-mlp-v2-300k.onnx`
  — runnable showcase model.
- `C:\Users\Alireza\Desktop\city-showcase-benchmark\artifacts\candidate\adaptive-cascades-showcase-v2\training-receipt.json`
  — architecture, dataset, timing, and ONNX-parity record.

Existing workbench tracks:

- `artifacts/city_recovery_ppo.v2.onnx` — runnable production v2 model.
- `artifacts/city_recovery_ppo.v2.metadata.json` — production training identity.
- `evaluation/feature_complete_report.v2.json` — production v2 evaluation.
- `artifacts/workbench/overview.v1.json` — workbench evidence contract.
- `artifacts/workbench/manifest.v1.json` — hash-pinned workbench manifest.
- `configs/v5/models/r18_solver_guided_autoregressive.json` — untrained R22 registration.

## Verify the presentation package

This is the fast, self-contained check to run before presenting. It verifies the
sealed evidence bundle, frontend build, API contract, and a real CPU ONNX inference:

```powershell
Set-Location 'C:\Users\Alireza\Desktop\city-model-workbench'
.\scripts\preflight.ps1 -Profile cpu
```

Expected status: `workbench-preflight-passed`, with 21 model inputs, 5 outputs,
`CPUExecutionProvider`, and no training stack imported.

## Verify the original sealed showcase source

This command is read-only and checks the sealed source, candidate, manifest, rows,
replay receipts, and result hashes:

```powershell
Set-Location 'C:\Users\Alireza\Desktop\city-showcase-benchmark'
& 'C:\Users\Alireza\Desktop\city-r17\.venv\Scripts\python.exe' scripts/run_campaign.py --version v2 verify
```

Expected status: `verified`, with 40 rows. Do not rerun training or final evaluation
for the completed v2 evidence package. The archived v1 package remains separately
verifiable with `--version v1`.

## Short answers for questions

- **Is 38 / 40 accuracy?** No. It is the number of scenarios in which the policy met
  the preregistered containment objective.
- **Why did the model win?** It learned a joint mapping from several public signals;
  the comparator was a static heuristic that only followed current visible need.
- **Was the comparison matched?** Yes. Both policies used the same 40 sealed tapes,
  action space, and policy-blind transition system.
- **Is this the production model?** No. It is the fifth, deliberately synthetic
  showcase track. Production v2 and the v4, R9, and R22/V10 research tracks retain
  their own identities and results.
- **Is it real-world validated?** No. It is a controlled artificial demonstration.
