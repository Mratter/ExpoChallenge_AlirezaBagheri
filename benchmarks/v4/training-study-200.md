# Training study: 200-case development protocol

This is development-only evidence. Every training endpoint, checkpoint selection row, and SB3-to-ONNX parity row uses the 200-case development roster. The final split was not used, so none of the numbers below is a final-split performance claim. The complete tracked digest and receipt index is in [`training-study-200-summary.json`](../../internal/developmental_runs/v4/training-study-200-summary.json).

## Five-seed baseline

The registered baseline used the original `v3_equivalent` reward, BC/DAgger actor warm-start, frozen observation RMS, VecNormalize, and 2,000,000 active actor-critic transitions.

| Policy seed | Solved | Solve rate | Mean resilience AUC | Mean minimum tail margin |
|---:|---:|---:|---:|---:|
| 37017 | 172/200 | 0.860 | 0.4878306170 | 0.0421613017 |
| 47017 | 171/200 | 0.855 | 0.4855700816 | 0.0458952172 |
| 57017 | 171/200 | 0.855 | 0.4866063695 | 0.0487350368 |
| 67017 | 174/200 | 0.870 | 0.4884108993 | 0.0474303055 |
| 77017 | 169/200 | 0.845 | 0.4882343725 | 0.0468973412 |

Mean solved count was 171.4/200 with sample standard deviation 1.816590212458495. All five endpoints had zero hard violations and maximum conservation residual 0.0.

## Matched ablations

Each treatment used seeds 37017, 47017, and 57017. Its paired control endpoints were 172, 171, and 171 solves respectively. Deltas are treatment minus the same-seed control.

| Treatment | Treatment endpoints | Paired solve deltas | Mean solved | Mean paired delta |
|---|---:|---:|---:|---:|
| No BC actor warm-start | 145 / 156 / 151 | -27 / -15 / -20 | 150.66666666666666 | -20.666666666666668 |
| Frozen risk-averse reward | 173 / 171 / 177 | +1 / 0 / +6 | 173.66666666666666 | +2.3333333333333335 |
| No VecNormalize | 140 / 134 / 144 | -32 / -37 / -27 | 139.33333333333334 | -32.0 |
| Preparedness alignment 2.0 | 169 / 170 / 173 | -3 / -1 / +2 | 170.66666666666666 | -0.6666666666666666 |
| 645k active-transition budget | 170 / 169 / 168 | -2 / -2 / -3 | 169.0 | -2.3333333333333335 |

The risk-averse arm used the already-frozen historical reward profile and its registered preparedness-alignment coefficient of 2.0. No reward coefficient was retuned, searched, or selected from these results. All 15 ablation endpoints also had zero hard violations and maximum conservation residual 0.0.

## Solve-count selection and export

Selection ranked 20 checkpoints only by solved count, with earlier active transitions and lower policy seed as neutral tie-breakers; resilience AUC was descriptive and was not used for selection. The winner was `seed-67017-ppo-1000000` at **178/200**, four solves ahead of the 174/200 runner-up. No tie-break was needed.

The selected SB3 checkpoint and exported ONNX policy both solved 178/200 development cases. Across 6,000 action vectors (132,000 action elements), maximum absolute action error was `1.9073486328125e-06` against a `1e-05` tolerance. Per-case outcomes matched exactly, maximum resilience-AUC error was `1.0000000050247593e-08`, deterministic replay mismatches were zero, hard violations were zero, and maximum conservation residual was 0.0. VecNormalize observation moments are embedded in the opset-17 graph, the raw ONNX output is `[batch, 22]`, and actions are clipped to `[-1, 1]`.

## Evidence chain

| Evidence | SHA-256 |
|---|---|
| External five-seed summary (`E:/city-recovery-v4-study-200-attempt-01/seed-sweep-summary.json`) | `7ab75187e8233aa088ea00a334c69d6d5a599e8f28841c85fc02746d6830ccc8` |
| External ablation summary (`E:/city-recovery-v4-study-200-attempt-01/ablation-summary.json`) | `1623d97423ff76d3a59662fcce93266ca67d0899a089e81eb762c28e060575e6` |
| [Checkpoint selection](../../internal/developmental_runs/v4/checkpoint-selection-200.json) | `65fefa91903e6e7539ead5e1a957528454a9c01e8084ace56fa5047738e73e00` |
| [Parity receipt](../../internal/developmental_runs/v4/city_recovery_ppo.v4.parity.json) | `e3b487df8221db75d58dc68eccbc9df93af16cb0e9f17b5bc60cf50a5b42ba6c` |
| [Raw exporter manifest](../../internal/developmental_runs/v4/city_recovery_ppo.v4.export-manifest.raw.json) | `a2ba49ebabcd79681b93d8d142523e2823b9476025ea0e2645a8d9dcfefac45c` |
| [Portable ONNX manifest](../../artifacts/city_recovery_ppo.v4.manifest.json) | `7ecc9948789163febf9cc9a455e20c0d5e5fb75c70919598169f21614e1a5a06` |
| [ONNX artifact](../../artifacts/city_recovery_ppo.v4.onnx) | `a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483` |

The selected checkpoint SHA-256 is `4e84c74eda1334212ac3680897b3077f62e22b271b7ecc943911f9fd4aa55f22`; its normalization file SHA-256 is `5f566a43549e5ef3757ed59982f20c2e2bf65e3cd7a3c0036fa13c10a5c320e6`.
