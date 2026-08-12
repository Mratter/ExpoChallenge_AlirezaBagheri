# Five-seed action-mean ensemble: 200-case development result

This is a **development-only exploratory candidate**, not a final-split result and not a deployed-policy result. The single preregistered candidate averaged the deterministic actions of the five 2,000,000-transition seed endpoints. It was retained as evidence but **not promoted**: the shipped single-actor ONNX artifact and application wiring remain unchanged.

## Result

| Development policy | Solved | Rate | Hard violations | Maximum conservation residual |
|---|---:|---:|---:|---:|
| Five-seed action mean | **179/200** | **0.895** | 0 | 0.0 |
| Selected single checkpoint | 178/200 | 0.890 | 0 | 0.0 |

The ensemble gained one solved case in aggregate. The matched rows show that it does not strictly dominate the selected checkpoint:

| Both solved | Ensemble only | Selected only | Neither solved |
|---:|---:|---:|---:|
| 176 | 3 | 2 | 19 |

The three ensemble-only rows were `v3_dev_river_flood:820009`, `v3_dev_seismic_cluster:820014`, and `v3_dev_health_compound:820012`. The two selected-only rows were `v3_dev_river_flood:820004` and `v3_dev_seismic_cluster:820037`.

## Exact ensemble identity

Each actor normalized the same raw observation with its **own frozen observation RMS**, produced one deterministic clipped 22-action vector, and contributed equal weight `0.2`. The arithmetic mean was accumulated in float64, clipped to `[-1, 1]`, and cast to float32 before the environment step.

| Policy seed | Checkpoint | Observation-RMS SHA-256 |
|---:|---|---|
| 37017 | `seed-37017-ppo-2000000` | `456c8fab41d53a8d1ecc23fdf461cc9df5642726cff0f84f5bb2f94643876835` |
| 47017 | `seed-47017-ppo-2000000` | `77156039dd87a2873fb4f1098385d2f163346f4a153a30fe728e7673a9f342ea` |
| 57017 | `seed-57017-ppo-2000000` | `a75c02959bde3cc909e9409c980ee3685ae726a5f94bf2cb5096e2ab19252c97` |
| 67017 | `seed-67017-ppo-2000000` | `6823fd134e915a0d22d149895479a003c51711d3f2b0649c37205674365cd022` |
| 77017 | `seed-77017-ppo-2000000` | `6cca61ae612700d33cbbcdae8e46d9e4997cd8916859f7255b16dbb9cb344b4f` |

## Deployment status

This result cannot be substituted for the shipped single-actor artifact. Promotion would require all of the following:

1. A self-contained five-actor export embedding all five frozen observation transforms.
2. SB3-to-ONNX action parity on identical observations.
3. Exact full-development outcome parity.
4. A new lightweight manifest identifying every actor and normalizer.
5. Explicit application wiring and served-path verification.

Those steps were not performed. The one-case development gain and the 3-versus-2 matched tradeoff should be weighed against the additional artifact size and five-actor inference cost.

## Evidence

- [Machine receipt](../../internal/developmental_runs/v4/action-mean-ensemble-5x2m-dev-200.json) SHA-256: `7b23372b2ec45910f404754de57a5c2e582e81b5ca96c2bbe24f6f1c703287c7`
- Ensemble rows SHA-256: `368425b60f669734c7315e7fe3146ddc7532d12cea0d75b0260b790061b69817`
- Ensemble-member identity SHA-256: `a84c7bf80a485e3ab1b8d2671af3a10482d44afae13b65ef8c3e6f2958399cb4`
- Matched selected-policy parity rows SHA-256: `ca9320566b86dfb7a02d2cb9232c7a28c80f08dbbd700dffc6d2af9af1c22d6b`

No final case was constructed or evaluated for this experiment.
