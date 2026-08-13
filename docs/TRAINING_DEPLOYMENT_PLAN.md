# Training deployment plan

This document records the gated path from durable training checkpoints to a served model. It did not itself authorize training or final evaluation. A later, separate owner authorization permitted exactly one learned-policy final run after the artifact was frozen; that run is complete, and further final reruns remain unauthorized.

The plan therefore has one pre-training stage and five ordered post-training stages: select on development cases, export the selected actor, prove SB3-to-ONNX parity, publish descriptive metadata, and connect the selected artifact to the application. Every performance decision in Stages 1–5 uses the 200 development cases.

## Current publication status

**Development oracle-solved reference: privileged future-aware CEM found solutions for 187 of 200 development cases; its 13 search failures are not proofs of infeasibility.** The completed five-seed sweep registered 20 development checkpoints across policy seeds `37017`, `47017`, `57017`, `67017`, and `77017`. Their 2M endpoints solved 172, 171, 171, 174, and 169 cases respectively, for mean **171.4 / 200**, or **171.4 / 187 = 91.7%** of that oracle-solved reference, with population standard deviation **1.62** and sample standard deviation **1.82**; no Wilson interval is reported for an optimizer-seed mean. The population value describes the complete registered five-seed sweep; the sample value describes dispersion when those seeds are treated as a sample of optimizer randomness. Solve-count selection chose seed `67017` at 1M active actor-critic transitions with **178 / 200**, or **178 / 187 = 95.2%** (descriptive post-hoc Wilson 95% **[0.9111, 0.9745]**), four solves and two percentage points ahead of the **174 / 200** runner-up; no tie-break was needed. The complete study index is `internal/developmental_runs/v4/training-study-200-summary.json`, with its human-readable report at `benchmarks/v4/training-study-200.md`.

`artifacts/city_recovery_ppo.v4.onnx` is the selected self-contained opset-17 artifact, SHA-256 `a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483`. Its raw graph contract is `observation: tensor(float)[batch,73]` to `action: tensor(float)[batch,22]`. The 200-case parity receipt records 178 solves for both SB3 and ONNX, maximum action error `1.9073486328125e-06`, maximum resilience-AUC error `1.0000000050247593e-08`, zero replay mismatches, zero hard violations, and exact conservation. The neighboring manifest binds the artifact, selected checkpoint, observation normalization, selection receipt, and parity receipt.

The runtime now uses that artifact without configuration. `-PolicyPath` overrides `INNOVERSE_POLICY_PATH`, which overrides the bundle; an invalid higher-priority path fails closed. Focused readiness and preflight checks verify this integration. The application-level gate has also passed: all 200 development cases ran through FastAPI `POST` → persist → `GET`, and the served policy exactly reproduced the accepted **178 / 200 development solves**. After this deployment chain and artifact identity were frozen, the exact shipped artifact's single owner-authorized final evaluation solved raw **163 / 200** with receipt-level Wilson 95% interval **[0.7554293724, 0.862698072]**; its oracle-solved-reference interpretation is recorded under Explicit boundaries below. Its [canonical report](../benchmarks/v4/final-results-200.md) and machine [success receipt](../internal/evaluation_runs/v4/final-evaluation-200.success.json) are separate from development selection and did not feed back into it.

## 0. Make milestone checkpoints durable before training

The current `scripts/train_policy.py` publishes complete, atomic checkpoint bundles at every registered development milestone and at the terminal budget. Each bundle is bound into the training receipt and remains available for deterministic selection, export, or explicitly characterized continuation.

Each bundle must be written through temporary files, flushed, atomically renamed, and then bound into the training receipt by path and SHA-256. It must contain:

- the complete Stable-Baselines3 policy and value state;
- optimizer state, `num_timesteps`, and schedule/progress state needed to interpret the checkpoint;
- the exact frozen observation RMS mean, variance, count, epsilon, and clipping settings;
- reward-normalization state needed to continue training, kept separate from deployable observation preprocessing;
- training configuration, seed set, completed transition count, and milestone ID; and
- a canonical bundle manifest covering every file hash and the observation RMS digest.

Before any future authorized training compute, retain the fresh-process round-trip test that reloads a synthetic checkpoint bundle and proves byte-identical policy hashes, optimizer-state hashes, counters, observation RMS digest, and deterministic actions on fixed raw observations. A bundle is always valid for selection, evaluation, and export; it is called bit-exact resumable only if vector-environment lane state, lane RNG state, and in-progress normalization returns are also captured and verified. Otherwise the receipt must label resume as non-bit-exact and must not present a restarted continuation as the uninterrupted registered run.

## 1. Select one checkpoint on development solve rate

Evaluate every complete candidate checkpoint from the authorized training run on the same 200 development cases and deterministic disaster tapes. Use the deterministic actor, its matching frozen observation-normalization state, and the canonical outcome implementation.

Rank checkpoints by:

1. development cases solved out of 200;
2. earlier active transition count if solve counts tie; and
3. lower registered policy seed if both solve count and transition count tie.

Resilience AUC and minimum assessment-tail margin remain descriptive diagnostics;
neither may affect checkpoint selection.

Write one selection record containing the complete checkpoint score table and, for the winner and runner-up:

- checkpoint path, SHA-256, and completed transition count;
- observation RMS SHA-256;
- solved count and solve rate;
- mean resilience AUC and mean minimum tail margin;
- the winning primary margin in solved cases and percentage points; and
- any tie-break level used, including its signed margin.

The selected checkpoint must be traceable to the training receipt and must have a complete optimizer checkpoint, actor weights, and observation RMS state. A checkpoint with a better secondary metric cannot displace one with a higher development solve count.

## 2. Export a self-contained CPU ONNX actor

Export only the selected deterministic actor using ONNX opset 17. The public interface remains:

| Contract | Required value |
| --- | --- |
| Input | `observation`, `tensor(float)[batch, 73]` |
| Output | `action`, `tensor(float)[batch, 22]` |
| Batch axis | Dynamic and named `batch` |
| Action bound | Every output finite and in `[-1, 1]` |
| Serving provider | `CPUExecutionProvider` only |

Prefer a self-contained graph that accepts raw environment observations. Bake the selected frozen VecNormalize observation transform into the graph before the actor:

```text
normalized = clip((observation - mean) / sqrt(var + 1e-8), -10, 10)
action = clip(deterministic_actor(normalized), -1, 1)
```

Embed the 73-value mean and variance as graph constants. Record the RMS count and the SHA-256 produced from the frozen mean, variance, and count using the trainer's canonical RMS digest. The export record must bind that RMS SHA-256 to both the selected checkpoint and the ONNX SHA-256.

If baking the transform proves technically impossible, stop the phase and report the blocker. Do not silently move normalization into an unrecorded serving wrapper or serve an actor that expects normalized values under the raw `observation` contract.

After export, inspect the ONNX graph and an ONNX Runtime CPU session. Reject an unexpected opset, extra public tensors, symbolic action width, wrong names or dtypes, unavailable CPU provider, non-finite output, or output outside the action bounds.

## 3. Prove full development parity

Run matched SB3 and ONNX candidate rollouts on all 200 development cases. Each implementation receives the same raw observation at each step: the SB3 reference applies the selected frozen VecNormalize state, while the ONNX graph applies the baked transform. Compare every one of the 22 action elements across all 30 days and 200 cases: **6,000 paired action vectors and 132,000 paired action-element comparisons** in total.

The parity receipt must include checkpoint, ONNX, and RMS hashes; interface inspection; per-case rows; sample counts; and observed maxima. It passes only when all of these conditions hold:

- maximum elementwise absolute SB3-versus-ONNX action error is at most `1e-5`, with `rtol=0`;
- every action from both implementations is finite and within `[-1, 1]`;
- every case has the same solved/failed result under SB3 and ONNX;
- the aggregate development solve counts are exactly equal;
- maximum per-case resilience-AUC absolute error is at most `1e-6`;
- ONNX replay produces no hard violations and a maximum conservation residual at or below `1e-6`; and
- a second ONNX replay of every case reproduces the first ONNX trajectory digest exactly.

Record the SB3 solve count, ONNX solve count, per-case outcome mismatch count, maximum action error, maximum resilience-AUC error, replay mismatch count, hard-violation count, maximum conservation residual, and a canonical hash of the 200 parity rows. Any failed condition stops deployment; it is not rounded away or accepted through an aggregate-only comparison.

## 4. Publish lightweight deployment metadata

Create a small descriptive manifest beside the selected ONNX artifact. It records provenance and interface facts but is not a release lock or a second policy loader. Include:

- manifest schema version and model identifier;
- ONNX path and SHA-256;
- selected checkpoint ID, path, SHA-256, and transition count;
- the complete training configuration and policy/training seed set recorded by the source training receipt;
- selection split, winning development solve count, runner-up solve count, and winning margin;
- observation RMS SHA-256 and `normalization_baked_into_graph: true`;
- input/output names, dtypes, shapes, action bounds, and opset;
- required runtime provider (`CPUExecutionProvider`);
- observation-order and action-order SHA-256 values;
- parity receipt path and SHA-256; and
- parity case count, action tolerance, observed maximum action error, and exact SB3/ONNX solve counts; and
- Python, NumPy, PyTorch, Stable-Baselines3, ONNX, ONNX Runtime, Gymnasium, and operating-system versions used for selection, export, and parity.

The manifest is non-enforcing: `model/policy.py` continues to validate the ONNX bytes, optional expected artifact SHA-256, CPU provider, tensor interface, finite outputs, and action bounds directly. Application readiness does not depend on a manifest signature, source seal, authorization file, or provenance chain. Tests may validate that a published manifest accurately describes its neighboring artifact and parity receipt.

## 5. Integrate and replay through the served path

Publish the selected artifact at `artifacts/city_recovery_ppo.v4.onnx`, which is the application's zero-configuration default. An explicit `-PolicyPath` takes precedence over a nonblank `INNOVERSE_POLICY_PATH`, and either overrides the bundle; `INNOVERSE_POLICY_SHA256` optionally constrains whichever path wins. Invalid explicit choices fail closed instead of falling through. Keep `model/policy.py` as the single CPU inference boundary; do not add a second inference or normalization path in `backend/app/main.py`.

Verify the configured artifact through the same path used by operators:

1. load it through `model.policy.load_policy`;
2. confirm `/health/ready` and `/api/v1/meta` report the configured ONNX identity and 73/22 contract;
3. exercise the comparison endpoint and persisted-result replay with that policy;
4. replay all 200 development cases through the served policy path; and
5. require the served-path per-case outcomes and total solve count to equal the accepted ONNX parity receipt.

The served-path replay proves that the application is using the selected bytes, raw-observation contract, baked normalization, and CPU execution path rather than a direct training-only helper.

## Acceptance gates

Deployment is accepted only when all five gates pass together:

| Gate | Acceptance condition |
| --- | --- |
| Checkpoints | Every selectable milestone has an atomic, hash-bound model/optimizer/normalization bundle that passes fresh-process reload; its resume capability is stated exactly. |
| Tests | The complete Python test suite passes, including ONNX interface, normalization, manifest-consistency, API, persistence, and replay coverage. Ruff remains clean. |
| Application | The configured model passes readiness and metadata checks, a comparison can be persisted and reloaded, and the 200-case served-path development replay matches the accepted artifact. |
| Parity | The full 200-case receipt passes every action, outcome, AUC, determinism, safety, and conservation condition above. |
| Manifest | Every required field is present and matches the selected checkpoint, frozen RMS, ONNX bytes, interface inspection, and parity receipt; the manifest remains descriptive rather than a runtime authorization mechanism. |

Record the commands, environment versions, artifact paths, hashes, observed gate values, and pass/fail result in the deployment report. Stop and report on the first failed gate. Do not replace the selected checkpoint, adjust normalization, relax tolerances, or rewrite a failed receipt inside the same publication attempt.

## Explicit boundaries

This deployment phase did not recreate the `ppo_v3` release ceremony: it added no source seals, semantic source hashes, append-only ledger, write lock, or hash-pinned runtime chain. The later final publisher used a narrow create-new claim/success/failure lifecycle only to bind the owner's one-run authorization, the frozen artifact, and the resulting evidence; it is not a runtime loader gate or a general provenance-enforcement system. Do not extend that single-use mechanism into the retired ceremony.

**Final oracle-solved reference: privileged future-aware CEM found solutions for 182 of 200 final cases; its 18 search failures are not proofs of infeasibility.** The expanded final roster has one retained learned-policy result: the exact shipped v4 artifact solved **163 / 182 = 89.6%** of that oracle-solved reference (descriptive post-hoc Wilson 95% **[0.8427, 0.9321]**), alongside its raw **163 / 200**, and finished **16 cases ahead** of the tuned constant rule at **147 / 200**. Their matched partition is **162 both solved, 1 policy-only, 20 oracle-only, and 17 neither**. Casewise policy coverage is **162 / 182 = 89.0%**, and the two methods jointly demonstrate solutions on **183 / 200** cases; those quantities are distinct because the finite CEM solved set does not contain every policy-solved case. The oracle sees the complete future shock tape and remains a privileged anytime achieved lower bound, not a causal submission baseline, proven ceiling, or mathematical optimum.

The overall final Wilson interval treats the 200 case outcomes as Bernoulli observation units. The roster is clustered within five fixed scenario families of 40 cases, so the interval does not model within-family dependence and may slightly overstate precision; the canonical report includes family-level rows. The five-seed development standard deviations instead measure optimizer-seed variation on one shared development roster and are not final-set sampling uncertainty.

The [owner claim](../internal/evaluation_runs/v4/final-evaluation-200.claim.json) and machine success receipt preserve the single-use evidence chain. Further learned-policy final runs remain unauthorized, including during future checkpoint selection, parity, application integration, served replay, or acceptance testing. Development selection and deployment evidence continue to use only the 200 development cases.
