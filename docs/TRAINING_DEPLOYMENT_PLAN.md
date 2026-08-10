# Training deployment plan

This document records the gated path from durable training checkpoints to a served model. It is a plan, not an authorization to run compute or publish a model. Stage 0 is a pre-training prerequisite that must be implemented and verified before the next owner-authorized training run. Stages 1–5 begin only after the owner identifies that run's completed receipt and explicitly authorizes checkpoint selection and deployment work.

The plan therefore has one pre-training stage and five ordered post-training stages: select on development cases, export the selected actor, prove SB3-to-ONNX parity, publish descriptive metadata, and connect the selected artifact to the application. Every performance decision in Stages 1–5 uses the 40 development cases.

## 0. Make milestone checkpoints durable before training

The current `scripts/train_policy.py` writes evaluation rows and state digests to its receipt, but it does not persist model weights or the observation-normalization state. Checkpoint selection and export are therefore blocked until the trainer can publish a complete artifact bundle at every registered development milestone and at the terminal budget.

Each bundle must be written through temporary files, flushed, atomically renamed, and then bound into the training receipt by path and SHA-256. It must contain:

- the complete Stable-Baselines3 policy and value state;
- optimizer state, `num_timesteps`, and schedule/progress state needed to interpret the checkpoint;
- the exact frozen observation RMS mean, variance, count, epsilon, and clipping settings;
- reward-normalization state needed to continue training, kept separate from deployable observation preprocessing;
- training configuration, seed set, completed transition count, and milestone ID; and
- a canonical bundle manifest covering every file hash and the observation RMS digest.

Before authorized training compute, add a fresh-process round-trip test that reloads a synthetic checkpoint bundle and proves byte-identical policy hashes, optimizer-state hashes, counters, observation RMS digest, and deterministic actions on fixed raw observations. Define continuation semantics explicitly: a bundle is always valid for selection, evaluation, and export; it is called bit-exact resumable only if vector-environment lane state, lane RNG state, and in-progress normalization returns are also captured and verified. Otherwise the receipt must label resume as non-bit-exact and must not present a restarted continuation as the uninterrupted registered run.

## 1. Select one checkpoint on development solve rate

Evaluate every complete candidate checkpoint from the authorized training run on the same 40 development cases and deterministic disaster tapes. Use the deterministic actor, its matching frozen observation-normalization state, and the canonical outcome implementation.

Rank checkpoints by:

1. development cases solved out of 40;
2. mean resilience AUC if solve counts tie;
3. mean minimum assessment-tail margin if the first two values tie; and
4. earlier transition count if every measured performance value ties.

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

Run matched SB3 and ONNX candidate rollouts on all 40 development cases. Each implementation receives the same raw observation at each step: the SB3 reference applies the selected frozen VecNormalize state, while the ONNX graph applies the baked transform. Compare every one of the 22 action elements across all 30 days and 40 cases, for 1,200 action samples in total.

The parity receipt must include checkpoint, ONNX, and RMS hashes; interface inspection; per-case rows; sample counts; and observed maxima. It passes only when all of these conditions hold:

- maximum elementwise absolute SB3-versus-ONNX action error is at most `1e-5`, with `rtol=0`;
- every action from both implementations is finite and within `[-1, 1]`;
- every case has the same solved/failed result under SB3 and ONNX;
- the aggregate development solve counts are exactly equal;
- maximum per-case resilience-AUC absolute error is at most `1e-6`;
- ONNX replay produces no hard violations and a maximum conservation residual at or below `1e-6`; and
- a second ONNX replay of every case reproduces the first ONNX trajectory digest exactly.

Record the SB3 solve count, ONNX solve count, per-case outcome mismatch count, maximum action error, maximum resilience-AUC error, replay mismatch count, hard-violation count, maximum conservation residual, and a canonical hash of the 40 parity rows. Any failed condition stops deployment; it is not rounded away or accepted through an aggregate-only comparison.

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

Configure the application with the selected artifact through `INNOVERSE_POLICY_PATH` and its optional `INNOVERSE_POLICY_SHA256`. Keep `model/policy.py` as the single CPU inference boundary; do not add a second inference or normalization path in `backend/app/main.py`.

Verify the configured artifact through the same path used by operators:

1. load it through `model.policy.load_policy`;
2. confirm `/health/ready` and `/api/v1/meta` report the configured ONNX identity and 73/22 contract;
3. exercise the comparison endpoint and persisted-result replay with that policy;
4. replay all 40 development cases through the served policy path; and
5. require the served-path per-case outcomes and total solve count to equal the accepted ONNX parity receipt.

The served-path replay proves that the application is using the selected bytes, raw-observation contract, baked normalization, and CPU execution path rather than a direct training-only helper.

## Acceptance gates

Deployment is accepted only when all five gates pass together:

| Gate | Acceptance condition |
| --- | --- |
| Checkpoints | Every selectable milestone has an atomic, hash-bound model/optimizer/normalization bundle that passes fresh-process reload; its resume capability is stated exactly. |
| Tests | The complete Python test suite passes, including ONNX interface, normalization, manifest-consistency, API, persistence, and replay coverage. Ruff remains clean. |
| Application | The configured model passes readiness and metadata checks, a comparison can be persisted and reloaded, and the 40-case served-path development replay matches the accepted artifact. |
| Parity | The full 40-case receipt passes every action, outcome, AUC, determinism, safety, and conservation condition above. |
| Manifest | Every required field is present and matches the selected checkpoint, frozen RMS, ONNX bytes, interface inspection, and parity receipt; the manifest remains descriptive rather than a runtime authorization mechanism. |

Record the commands, environment versions, artifact paths, hashes, observed gate values, and pass/fail result in the deployment report. Stop and report on the first failed gate. Do not replace the selected checkpoint, adjust normalization, relax tolerances, or rewrite a failed receipt inside the same publication attempt.

## Explicit boundaries

This phase does not recreate the `ppo_v3` release ceremony. Do not add source seals, semantic source hashes, preregistration files, training or final authorization tokens, write-once receipts, append-only ledgers, write locks, or a hash-pinned runtime chain. The selected ONNX hash, checkpoint hash, RMS hash, parity receipt, and lightweight manifest provide the necessary technical traceability without turning deployment metadata into an enforcement system.

The legacy command `scripts/evaluate.py --split final` and its `14 / 30 / 25` result are a separate regression gate. It must not run without explicit final authorization from the owner, including during checkpoint selection, ONNX parity, application integration, served-path replay, or acceptance testing. Development selection and deployment evidence use only the 40 development cases.
