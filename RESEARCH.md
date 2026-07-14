# Research

Research was used to frame the interface and future model contract, not to supply empirical coefficients. Accessed 2026-07-14.

| Primary source | What it supports | Boundary |
|---|---|---|
| FEMA, [Community Lifelines](https://www.fema.gov/emergency-managers/practitioners/lifelines), US Government work | Treat essential services as an interdependent stabilization network and keep service condition visible. | The Gate 2 five-service taxonomy is a deliberate simplification and does not reproduce FEMA's seven lifelines. |
| NIST SP 1190v1, [Community Resilience Planning Guide](https://doi.org/10.6028/NIST.SP.1190v1), 2015, US Government publication | Recovery time, dependencies, priorities, and resource allocation are legitimate planning concerns. | NIST supplies no coefficient used here; the simulator must not be presented as NIST validated. |
| Gymnasium, [seeding utilities](https://gymnasium.farama.org/api/utils/), MIT license | A non-negative seed should create an explicit NumPy generator. | This slice uses NumPy `Generator(PCG64)` directly and emits the generator name and schedule hash. |
| Stable-Baselines3, [model export guide](https://stable-baselines3.readthedocs.io/en/master/guide/export.html), MIT license, master docs reviewed | Defines the planned PPO-to-ONNX handoff and warns that continuous action post-processing remains external. | SB3 is not installed at Gate 2 and no PPO claim is made. The common projector will remain outside the learned policy. |
| ONNX Runtime, [Python API](https://onnxruntime.ai/docs/api/python/api_summary.html), MIT license | Provides the planned CPU/GPU inference boundary via `InferenceSession` and explicit execution providers. | ONNX Runtime is not installed until a trained artifact and parity tests exist. |
| NumPy, [PCG64](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html), BSD-3-Clause | Defines the concrete seeded generator committed in the API contract. | Reproducibility is pinned to the locked NumPy version and verified from canonical response bytes. |

## Rejected Alternatives

- An LLM planner was rejected: it would add network/model dependencies and weaken deterministic constraint evidence.
- A hidden optimizer baseline was rejected: judges need a visible urgency equation and inspectable daily allocations.
- Calling the current artifact "trained PPO" was rejected: it is a 56-candidate deterministic grid selection over five synthetic calibration scenarios.
- Real municipal data was deferred: no licensed, fit-for-purpose dataset with defensible service-condition labels was established for Gate 2.

## Next Research Gate

Before PPO training, freeze a Gymnasium environment specification, pre-register scenario-family splits, establish empirical or expert-reviewed coefficient provenance, train with SB3 under multiple seeds, export deterministic inference to ONNX, and prove PyTorch/ONNX action parity before and after the shared projector.
