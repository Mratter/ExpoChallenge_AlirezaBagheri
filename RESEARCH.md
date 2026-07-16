# Research

Research frames the simulator and implementation contract; it supplies no empirical coefficient. Sources were reviewed 2026-07-14. The Feature Complete implementation remains synthetic and non-empirical.

| Primary source | Implemented use | Boundary |
|---|---|---|
| FEMA, [Community Lifelines](https://www.fema.gov/emergency-managers/practitioners/lifelines), US Government work | Essential services are represented as an interdependent stabilization network with visible condition. | The five-resource taxonomy is an authored simplification, not FEMA's seven lifelines. |
| NIST SP 1190v1, [Community Resilience Planning Guide](https://doi.org/10.6028/NIST.SP.1190v1), 2015, US Government publication | Recovery time, service dependencies, priorities, and resource allocation motivate the inspectable planning workflow. | NIST supplies no scenario, coefficient, reward, or validation claim. |
| Gymnasium, [environment creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) and [seeding utilities](https://gymnasium.farama.org/api/utils/), MIT license | `CityRecoveryEnv` implements bounded observation/action spaces, `reset`, `step`, termination, explicit seed handling, and complete trajectory rendering. | Reproducibility is pinned to the locked versions and repository tests. |
| Stable-Baselines3, [PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and [model export guide](https://stable-baselines3.readthedocs.io/en/master/guide/export.html), MIT license | A real SB3 PPO `MlpPolicy` is trained on CPU and its deterministic policy action is exported. Post-processing stays in the common external projector. | PPO is trained only on authored synthetic scenarios and is not municipal guidance. |
| ONNX Runtime, [Python API](https://onnxruntime.ai/docs/api/python/api_summary.html), MIT license | The frozen ONNX graph is parsed, schema-checked, smoke-run, and served through sequential single-thread `CPUExecutionProvider`. | Missing, corrupt, or incompatible ONNX blocks the product; no alternate inference path exists. |
| OR-Tools, [linear optimization](https://developers.google.com/optimization/lp), Apache-2.0 license | The visible GLOP baseline solves a declared one-day linear recovery objective under the same daily bounds and budget. | It receives current state only and cannot see future shocks. |
| NumPy, [PCG64](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html), BSD-3-Clause | Every complete shock tape and authored family member is generated from an explicit PCG64 seed. | Reproducibility is version-pinned and verified from canonical bytes. |

## Implemented Decisions

- An LLM planner remains rejected because it adds network/model dependencies and weakens deterministic evidence.
- The accepted linear candidate remains preserved as historical Gate 2 evidence. It is never called PPO and is not a runtime fallback.
- OR-Tools is visible in API metadata, result metadata, and every daily baseline record, including objective coefficients and solver status.
- The learned action produces a proposal only. The identical external capped-simplex projector owns hard constraints for learned and baseline plans.
- PyTorch/ONNX parity is measured both before and after projection because continuous-action post-processing is external to the exported graph.
- Evaluation uses complete scenario-family member plus seed as the split unit. No day-level split is allowed.

## Remaining Research Boundary

No licensed real municipal dataset or expert-reviewed calibration establishes that the authored dynamics represent disaster outcomes. Operational use would require domain governance, empirical calibration, distribution-shift study, equity/safety review, and accountable human decision authority. None is implied here.
