# Status

- Phase: Gate 2 tester-blocker repair candidate
- Current gate: Awaiting the AI17 tester's independent round-2 re-test
- Tester history: Round 1 REVISE at builder commit `4e4e5cadedeed319eefe703dfdae612d51fced5a`
- Independent judge verdict: Not recorded
- Runtime: CPU, Python 3.12, one loopback process at `127.0.0.1:4117`
- Default metadata seed: `20260714`
- Policy: checksum-verified deterministic synthetic linear candidate, explicitly not PPO
- Verified base: tester-feedback commit `583ca802a0428467fd083a18b9a504f6e506391e`
- Last updated: 2026-07-14

## Repaired

- Missing, unreadable, byte-drifted, hash-drifted, license-drifted, source-drifted, path-drifted, and schema-drifted policy bundles now become structured `503 DEPENDENCY_NOT_READY` responses.
- Every route except `/health/live`, including `/` and compiled assets, is guarded by the required policy dependency. The primary UI is not served during live dependency failure.
- Invalid, dependency, computation, and network failures clear prior comparison evidence before rendering an error. Dependency failures show `Policy blocked` in the compact header.
- The proof badge reduces candidate and baseline violation totals from every returned daily projection and remains fully labeled at mobile width.
- Full preflight validates manifest path, bytes, hash, license, source and version; policy identity, version, feature order, weights, calibration schema and disclosure; and exposed API/model/dataset version and schema metadata.
- Contrast roles were darkened and every visible button, input, checkbox, range, and tab owns at least a 40 by 40 px target.

## Latest Evidence

- `scripts/setup.ps1 -Profile cpu`: PASS; locked environment installed and production UI built.
- `scripts/preflight.ps1 -Profile cpu -Full`: PASS; 17 backend tests, Ruff, 5 frontend tests, strict TypeScript, and Vite build passed.
- `scripts/verify.ps1 -Profile cpu`: PASS; five identical unseen responses; candidate `0.47511509`, urgency `0.47376384`, schedule `7b590dcb...`.
- Fixed fixture remains candidate `0.49401335`, urgency `0.49166123`, schedule `af3a57e9...`, with zero measured violations for both planners.
- Frozen policy remains 722 bytes with SHA-256 `23762a44d67e83dd487558d595d3d9ed5f5e406915f488a076ac21190ab9a6e3` and license `CC0-1.0`.
- Live missing and corrupt probes: liveness `200`; readiness, metadata, comparison, and `/` each returned JSON `503 DEPENDENCY_NOT_READY`; the restored artifact returned readiness `200`.
- Browser QA at `1440x900`, `1280x720`, and `390x844`: zero axe WCAG A/AA violations, zero sub-40px targets, zero horizontal overflow, no normal-flow console/request errors, visible measured proof, and clean keyboard traversal.
- Ignored QA evidence: `output/repair-qa/qa-report.json`, three viewport screenshots, and mobile invalid/dependency screenshots.
- Verification and QA released port 4117 and left no artifact backup.

## Remaining Gates

The repair still requires the same AI17 tester's independent re-test. Gate 3 requires a pre-registered larger scenario-family evaluation and a real SB3/ONNX policy only after provenance and parity checks. Gate 5 clean-machine installation, outbound-firewall runtime enforcement, and offline release verification remain unproven here.
