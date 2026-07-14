# Tester Feedback

- Project: AI17 - Civic Relay / Autonomous City Recovery Planner
- Tester canonical ID: `/root/ai17_tester`
- Builder commit: `0d59e9b842ec3a60b93e4f46a81594b66c88b220`
- Test round: 2
- Recommendation: PASS
- Exact commands and environments:
  - Windows 11 x64, PowerShell 7.6.3, `uv 0.11.24`, Node `v24.15.0`, npm/npx `11.12.1`, locked CPython 3.12 CPU profile; clean `main` at the reviewed commit and `origin/main` contains it.
  - `./scripts/setup.ps1 -Profile cpu`
  - `./scripts/preflight.ps1 -Profile cpu -Full`
  - `./scripts/verify.ps1 -Profile cpu`
  - Nine-mutation drift probe using `load_policy_bundle` and `validate_exposed_metadata`; transcript: `output/tester-round2/04-preflight-drift.txt`.
  - `./.venv/Scripts/python.exe output/tester/contract_probe.py --save-response output/tester-round2/unseen-before-restart.json`, followed after process death/restart by `--compare-response output/tester-round2/unseen-before-restart.json`.
  - Live dependency probes temporarily moved or replaced `artifacts/frozen_policy.v1.json`, queried live/ready/meta/compare/root with `Invoke-WebRequest`, and restored SHA-256 `23762a44...`; evidence: `output/tester-round2/07-live-missing-artifact.txt` and `08-live-corrupt-artifact.txt`.
  - Browser: Playwright CLI named sessions with system Chrome, snapshots/tracing, `resize` and screenshots at 1440x900, 1280x720, and 390x844, response interception for nonzero measured violations, invalid/dependency flows, console/requests, and keyboard traversal. Trace: `.playwright-cli/traces/trace-1784040006352.trace`.
  - Accessibility: `axe-core 4.10.3` through `node output/tester-round2/axe_audit.cjs`; layout audit via `playwright-cli run-code --filename output/tester/layout_audit.js` at all three viewports.
  - Fixed-port and process tests used an occupied `python -m http.server 4117 --bind 127.0.0.1`, `Get-NetTCPConnection`, listener `Stop-Process`, descendant inspection, restart comparison, and 30-request socket sampling.
- Objective evidence:
  - Setup PASS. Full preflight PASS: artifact path/source/license/bytes and manifest/policy/API/model/dataset schema metadata matched; backend `17 passed`, Ruff PASS, frontend `5 passed`, TypeScript/Vite build PASS, npm audit reported zero vulnerabilities. `verify.ps1` PASS with the unchanged declared unseen metrics.
  - Drift regression PASS: manifest license/path/bytes/source, policy version/feature order, and API/model/dataset schema mutations were all rejected (9/9).
  - Live dependency regression PASS: with either a missing or corrupt policy, `/health/live` remained JSON 200 while `/health/ready`, `/api/v1/meta`, `/api/v1/simulations/compare`, and `/` each returned structured JSON 503 `DEPENDENCY_NOT_READY`; the root response contained no Civic Relay UI. Readiness returned 200 after byte-exact restoration.
  - UI state regression PASS: a real Days=31 response left only the labeled `Scenario invalid` alert; a real corrupt-policy response left only `Comparison blocked` plus mobile-visible `Policy blocked`. Neither state retained comparison summary, trajectory, violation proof, shock tape, or policy hash.
  - Constraint-proof regression PASS: an intercepted real response with candidate daily violations 2 and baseline 1 rendered `Candidate 2 / baseline 1` and accessible label `Measured constraint violations: candidate 2, baseline 1` at all three required viewports, proving totals derive from daily response fields.
  - Accessibility/responsive regression PASS: axe reported 0 violations at 1440x900, 1280x720, and 390x844; layout audits reported no visible control below 40px, no element outside the viewport, and document widths exactly equal to viewport widths. Keyboard Tab traversal reached all critical controls with visible focus; fresh success sessions had 0 console errors/warnings and only loopback requests.
  - Determinism/runtime regression PASS: unseen seed `983471` produced five byte-identical responses, 170 independent cap/sum checks, five structured invalid cases, changed seed/input sensitivity, and response SHA-256 `8d774b31...` before and after restart. Port collision selected no alternate uvicorn, listener death terminated the full run child tree, 30-request sampling found zero external remotes, and port 4117 was released.
- Visual evidence:
  - `output/tester-round2/viewport-1440x900.png`, `viewport-1280x720.png`, and `viewport-390x844.png`: no overlap/clipping/overflow; measured candidate/baseline totals remain legible, including mobile.
  - `output/tester-round2/invalid-cleared-mobile.png` and `dependency-cleared-mobile.png`: blocking states replace all prior result evidence.
  - `output/tester-round2/axe-report.json`, `layout-1440x900.txt`, `layout-1280x720.txt`, `layout-390x844.txt`, and `13-constraint-response.txt` provide machine-readable accessibility, sizing, and computed-proof evidence.
- Blocking issues (maximum five): None.
- Residual risks:
  - This was not an empty-cache clean Windows install, and OS-level outbound firewall blocking was not available. Gate 5 clean-machine/offline enforcement remains unproven; source inspection, browser traffic, and sampled runtime sockets were loopback-only.
  - The repository registry/private-visibility advisory encountered `ECONNRESET`; that is network evidence, not a passing registry verification. Local origin configuration, `origin/main`, and reviewed-commit containment were confirmed, but canonical registry matching/private visibility remains for the coordinator or judge to verify.
  - The Playwright browser CDN remains region-blocked, so installed system Chrome was used. This did not affect the application lifecycle or browser assertions.
  - No job/SSE or persistence contract is declared for this synchronous stateless Gate 2 slice; later-gate model training, larger evaluation, clean-machine release, and operational validation remain out of scope.
- Next required builder state: Preserve this clean repair commit for independent global judge review. This tester recommendation does not substitute for the official gate verdict.

The tester changes no implementation file and does not issue the global judge verdict.
