# Tester Feedback

- Project: AI17 - Civic Relay / Autonomous City Recovery Planner
- Tester canonical ID: `/root/ai17_tester`
- Builder commit: `4e4e5cadedeed319eefe703dfdae612d51fced5a`
- Test round: 1
- Recommendation: REVISE
- Exact commands and environments:
  - Windows 11 x64, PowerShell 7.6.3, `uv 0.11.24`, Node `v24.15.0`, npm/npx `11.12.1`, locked CPython 3.12 CPU profile; branch `main`, origin `https://github.com/Mratter/innoverse-ai17-city-recovery.git`, and `origin/main` contains the reviewed commit.
  - `./scripts/setup.ps1 -Profile cpu`
  - `./scripts/preflight.ps1 -Profile cpu -Full`
  - `./scripts/verify.ps1 -Profile cpu`
  - `./.venv/Scripts/python.exe output/tester/contract_probe.py --save-response output/tester/unseen-before-restart.json`, then after killing the listener and restarting with `./scripts/run.ps1 -Profile cpu`: `./.venv/Scripts/python.exe output/tester/contract_probe.py --compare-response output/tester/unseen-before-restart.json`.
  - Fixed-port, child-failure, missing/corrupt dependency, live readiness, and loopback probes used `Get-NetTCPConnection`, `Stop-Process`, temporary `Move-Item`/restore of `artifacts/frozen_policy.v1.json`, `Invoke-WebRequest`, and an occupied `python -m http.server 4117 --bind 127.0.0.1`; transcripts are `output/tester/06-live-missing-artifact.txt` through `output/tester/17-process-cleanup.txt`.
  - Browser: `npx --yes --package @playwright/cli playwright-cli -s=ai17tester open http://127.0.0.1:4117 --browser chrome`, `resize`/`screenshot` at `1440 900`, `1280 720`, and `390 844`, snapshots, keyboard traversal, console/requests inspection, and trace capture. System Chrome was used because the Playwright Chromium CDN returned a regional 403.
  - Accessibility: `node output/tester/axe_audit.cjs` with local `axe-core 4.10.3` and Playwright Core against system Chrome; report `output/tester/axe-report.json`.
- Objective evidence:
  - Setup PASS. Full preflight PASS: artifact smoke hash `23762a44...`, fixture AUCs `0.49401335` and `0.49166123`, backend `7 passed`, Ruff PASS, frontend `2 passed`, TypeScript/Vite build PASS, npm audit zero vulnerabilities. `verify.ps1` PASS with the declared unseen metrics.
  - Independent unseen seed `983471`: five byte-identical responses, response SHA-256 `8d774b31...`, 170 allocation cap/sum checks across five resource classes, five structured invalid cases, and changed seed/input changed the appropriate outputs. The exact response remained byte-identical after restart.
  - `/health/live`, `/health/ready`, and `/api/v1/meta` otherwise exposed live/ready status, commit `4e4e5cadedee`, CPU profile, seed, model/version/hash, and dataset/version/license/synthetic boundary. No job/SSE or persistence contract is declared for this synchronous stateless Gate 2 slice.
  - Occupied port blocked startup with no alternate uvicorn, missing and corrupt artifacts blocked cold startup, killing the actual listener terminated `run.ps1` and its child tree, and port 4117 was released. Connection sampling under 30 live comparisons found a `127.0.0.1:4117` listener and zero external remote addresses.
  - Source-claim audit independently reproduced 56 candidates, five calibration seeds `8100..8104`, winning objective `0.51479517`, frozen weights, 722-byte artifact, SHA-256, and CC0-1.0 manifest claim. Research boundaries consistently disclose the synthetic, non-PPO, non-empirical scope.
- Visual evidence:
  - `output/tester/viewport-1440x900.png`, `output/tester/viewport-1280x720.png`, and `output/tester/viewport-390x844-full.png` show no overlap or horizontal overflow; fresh success sessions had zero console errors and all requests were loopback.
  - `output/tester/invalid-stale-mobile.png` and the Playwright snapshot show a blocking invalid-input alert coexisting with the prior successful trajectory and hashes.
  - Trace: `.playwright-cli/traces/trace-1784033418978.trace`; axe: `output/tester/axe-report.json`; target-size audit: `output/tester/layout_audit.js` output.
- Blocking issues (maximum five):
  1. Live loss of the required policy is not normalized or fail-closed. Removing the artifact after startup makes `/health/ready` and `/api/v1/meta` return plain-text HTTP 500, because `sha256_file(POLICY_PATH)` at `backend/app/artifact.py:32` can raise outside the translated error path; `/` still returns 200. Corrupt bytes correctly return structured 503 from API endpoints, but `/` still serves the primary UI. This fails the required missing/corrupt dependency readiness and no-degraded-primary-UI contract. Evidence: `output/tester/06-live-missing-artifact.txt` and `07-live-corrupt-artifact.txt`.
  2. Invalid/dependency errors retain stale success evidence. `frontend/src/App.tsx:293` sets the error without clearing the result set at line 289, while the result renders independently from line 350. A Days=31 request shows `Comparison blocked` above the previous 14-day metrics, chart, allocations, and hashes, so an error response visibly coexists with a success payload. Evidence: `output/tester/invalid-stale-mobile.png`.
  3. The judge-visible hard-constraint proof is hardcoded and disappears on mobile. `frontend/src/App.tsx:357` renders literal `0 violations` instead of deriving candidate/baseline measured totals; `frontend/src/styles.css:205` hides that text at 390px, leaving an unlabeled shield. This is not trustworthy computed proof and fails the no-hardcoded-output/proof-legibility requirement.
  4. Full preflight does not check all acceptance inputs. `scripts/preflight_check.py:18-28` checks the checksum, one smoke fixture, trajectory length/sums, and a forced shock, but does not validate manifest license/bytes/path/source, policy schema/feature order, or exposed dataset/model schema/version metadata. The current files independently audit correctly, but `preflight.ps1 -Full` cannot detect the license/schema drift that the acceptance matrix requires it to check.
  5. Accessibility fails the UI Constitution. Axe reports a serious `color-contrast` violation at all three required viewports (10 nodes, ratios 3.43-4.18), including service/ledger headers, inactive tab, and shock labels. The layout audit also found 21 visible controls below 40px at 390x844, including 34px service inputs, 36px tabs, and an 18px checkbox (`frontend/src/styles.css:70,85,120`).
- Residual risks:
  - This was not an empty-cache clean Windows install and outbound firewall blocking was not available; Gate 5 clean-machine/offline enforcement remains unproven. Runtime source and sampled sockets were loopback-only.
  - FEMA returned 403 and the NIST DOI could not be fetched through the browser tool; Gymnasium, SB3, ONNX Runtime, and NumPy official documentation resolved and supported the narrow claims. No empirical coefficient claim depends on either unavailable page.
  - Repository-registry matching was not checked because the registry was outside the tester's permitted neutral inputs; origin/main and reviewed-commit containment were checked directly.
- Next required builder state: Commit a clean revision that fixes all five blockers, preserves deterministic metrics and artifact hashes or documents intentional revisions, expands automated coverage for live missing-artifact and stale-result UI behavior, and returns to this same AI17 tester for a full re-test.

The tester changes no implementation file and does not issue the global judge verdict.
