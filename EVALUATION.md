# Evaluation

## Pre-Registered Gate 2 Protocol

- Baseline: visible urgency score `w*(1-q)*(2.5 if q<.30 else 1)`, budget-normalized and passed through the shared projector
- Candidate: checksum-verified synthetic linear policy, never adapted on evaluation input
- Primary metric: weighted daily resilience AUC, implemented as the arithmetic mean across days of the priority-normalized dot product with service state
- Hard invariant: zero measured allocation sum/lower/upper violations for both planners
- Determinism: canonical response bytes identical for five repeated unseen requests
- Honest outcome: candidate may win, tie, or lose; the API returns that outcome and signed delta rather than hiding a trade-off

## Fixed Fixture

Seed `424242`, 14 days, budget 180, services `[.34,.26,.41,.38,.30]`, priorities `[1,1.1,1.2,1.4,1]`, shock probability `.20`, severity `.10-.28`, forced utility failure day 5 at `.26`.

| Measurement | Frozen candidate | Urgency baseline |
|---|---:|---:|
| Weighted daily resilience AUC | 0.49401335 | 0.49166123 |
| Measured constraint violations | 0 | 0 |

Signed AUC delta is `+0.00235212`; shared shock-tape SHA-256 is `af3a57e9b378700a49a2da8d2042ebc9eb08178cc525cad93f4954306ae5ec81`.

## Unseen Runtime Fixture

`scripts/verify_runtime.py` uses seed `118773`, an 11-day new state/budget/priority combination, and a forced day-7 weather shock. Across five live HTTP calls, canonical bytes were identical. Candidate AUC was `0.47511509`, urgency AUC `0.47376384`, delta `+0.00135125`, zero measured constraint violations, shock hash `7b590dcb045b5ca4026ba77a725e2631f70399be8b9e1433216ff325330a6c22`.

## Failure Evidence

- Horizon 31 returns HTTP 422 `INVALID_SCENARIO` with field details.
- A mocked corrupt artifact returns HTTP 503 `DEPENDENCY_NOT_READY`.
- An occupied fixed port blocks preflight.
- A missing compiled frontend blocks runtime import when `INNOVERSE_RUNTIME=1`.

## Limitation

The small synthetic holdout demonstrates computation, determinism, constraint enforcement, and comparison plumbing only. It does not establish statistical superiority, causal benefit, calibration to disaster outcomes, equity, cost effectiveness, or suitability for municipal decisions.
