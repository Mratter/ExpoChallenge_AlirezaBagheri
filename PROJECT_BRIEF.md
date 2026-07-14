# Project Brief

- Challenge: AI17
- Product: Civic Relay / Autonomous City Recovery Planner
- Audience: competition judges first; municipal resilience analysts as the design reference
- Primary job: compare two recovery allocation plans on a newly authored bounded scenario
- Runtime: React/Vite compiled into a Python 3.12 FastAPI process
- Bind: `127.0.0.1:4117`
- Default metadata seed: `20260714`
- Gate 2 fixture seed: `424242`
- Runtime network: loopback only
- Production mocks and silent fallback: forbidden

## Falsifiable Thesis

For a fixed synthetic city state, budget, priority vector, and PCG64 shock tape, the frozen deterministic policy candidate can produce a different allocation trajectory than a visible urgency baseline while both satisfy the same daily capped-simplex constraints. The claim fails if the response changes across identical runs, either planner receives a different shock, any allocation violates its daily sum/lower/upper bound, or the candidate's measured resilience AUC does not match the reported comparison.

This is a simulator thesis, not a claim of real-world recovery effectiveness. All coefficients and calibration scenarios are synthetic and non-empirical.

## Gate 2 Scope

The implemented slice accepts five ordered services (`transport`, `housing`, `food`, `healthcare`, `public_services`), generates the complete shock schedule once, runs both planners, and returns every daily shock, proposal, projected allocation, state transition, and metric. A checksum-verified linear policy candidate proves the artifact interface. It is explicitly not PPO; SB3 training, held-out evaluation at scale, and ONNX export are later gates.

## Done At This Gate

`setup.ps1`, full preflight, runtime, backend/frontend tests, production build, and five-repeat unseen verification pass on CPU. Gate 3-5 work remains and no independent judge verdict is asserted here.
