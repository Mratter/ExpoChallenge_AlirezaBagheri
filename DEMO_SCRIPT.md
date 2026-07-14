# Demo Script

## Three-Minute Gate 2 Walkthrough

1. Start at `http://127.0.0.1:4117`. Point out the local deterministic and synthetic labels; do not describe the candidate as PPO.
2. In Recovery envelope, show five ordered services, their initial state/priority, seed `424242`, 14 days, 180 units, 20% shock chance, and the forced day-5 utility failure.
3. Run comparison. The proof moment is the day-5 ledger: both planners show the exact same utility shock and available budget, different raw/projection allocations, full service end states, and measured zero violations.
4. Read the fixed-fixture AUCs: candidate `49.401335%`, urgency `49.166123%`, a measured `+0.235212` percentage-point delta for this synthetic fixture.
5. Scrub another day and open Daily audit to show that all 14 days are returned, not a prepared endpoint summary.
6. Change one service value or seed and rerun. The shock hash/trajectory changes through real API computation.
7. Close on the footer: the exact shock and policy hashes remain visible, as does "not PPO, not empirically trained, and not operational guidance."

## Automated Judge Path

Run `scripts/verify.ps1 -Profile cpu`. It executes the complete test/build suite, starts the compiled app, submits a different 11-day fixture five times, compares canonical bytes, checks every daily cap/sum, checks shared shock identity, tests invalid input, and stops the process.
