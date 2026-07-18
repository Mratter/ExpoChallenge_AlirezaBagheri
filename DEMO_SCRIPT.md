# Demo Script

## Three-minute game-first walkthrough

1. Open `http://127.0.0.1:4117/#/game`. Introduce Civic Relay as a deterministic synthetic recovery simulation: the candidate is a frozen SB3 PPO exported to checksum-pinned ONNX CPU inference, all computation is local, and the environment is authored rather than empirical.
2. On the start screen choose **Stress Test** and **Moderate**. Explain that Stress Test has six disasters and that difficulty changes the world, not the arsenal. Moderate maps to a 20% ambient shock probability, a `0.10–0.28` severity band, and 180 daily units. Start the run; the initial gesture also permits the subtle procedural audio. Point out the visible sound control.
3. As day playback begins, drag the plate to orbit and scroll to zoom. Show the service strip, city condition, and `Budget today` readout. Explain that the base capacity arrives each day, shocks may reduce that day's amount, and RELAY allocates the full available amount across the five services.
4. Pause on a clear view of the central silo and RELAY orb. Follow a convoy from the silo. When a repair is active, follow its vehicle and crane; otherwise note that the current day record did not trigger rebuilding. State that vehicle frequency follows the returned allocation and rebuild work follows real service recovery. Read the current RELAY line and connect each fact in it to the visible day record.
5. Set severity to `0.31`, drag a disaster card over two districts, and show how the typed impact readout changes. Release it over the city. Briefly show the equivalent touch and keyboard path: select a card, choose a named district, and confirm. Explain that either path strikes overnight at the next day boundary: the client appends one forced shock, repeats the deterministic both-planner comparison, and resumes at the same playback position.
6. Let the restrained impact wave, rubble scatter, smoke, and response traffic play. Show that one Stress Test disaster was consumed only after the returned schedule confirmed the appended event. If a service crosses below `0.12`, call that a stumble; three consecutive days darkens its district. A fall requires food or healthcare below the floor for four consecutive days, or two or more services below it on each of two consecutive days; the cascade pair may change.
7. At the debrief, read the actual disasters endured, worst day, recovery count, final weighted wellbeing, resilience AUC, and survival or fall. Then read the **conventional rule-based planner** counterfactual. Emphasize that it uses the already-returned baseline under the identical kicks and collapse rules; there is never a second rendered city.
8. Choose **Inspect this run in the Analyst Toolbox**. Show that the exact final result opens without a replacement comparison. Use the trajectory view, select a day in the allocation ledger, open Daily audit, then restore a saved result. Close by launching that authored result back into the city view as a custom Sandbox run.

If the selected run remains too healthy to demonstrate dark or fall states within the presentation window, do not imply that either occurred. Use the debrief metrics as returned, or restore a previously persisted synthetic fixture whose inputs and result are visible in the Toolbox.

## Truth anchors for narration

- Player actions are limited to disasters, camera, and time controls; RELAY alone allocates units.
- Every throw is one ordered `forced_shocks` entry followed by a full local comparison. Same seed, scenario, and ordered throw list means the same canonical result.
- Shock visuals, damage states, convoys, repairs, district condition, and RELAY speech are derived from the returned trajectory.
- Both planners receive the same shock schedule and bounds. The conventional planner appears only as a debrief counterfactual in the game and as numeric evidence in the Toolbox.
- Collapse is based only on service condition. Daily capacity is a fully allocated flow, not a stockpile.
- Claims apply to an authored synthetic simulator. Do not present them as geographic, municipal, disaster-response, or empirical validation.

## Automated judge path

With port `4117` free, run:

```powershell
.\scripts\verify.ps1 -Profile cpu
```

The bounded verifier runs backend and frontend tests, builds the compiled app, verifies the frozen bundle, submits one 11-day fixture five times, compares canonical bytes and allocation invariants, restarts the loopback server, restores the persisted result byte-identically, checks invalid input, and stops the process. This is functional evidence; it does not substitute for an independently accepted Presentation or Release review tied to the exact commit.
