import { BarChart3, CheckCircle2, RefreshCw, XCircle } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { shockDisplayName } from '../shockPresentation'
import type { CompareResponse, OfficialOutcome } from '../types'
import { outcomeReasonLabel, serviceLabel } from '../v3ViewModel'
import type { Difficulty, GameMode } from './session'
import { DIFFICULTY_DETAILS, MODE_DETAILS } from './session'
import './run-outcome.css'

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function OutcomeReceipt({ outcome }: { outcome: OfficialOutcome }) {
  return (
    <ul className="official-outcome-checks">
      {Object.entries(outcome.checks).map(([name, passed]) => (
        <li key={name} data-pass={passed}>
          {passed ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
          <span>{outcomeReasonLabel(name)}</span>
        </li>
      ))}
    </ul>
  )
}

export function RunDebriefScreen({
  result,
  mode,
  difficulty,
  playerKicks,
  onOpenToolbox,
  onRestart,
}: {
  result: CompareResponse
  mode: GameMode
  difficulty: Difficulty | null
  playerKicks: number
  onOpenToolbox: () => void
  onRestart: () => void
}) {
  const headingRef = useRef<HTMLHeadingElement>(null)
  useEffect(() => { headingRef.current?.focus() }, [])
  const candidate = result.candidate.absolute_outcome
  const baseline = result.baseline.absolute_outcome
  const shocks = result.shock_schedule.filter((shock) => shock.type)
  const lastDay = result.candidate.trajectory.at(-1)!
  const totalMaterial = result.candidate.trajectory.reduce((sum, day) => sum + day.material_used, 0)
  const totalCrew = result.candidate.trajectory.reduce((sum, day) => sum + day.crew_used, 0)
  const totalPreparedness = result.candidate.trajectory.reduce(
    (sum, day) => sum + day.preparedness_investment.reduce((daySum, value) => daySum + value, 0),
    0,
  )
  const modeLabel = MODE_DETAILS[mode].label
  const difficultyLabel = difficulty ? DIFFICULTY_DETAILS[difficulty].label : 'Custom conditions'
  return (
    <section className="debrief-screen" role="dialog" aria-modal="true" aria-labelledby="debrief-heading">
      <div className="debrief-sheet">
        <header className="debrief-heading">
          <div>
            <p>Official end-of-run receipt</p>
            <h1 id="debrief-heading" ref={headingRef} tabIndex={-1}>PPO V3 {candidate.solved ? 'solved' : 'failed'} the scenario.</h1>
            <span>{modeLabel} · {difficultyLabel} · 30-day protocol</span>
          </div>
          <span className="debrief-condition"><i data-survived={candidate.solved} />{candidate.solved ? 'SOLVED' : 'FAILED'}</span>
        </header>

        <div className="debrief-summary" aria-label="Official run summary">
          <article><span>Shared shocks</span><strong>{shocks.length}</strong><small>{playerKicks} operator-injected before the tail</small></article>
          <article><span>Resilience AUC</span><strong>{percent(candidate.resilience_auc)}</strong><small>official floor {percent(candidate.resilience_auc_floor)}</small></article>
          <article><span>Final resilience</span><strong>{percent(result.candidate.final_resilience)}</strong><small>day 30 weighted service state</small></article>
          <article><span>Critical service-days</span><strong>{candidate.critical_service_days}</strong><small>cap {candidate.critical_service_day_cap}</small></article>
          <article><span>Hard violations</span><strong>{candidate.hard_violation_count}</strong><small>conservation residual {candidate.max_conservation_residual.toExponential(2)}</small></article>
        </div>

        <section className="official-verdict-grid" aria-label="Independent planner verdicts">
          <article data-solved={candidate.solved}>
            <p>PPO V3 / ONNX</p><h2>{candidate.solved ? 'SOLVED' : 'FAILED'}</h2><span>Independent absolute outcome</span><OutcomeReceipt outcome={candidate} />
          </article>
          <article data-solved={baseline.solved}>
            <p>Same tape / same public contract</p><h2>Reactive public heuristic: {baseline.solved ? 'SOLVED' : 'FAILED'}</h2><span>{result.comparison.absolute_outcome_pair.replaceAll('_', ' ')}</span><OutcomeReceipt outcome={baseline} />
          </article>
        </section>

        <section className="debrief-logistics" aria-labelledby="debrief-logistics-heading">
          <div className="debrief-section-heading"><p>Physical ledger</p><h2 id="debrief-logistics-heading">What the PPO recovery operation used</h2></div>
          <div className="debrief-arrival-cost"><span>Thirty-day totals</span><strong>{totalMaterial.toFixed(1)} material · {totalCrew.toFixed(1)} crew</strong><small>{totalPreparedness.toFixed(1)} effective preparedness investment. These values come from the returned engine ledger.</small></div>
          <div className="tail-target-grid">
            {result.services.map((service, index) => (
              <div key={service} data-pass={candidate.target_met_by_service[index]}><span>{serviceLabel(service)}</span><b>{percent(candidate.tail_minimum_services[index])}</b><small>target {percent(candidate.recovery_targets[index])}</small></div>
            ))}
          </div>
          <p className="terminal-stock-line">Terminal depot stock {lastDay.logistics.depot_stock_end.reduce((sum, value) => sum + value, 0).toFixed(1)} · pending arrivals {lastDay.logistics.pending_next_day.reduce((sum, value) => sum + value, 0).toFixed(1)}</p>
        </section>

        <section className="debrief-aar" aria-labelledby="debrief-aar-heading">
          <div className="debrief-section-heading"><p>Shared hazard record</p><h2 id="debrief-aar-heading">Every incident on the tape</h2></div>
          <div className="debrief-aar-list">
            {shocks.length ? shocks.map((shock) => {
              const candidateDay = result.candidate.trajectory[shock.day - 1]
              const baselineDay = result.baseline.trajectory[shock.day - 1]
              return (
                <article key={`${shock.day}-${shock.type}`}>
                  <header><b>Day {shock.day} · {shock.type ? shockDisplayName(shock.type) : 'Clear'}</b><span>{shock.forced ? 'operator-injected' : 'ambient'} · {percent(shock.severity)}</span></header>
                  <p>Same recorded footprint for both planners. End-of-day resilience: PPO {percent(candidateDay.resilience)} · heuristic {percent(baselineDay.resilience)}.</p>
                </article>
              )
            }) : <p>No shocks were recorded on this scenario tape.</p>}
          </div>
        </section>

        <footer className="debrief-actions">
          <button className="debrief-toolbox" type="button" onClick={onOpenToolbox}><BarChart3 size={16} />Inspect all 73 inputs and 22 actions</button>
          <button className="debrief-restart" type="button" onClick={onRestart}><RefreshCw size={15} />Start a new run</button>
        </footer>
      </div>
    </section>
  )
}
