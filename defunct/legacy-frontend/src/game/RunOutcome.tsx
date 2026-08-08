import { ArrowRight, BarChart3, RefreshCw } from 'lucide-react'
import { useEffect, useRef } from 'react'
import type { Difficulty, GameMode } from './session'
import { DIFFICULTY_DETAILS, MODE_DETAILS } from './session'
import type { CityOutcome, FallCause, RunDebrief } from './stakes'
import { SERVICE_LABELS } from './model'
import { shockDisplayName } from '../shockPresentation'
import './run-outcome.css'

function percent(value: number | null): string {
  return value === null ? '—' : `${Math.round(value * 100)}%`
}

function fallCauseLine(cause: FallCause): string {
  const names = cause.services.map((service) => SERVICE_LABELS[service])
  if (cause.kind === 'essential') {
    return `${names.join(' and ')} remained below the critical floor for four consecutive days.`
  }
  return 'At least two services were below the critical floor on two consecutive days.'
}

export function CollapseScreen({
  outcome,
  onDebrief,
}: {
  outcome: CityOutcome
  onDebrief: () => void
}) {
  const fall = outcome.fall
  const headingRef = useRef<HTMLHeadingElement>(null)
  useEffect(() => {
    if (fall) headingRef.current?.focus()
  }, [fall])
  if (!fall) return null
  return (
    <section className="collapse-screen" role="dialog" aria-modal="true" aria-labelledby="collapse-heading">
      <div className="collapse-haze" aria-hidden="true"><i /><i /><i /></div>
      <div className="collapse-card">
        <p>RELAY / RUN TERMINATED</p>
        <h1 id="collapse-heading" ref={headingRef} tabIndex={-1}>Relay City fell on day {fall.day}.</h1>
        <div className="collapse-rule" aria-hidden="true" />
        {fall.causes.map((cause) => <span key={cause.kind}>{fallCauseLine(cause)}</span>)}
        <small>The trajectory stops here. The collapse is measured from Relay City condition only.</small>
        <button type="button" onClick={onDebrief}>
          Review what happened <ArrowRight size={17} aria-hidden="true" />
        </button>
      </div>
    </section>
  )
}

function OutcomeStatement({ outcome }: { outcome: CityOutcome }) {
  if (outcome.fall) return <>{`Relay City fell on day ${outcome.fall.day}`}</>
  return <>Relay City held through day {outcome.terminalDay}</>
}

export function RunDebriefScreen({
  debrief,
  mode,
  difficulty,
  runLabel,
  playerKicks,
  onOpenToolbox,
  onRestart,
}: {
  debrief: RunDebrief
  mode: GameMode
  difficulty: Difficulty | null
  runLabel?: string
  playerKicks: number
  onOpenToolbox: () => void
  onRestart: () => void
}) {
  const outcome = debrief.candidate
  const headingRef = useRef<HTMLHeadingElement>(null)
  useEffect(() => {
    headingRef.current?.focus()
  }, [])
  const worst = outcome.worstMoment
  const modeLabel = MODE_DETAILS[mode].label
  const difficultyLabel = difficulty ? DIFFICULTY_DETAILS[difficulty].label : 'Custom conditions'
  const conditionLabel = runLabel ?? `${modeLabel} · ${difficultyLabel}`
  return (
    <section className="debrief-screen" role="dialog" aria-modal="true" aria-labelledby="debrief-heading">
      <div className="debrief-sheet">
        <header className="debrief-heading">
          <div>
            <p>End-of-run debrief</p>
            <h1 id="debrief-heading" ref={headingRef} tabIndex={-1}><OutcomeStatement outcome={outcome} /></h1>
            <span>{conditionLabel}</span>
          </div>
          <span className="debrief-condition"><i data-survived={outcome.survived} />{outcome.survived ? 'Relay City standing' : 'Relay City fallen'}</span>
        </header>

        <div className="debrief-summary" aria-label="Run summary">
          <article><span>Disasters endured</span><strong>{debrief.disasters.total}</strong><small>{debrief.disasters.ambient} seeded ambient · {debrief.disasters.authored} authored · {debrief.disasters.player} player{debrief.disasters.storedUnknown ? ` · ${debrief.disasters.storedUnknown} stored origin unknown` : ''}</small></article>
          <article><span>Worst moment</span><strong>{worst ? `Day ${worst.day}` : '—'}</strong><small>{worst ? `${percent(worst.wellbeing)} wellbeing · ${SERVICE_LABELS[worst.weakestService]} weakest` : 'No trajectory days'}</small></article>
          <article><span>Recoveries</span><strong>{outcome.recoveryCount}</strong><small>critical-floor crossings back to safety</small></article>
          <article><span>Final wellbeing</span><strong>{percent(outcome.finalWellbeing)}</strong><small>weighted Relay City condition</small></article>
          <article><span>Resilience AUC</span><strong>{percent(outcome.resilienceAuc)}</strong><small>through the terminal day</small></article>
        </div>

        <article className="counterfactual-card">
          <p>Same kicks · same shock tape</p>
          <h2>conventional rule-based planner</h2>
          <span>{debrief.conventionalCounterfactual}</span>
        </article>

        <section className="debrief-schedule" aria-labelledby="debrief-schedule-heading">
          <div className="debrief-section-heading">
            <p>Schedule disclosure</p>
            <h2 id="debrief-schedule-heading">Returned incidents + authored reconciliation</h2>
          </div>
          <p className="debrief-schedule-note">This ordered record is shown after the run. Overridden authored entries are not counted as strikes; rows after a fall are marked not reached.</p>
          {debrief.schedule.length ? (
            <ol>
              {debrief.schedule.map((entry, index) => (
                <li key={`${entry.day}-${entry.type}-${entry.status}-${index}`} data-status={entry.status}>
                  <b>Day {entry.day}</b>
                  <span>{shockDisplayName(entry.type)} · raw {entry.severity.toFixed(2)}</span>
                  <em>{entry.source}</em>
                  <small>{entry.status === 'overridden'
                    ? 'Overridden before the returned run; this authored entry did not strike.'
                    : entry.status === 'not-reached'
                      ? 'Present in the returned tape, but Relay City fell before this day.'
                      : `Reached · strongest returned footprint ${entry.strongestService ? SERVICE_LABELS[entry.strongestService] : 'not recorded'}.`}</small>
                </li>
              ))}
            </ol>
          ) : <p>The returned shock tape contains no recorded incidents.</p>}
        </section>

        <section className="debrief-logistics" aria-labelledby="debrief-logistics-heading">
          <div className="debrief-section-heading">
            <p>Logistics record</p>
            <h2 id="debrief-logistics-heading">What the recovery operation moved</h2>
          </div>
          <div className="debrief-arrival-cost">
            <span>Cost of disasters</span>
            <strong>{debrief.shockAdjustedArrivalShortfall.toFixed(1)} supply units did not arrive</strong>
            <small>Compared with the calm arithmetic baseline of {debrief.calmArrivalBaseline.toFixed(1)} units through this run. This is an arrival shortfall, not currency or an unspent balance.</small>
          </div>
          <div className="debrief-milestones">
            <h3>Recovery timeline</h3>
            {debrief.milestones.length ? (
              <ol>
                {debrief.milestones.map((milestone) => (
                  <li key={`${milestone.day}-${milestone.label}`}>
                    <b>Day {milestone.day}</b><span>{milestone.label}</span><small>{milestone.source}</small>
                  </li>
                ))}
              </ol>
            ) : <p>No configured reopening threshold was crossed before the terminal day.</p>}
          </div>
        </section>

        <section className="debrief-aar" aria-labelledby="debrief-aar-heading">
          <div className="debrief-section-heading">
            <p>After-action record</p>
            <h2 id="debrief-aar-heading">Each recorded incident</h2>
          </div>
          <div className="debrief-aar-list">
            {debrief.afterActions.map((report) => (
              <article key={`${report.day}-${report.type}`}>
                <header><b>Day {report.day} · {shockDisplayName(report.type)}</b><span>raw {report.severity.toFixed(2)}</span></header>
                <p>
                  Strongest typed footprint: {SERVICE_LABELS[report.strongestService]}. Emergency wave {report.emergencyWaveVehicles} presentation vehicles; line-haul {report.lineHaulHeavyTrucks} heavy-load equivalents; last-mile {report.lastMileVehicles} load equivalents; mutual aid {report.mutualAidVehicles} load equivalents.{' '}
                  {report.logisticsRecorded
                    ? 'Load-equivalent counts are derived from recorded engine-v2 quantities; the bounded road scene shows a deterministic subset.'
                    : 'Load-equivalent counts are allocation-backed presentation; legacy v1 depot operations were not recorded.'}
                </p>
                <dl>
                  {Object.entries(report.recoveryDays).map(([service, days]) => (
                    <div key={service}><dt>{SERVICE_LABELS[service as keyof typeof SERVICE_LABELS]}</dt><dd>{days === null ? 'Not recovered in horizon' : `${days} day${days === 1 ? '' : 's'} to pre-event state`}</dd></div>
                  ))}
                </dl>
              </article>
            ))}
          </div>
        </section>

        {outcome.survived && playerKicks >= 4 ? <p className="debrief-closing">Relay City stands.</p> : null}

        <p className="debrief-disclosure">
          structurally realistic, authored-synthetic, not empirically calibrated to real disasters
        </p>

        <footer className="debrief-actions">
          <button className="debrief-toolbox" type="button" onClick={onOpenToolbox}>
            <BarChart3 size={16} aria-hidden="true" />Inspect this run in the Analyst Toolbox
          </button>
          <button className="debrief-restart" type="button" onClick={onRestart}>
            <RefreshCw size={15} aria-hidden="true" />Start a new run
          </button>
        </footer>
      </div>
    </section>
  )
}
