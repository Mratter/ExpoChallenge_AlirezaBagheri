import { ArrowRight, BarChart3, RefreshCw } from 'lucide-react'
import type { Difficulty, GameMode } from './session'
import { DIFFICULTY_DETAILS, MODE_DETAILS } from './session'
import type { CityOutcome, FallCause, RunDebrief } from './stakes'
import { SERVICE_LABELS } from './model'
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
  if (!fall) return null
  return (
    <section className="collapse-screen" role="dialog" aria-modal="true" aria-labelledby="collapse-heading">
      <div className="collapse-haze" aria-hidden="true"><i /><i /><i /></div>
      <div className="collapse-card">
        <p>RELAY / RUN TERMINATED</p>
        <h1 id="collapse-heading">The city fell on day {fall.day}.</h1>
        <div className="collapse-rule" aria-hidden="true" />
        {fall.causes.map((cause) => <span key={cause.kind}>{fallCauseLine(cause)}</span>)}
        <small>The trajectory stops here. The collapse is measured from city condition only.</small>
        <button type="button" onClick={onDebrief}>
          Review what happened <ArrowRight size={17} aria-hidden="true" />
        </button>
      </div>
    </section>
  )
}

function OutcomeStatement({ outcome }: { outcome: CityOutcome }) {
  if (outcome.fall) return <>{`Fell on day ${outcome.fall.day}`}</>
  return <>Held through day {outcome.terminalDay}</>
}

export function RunDebriefScreen({
  debrief,
  mode,
  difficulty,
  playerKicks,
  onOpenToolbox,
  onRestart,
}: {
  debrief: RunDebrief
  mode: GameMode
  difficulty: Difficulty | null
  playerKicks: number
  onOpenToolbox: () => void
  onRestart: () => void
}) {
  const outcome = debrief.candidate
  const worst = outcome.worstMoment
  const modeLabel = MODE_DETAILS[mode].label
  const difficultyLabel = difficulty ? DIFFICULTY_DETAILS[difficulty].label : 'Custom conditions'
  return (
    <section className="debrief-screen" role="dialog" aria-modal="true" aria-labelledby="debrief-heading">
      <div className="debrief-sheet">
        <header className="debrief-heading">
          <div>
            <p>End-of-run debrief</p>
            <h1 id="debrief-heading"><OutcomeStatement outcome={outcome} /></h1>
            <span>{modeLabel} · {difficultyLabel}</span>
          </div>
          <span className="debrief-condition"><i data-survived={outcome.survived} />{outcome.survived ? 'City standing' : 'City fallen'}</span>
        </header>

        <div className="debrief-summary" aria-label="Run summary">
          <article><span>Disasters endured</span><strong>{debrief.disasters.total}</strong><small>{debrief.disasters.ambient} world · {debrief.disasters.player} player</small></article>
          <article><span>Worst moment</span><strong>{worst ? `Day ${worst.day}` : '—'}</strong><small>{worst ? `${percent(worst.wellbeing)} wellbeing · ${SERVICE_LABELS[worst.weakestService]} weakest` : 'No trajectory days'}</small></article>
          <article><span>Recoveries</span><strong>{outcome.recoveryCount}</strong><small>critical-floor crossings back to safety</small></article>
          <article><span>Final wellbeing</span><strong>{percent(outcome.finalWellbeing)}</strong><small>weighted city condition</small></article>
          <article><span>Resilience AUC</span><strong>{percent(outcome.resilienceAuc)}</strong><small>through the terminal day</small></article>
        </div>

        <article className="counterfactual-card">
          <p>Same kicks · same shock tape</p>
          <h2>conventional rule-based planner</h2>
          <span>{debrief.conventionalCounterfactual}</span>
        </article>

        {outcome.survived && playerKicks >= 4 ? <p className="debrief-closing">The city stands.</p> : null}

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
