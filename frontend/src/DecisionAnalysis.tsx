import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FlaskConical,
  Leaf,
  LoaderCircle,
  RotateCcw,
} from 'lucide-react'
import {
  AnalysisApiError,
  createCounterfactualRequest,
  fetchDecisionExplanations,
  recoveryPlanUrl,
  runCounterfactualAnalysis,
  type AnalysisPlanner,
  type CounterfactualResponse,
  type ExplanationChannel,
  type ExplanationResponse,
} from './analysisApi'
import { services, type CompareResponse, type DayResult, type Service, type Vector5 } from './types'
import './DecisionAnalysis.css'

export type DecisionAnalysisProps = {
  result: CompareResponse
  resultId: string
  selectedDay: number
  planner: AnalysisPlanner
  onSelectedDayChange: (day: number) => void
  onPlannerChange?: (planner: AnalysisPlanner) => void
}

type ShareInputs = [string, string, string, string, string]

type SustainabilityPoint = {
  day: number
  preparednessMaterial: number
  preparednessCrew: number
  absorbedServicePoints: number
  shockType: string | null
}

const serviceNames: Record<Service, string> = {
  transport: 'Transport',
  housing: 'Housing',
  food: 'Food supply',
  healthcare: 'Healthcare',
  public_services: 'Public services',
}

function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`
}

function signedPercentagePoints(value: number): string {
  const points = value * 100
  return `${points >= 0 ? '+' : ''}${points.toFixed(2)} pp`
}

function units(value: number): string {
  return value.toFixed(1)
}

function humanize(value: string): string {
  return value.replaceAll('_', ' ')
}

function compactEvidence(value: unknown): string {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4)
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (typeof value === 'string') return humanize(value)
  if (value === null) return 'none'
  if (Array.isArray(value)) return value.map(compactEvidence).join(' · ')
  if (typeof value === 'object') {
    return Object.entries(value).map(([key, entry]) => `${humanize(key)}: ${compactEvidence(entry)}`).join(' · ')
  }
  return 'not recorded'
}

function shareInputs(values: Vector5): ShareInputs {
  const total = values.reduce((sum, value) => sum + Math.max(0, value), 0)
  if (total <= 0) return ['20', '20', '20', '20', '20']
  return values.map((value) => (100 * Math.max(0, value) / total).toFixed(4)) as ShareInputs
}

function parsedShareInputs(values: ShareInputs): number[] {
  return values.map((value) => value.trim() === '' ? Number.NaN : Number(value))
}

function updateShareInput(values: ShareInputs, index: number, value: string): ShareInputs {
  const updated = [...values] as ShareInputs
  updated[index] = value
  return updated
}

function topChannels(day: ExplanationResponse['days'][number] | undefined): ExplanationChannel[] {
  if (day === undefined) return []
  return [...day.channels].sort((left, right) => left.influence_rank - right.influence_rank)
}

function strongestChannel(day: ExplanationResponse['days'][number] | undefined): string {
  const strongest = day?.channels.find((channel) => channel.influence_rank === 1)
  return strongest === undefined ? 'Awaiting explanation receipt' : humanize(strongest.observation_name)
}

function heuristicSignal(day: DayResult): string {
  const evidence = day.planner_evidence
  if (evidence === null) return 'No heuristic evidence recorded'
  if (Array.isArray(evidence.expected_public_impact)) {
    const values = evidence.expected_public_impact.filter(
      (value): value is number => typeof value === 'number' && Number.isFinite(value),
    )
    if (values.length === services.length) {
      let highestIndex = 0
      for (let index = 1; index < values.length; index += 1) {
        if (values[index] > values[highestIndex]) highestIndex = index
      }
      return `Highest public impact: ${serviceNames[services[highestIndex]]}`
    }
  }
  return 'Recorded public-state rule evidence'
}

function realizedShockAbsorption(day: DayResult): number {
  let absorbed = 0
  for (let index = 0; index < services.length; index += 1) {
    const unprotected = Math.max(
      0,
      day.services_before[index] * (1 - day.shock.severity * day.shock.impact[index]),
    )
    absorbed += Math.max(0, day.services_after_shock[index] - unprotected)
  }
  return absorbed
}

export function preparednessResourcesForDay(day: DayResult): { material: number; crew: number } {
  return {
    material: day.logistics.preparedness_material_consumed.reduce((sum, value) => sum + value, 0),
    crew: day.logistics.preparedness_crew_utilized.reduce((sum, value) => sum + value, 0),
  }
}

function sustainabilityPoints(result: CompareResponse, planner: AnalysisPlanner): SustainabilityPoint[] {
  return result[planner].trajectory.map((day) => {
    const preparedness = preparednessResourcesForDay(day)
    return {
      day: day.day,
      preparednessMaterial: preparedness.material,
      preparednessCrew: preparedness.crew,
      absorbedServicePoints: realizedShockAbsorption(day),
      shockType: day.shock.type,
    }
  })
}

function PlannerControl({
  planner,
  onPlannerChange,
}: {
  planner: AnalysisPlanner
  onPlannerChange?: (planner: AnalysisPlanner) => void
}) {
  if (onPlannerChange === undefined) {
    return <span className="decision-analysis__planner-label">{planner === 'candidate' ? 'PPO policy' : 'Reactive heuristic'}</span>
  }
  return (
    <div className="decision-analysis__planner" role="group" aria-label="Decision planner">
      <button
        type="button"
        aria-pressed={planner === 'candidate'}
        onClick={() => onPlannerChange('candidate')}
      >
        PPO policy
      </button>
      <button
        type="button"
        aria-pressed={planner === 'baseline'}
        onClick={() => onPlannerChange('baseline')}
      >
        Heuristic
      </button>
    </div>
  )
}

function AttributionPanel({
  selectedDay,
  explanations,
  loading,
  error,
  onRetry,
}: {
  selectedDay: number
  explanations: ExplanationResponse | null
  loading: boolean
  error: string | null
  onRetry: () => void
}) {
  const allRanked = useMemo(
    () => topChannels(explanations?.days[selectedDay - 1]),
    [explanations, selectedDay],
  )
  const topFive = allRanked.slice(0, 5)
  const tableId = `attribution-all-day-${selectedDay}`

  if (loading) {
    return (
      <div className="decision-analysis__status" role="status">
        <LoaderCircle className="decision-analysis__spin" size={17} aria-hidden="true" />
        Replaying the policy to measure 73 local action sensitivities for each day…
      </div>
    )
  }
  if (error !== null) {
    return (
      <div className="decision-analysis__error" role="alert">
        <AlertTriangle size={18} aria-hidden="true" />
        <div><strong>Explanation unavailable</strong><span>{error}</span></div>
        <button type="button" onClick={onRetry}>Retry</button>
      </div>
    )
  }
  if (explanations === null || topFive.length === 0) {
    return <p className="decision-analysis__quiet">No explanation receipt is loaded.</p>
  }

  return (
    <section className="decision-analysis__attribution" aria-labelledby="local-sensitivity-heading">
      <header>
        <div>
          <p className="decision-analysis__eyebrow">Day {selectedDay} / PPO policy</p>
          <h4 id="local-sensitivity-heading">Strongest local action sensitivities</h4>
        </div>
        <span>Top 5 of {explanations.observation_count}</span>
      </header>
      <ol className="decision-analysis__top-signals">
        {topFive.map((channel) => (
          <li key={channel.observation_name}>
            <b>{channel.influence_rank}</b>
            <div>
              <code>{channel.observation_name}</code>
              <small>
                Most affected: {humanize(channel.most_affected_action)} · action Δ{' '}
                {channel.signed_action_delta >= 0 ? '+' : ''}{channel.signed_action_delta.toFixed(5)}
              </small>
            </div>
            <span>{percent(channel.normalized_influence, 2)}</span>
            <meter
              min={0}
              max={1}
              value={channel.normalized_influence}
              aria-label={`${humanize(channel.observation_name)} normalized local influence`}
            />
          </li>
        ))}
      </ol>
      <details className="decision-analysis__all-signals">
        <summary aria-controls={tableId}>Inspect all 73 named observation channels</summary>
        <div
          id={tableId}
          className="decision-analysis__table-scroll"
          role="region"
          tabIndex={0}
          aria-label={`All local action sensitivities for day ${selectedDay}`}
        >
          <table>
            <thead>
              <tr>
                <th scope="col">Rank</th>
                <th scope="col">Observation channel</th>
                <th scope="col">Observed</th>
                <th scope="col">Mean |action Δ|</th>
                <th scope="col">Influence</th>
                <th scope="col">Most affected action</th>
                <th scope="col">Signed Δ</th>
              </tr>
            </thead>
            <tbody>
              {allRanked.map((channel) => (
                <tr key={channel.observation_name}>
                  <td>{channel.influence_rank}</td>
                  <td><code>{channel.observation_name}</code></td>
                  <td>{channel.observed_value.toFixed(5)}</td>
                  <td>{channel.mean_absolute_action_delta.toFixed(7)}</td>
                  <td>{percent(channel.normalized_influence, 3)}</td>
                  <td><code>{channel.most_affected_action}</code></td>
                  <td>{channel.signed_action_delta >= 0 ? '+' : ''}{channel.signed_action_delta.toFixed(7)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <p className="decision-analysis__method-note">
        <strong>How to read this:</strong> {explanations.method.interpretation} The diagnostic
        independently zeros one current observation channel, measures the mean absolute change
        across 22 raw actions, and never reveals the future shock tape.
      </p>
    </section>
  )
}

function HeuristicEvidence({ day }: { day: DayResult }) {
  const entries = day.planner_evidence === null ? [] : Object.entries(day.planner_evidence)
  return (
    <section className="decision-analysis__heuristic" aria-labelledby="heuristic-evidence-heading">
      <header>
        <div>
          <p className="decision-analysis__eyebrow">Day {day.day} / reactive heuristic</p>
          <h4 id="heuristic-evidence-heading">Recorded rule evidence</h4>
        </div>
        <span>Public state only</span>
      </header>
      {entries.length === 0 ? (
        <p className="decision-analysis__quiet">No planner evidence was recorded for this day.</p>
      ) : (
        <dl className="decision-analysis__evidence-list">
          {entries.map(([name, value]) => (
            <div key={name}>
              <dt>{humanize(name)}</dt>
              <dd>{compactEvidence(value)}</dd>
            </div>
          ))}
        </dl>
      )}
      <p className="decision-analysis__method-note">
        These values are the heuristic’s persisted <code>planner_evidence</code>. No PPO-style
        attribution is inferred or fabricated for the rule-based planner.
      </p>
    </section>
  )
}

function WhatIfResult({ result, comparison }: { result: CounterfactualResponse; comparison: CompareResponse }) {
  const originalTailFloor = Math.min(...result.original.absolute_outcome.tail_minimum_services)
  const changedTailFloor = Math.min(...result.counterfactual.absolute_outcome.tail_minimum_services)
  const finalDelta = result.daily_deltas.at(-1)?.services_end ?? [0, 0, 0, 0, 0]
  const originalFinal = comparison.candidate.trajectory.at(-1)?.services_end ?? [0, 0, 0, 0, 0]
  return (
    <section className="decision-analysis__what-if-result" aria-labelledby="what-if-result-heading">
      <header>
        <div>
          <p className="decision-analysis__eyebrow">Analysis-only replay / same disaster tape</p>
          <h4 id="what-if-result-heading">What changed after the one-day override</h4>
        </div>
        <span><CheckCircle2 size={15} aria-hidden="true" /> Prefix days {result.unchanged_prefix.days} verified</span>
      </header>
      <div className="decision-analysis__comparison-cards">
        <div>
          <span>Solved verdict</span>
          <b>{result.original.solved ? 'Solved' : 'Not solved'} → {result.counterfactual.solved ? 'Solved' : 'Not solved'}</b>
          <small>{result.original.solved === result.counterfactual.solved ? 'unchanged' : 'verdict changed'}</small>
        </div>
        <div>
          <span>Resilience AUC</span>
          <b>{result.original.rauc.toFixed(5)} → {result.counterfactual.rauc.toFixed(5)}</b>
          <small>{signedPercentagePoints(result.counterfactual.rauc - result.original.rauc)}</small>
        </div>
        <div>
          <span>Assessment-tail floor</span>
          <b>{percent(originalTailFloor, 2)} → {percent(changedTailFloor, 2)}</b>
          <small>{signedPercentagePoints(changedTailFloor - originalTailFloor)}</small>
        </div>
        <div>
          <span>Minimum resilience</span>
          <b>{percent(result.original.minimum_resilience, 2)} → {percent(result.counterfactual.minimum_resilience, 2)}</b>
          <small>{signedPercentagePoints(result.counterfactual.minimum_resilience - result.original.minimum_resilience)}</small>
        </div>
      </div>
      <div className="decision-analysis__table-scroll" role="region" tabIndex={0} aria-label="Final service comparison after the what-if replay">
        <table>
          <thead><tr><th scope="col">Service</th><th scope="col">Original final</th><th scope="col">What-if final</th><th scope="col">Delta</th></tr></thead>
          <tbody>
            {services.map((service, index) => (
              <tr key={service}>
                <th scope="row">{serviceNames[service]}</th>
                <td>{percent(originalFinal[index], 2)}</td>
                <td>{percent(originalFinal[index] + finalDelta[index], 2)}</td>
                <td className={finalDelta[index] >= 0 ? 'decision-analysis__positive' : 'decision-analysis__negative'}>
                  {signedPercentagePoints(finalDelta[index])}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="decision-analysis__method-note">
        The selected day is changed, the original PPO policy resumes on the following day, and the
        derived trajectory is not persisted as an official comparison.
      </p>
    </section>
  )
}

function CounterfactualPanel({
  result,
  selectedDay,
}: {
  result: CompareResponse
  selectedDay: number
}) {
  const formId = useId()
  const errorRef = useRef<HTMLDivElement>(null)
  const requestController = useRef<AbortController | null>(null)
  const selectedCandidate = result.candidate.trajectory[selectedDay - 1]
  const [material, setMaterial] = useState<ShareInputs>(() => shareInputs(selectedCandidate.material_allocation))
  const [crew, setCrew] = useState<ShareInputs>(() => shareInputs(selectedCandidate.crew_allocation))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<CounterfactualResponse | null>(null)

  const reset = useCallback(() => {
    setMaterial(shareInputs(selectedCandidate.material_allocation))
    setCrew(shareInputs(selectedCandidate.crew_allocation))
    setError(null)
    setAnalysis(null)
  }, [selectedCandidate])

  useEffect(() => {
    requestController.current?.abort()
    setBusy(false)
    reset()
  }, [reset, result.result_id, selectedDay])

  useEffect(() => () => requestController.current?.abort(), [])

  useEffect(() => {
    if (error !== null) errorRef.current?.focus()
  }, [error])

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError(null)
    setAnalysis(null)
    try {
      const request = createCounterfactualRequest(selectedDay, {
        materialShares: parsedShareInputs(material),
        crewShares: parsedShareInputs(crew),
      })
      requestController.current?.abort()
      const controller = new AbortController()
      requestController.current = controller
      setBusy(true)
      const response = await runCounterfactualAnalysis(result, request, controller.signal)
      if (!controller.signal.aborted) setAnalysis(response)
    } catch (caught) {
      if (requestController.current?.signal.aborted) return
      setError(caught instanceof Error ? caught.message : 'The counterfactual replay failed.')
    } finally {
      if (!requestController.current?.signal.aborted) setBusy(false)
    }
  }

  const invalid = error !== null
  const helpId = `${formId}-shares-help`
  const errorId = `${formId}-shares-error`
  return (
    <section className="decision-analysis__what-if" aria-labelledby={`${formId}-heading`}>
      <header>
        <div>
          <p className="decision-analysis__eyebrow">Counterfactual / candidate policy</p>
          <h4 id={`${formId}-heading`}>Override day {selectedDay}, then let PPO resume</h4>
        </div>
        <FlaskConical size={20} aria-hidden="true" />
      </header>
      <form onSubmit={(event) => void submit(event)} aria-busy={busy}>
        <p id={helpId} className="decision-analysis__form-help">
          Enter non-negative relative weights. Each row is normalized to 100%; material and crew
          totals do not need to add up. Reset preserves the executed allocation as an exact no-op;
          changed weights still pass through the policy's bounded decoder and feasibility projection.
        </p>
        <div className="decision-analysis__share-grid">
          <fieldset disabled={busy}>
            <legend>Material shares</legend>
            {services.map((service, index) => (
              <label key={service} htmlFor={`${formId}-material-${service}`}>
                <span>{serviceNames[service]}</span>
                <input
                  id={`${formId}-material-${service}`}
                  type="number"
                  inputMode="decimal"
                  min="0"
                  step="any"
                  value={material[index]}
                  aria-invalid={invalid}
                  aria-describedby={`${helpId}${invalid ? ` ${errorId}` : ''}`}
                  onChange={(event) => setMaterial((current) => updateShareInput(current, index, event.target.value))}
                />
              </label>
            ))}
          </fieldset>
          <fieldset disabled={busy}>
            <legend>Crew shares</legend>
            {services.map((service, index) => (
              <label key={service} htmlFor={`${formId}-crew-${service}`}>
                <span>{serviceNames[service]}</span>
                <input
                  id={`${formId}-crew-${service}`}
                  type="number"
                  inputMode="decimal"
                  min="0"
                  step="any"
                  value={crew[index]}
                  aria-invalid={invalid}
                  aria-describedby={`${helpId}${invalid ? ` ${errorId}` : ''}`}
                  onChange={(event) => setCrew((current) => updateShareInput(current, index, event.target.value))}
                />
              </label>
            ))}
          </fieldset>
        </div>
        {error === null ? null : (
          <div id={errorId} className="decision-analysis__form-error" ref={errorRef} tabIndex={-1} role="alert">
            <AlertTriangle size={16} aria-hidden="true" />{error}
          </div>
        )}
        <div className="decision-analysis__form-actions">
          <button type="button" onClick={reset} disabled={busy}>
            <RotateCcw size={15} aria-hidden="true" /> Reset day {selectedDay}
          </button>
          <button type="submit" className="decision-analysis__primary" disabled={busy}>
            {busy
              ? <><LoaderCircle className="decision-analysis__spin" size={15} aria-hidden="true" /> Replaying 30 days…</>
              : <><FlaskConical size={15} aria-hidden="true" /> Run one-day what-if</>}
          </button>
        </div>
      </form>
      <div aria-live="polite">
        {analysis === null ? null : <WhatIfResult result={analysis} comparison={result} />}
      </div>
    </section>
  )
}

function SustainabilityPanel({ result, planner }: { result: CompareResponse; planner: AnalysisPlanner }) {
  const headingId = useId()
  const points = useMemo(() => sustainabilityPoints(result, planner), [planner, result])
  const totalPreparednessMaterial = points.reduce((sum, point) => sum + point.preparednessMaterial, 0)
  const totalPreparednessCrew = points.reduce((sum, point) => sum + point.preparednessCrew, 0)
  const totalAbsorbed = points.reduce((sum, point) => sum + point.absorbedServicePoints, 0)
  const shockDays = points.filter((point) => point.shockType !== null).length
  const peakAbsorption = points.reduce(
    (peak, point) => point.absorbedServicePoints > peak.absorbedServicePoints ? point : peak,
    points[0],
  )
  const maxPreparednessMaterial = Math.max(...points.map((point) => point.preparednessMaterial), 1e-12)
  const maxPreparednessCrew = Math.max(...points.map((point) => point.preparednessCrew), 1e-12)
  const maxAbsorption = Math.max(...points.map((point) => point.absorbedServicePoints), 1e-12)

  return (
    <section className="decision-analysis__sustainability" aria-labelledby={headingId}>
      <header>
        <div>
          <p className="decision-analysis__eyebrow">30-day sustainability evidence</p>
          <h4 id={headingId}>Preparedness invested vs shock impact absorbed</h4>
        </div>
        <Leaf size={20} aria-hidden="true" />
      </header>
      <div className="decision-analysis__sustainability-summary">
        <div><span>Prep material consumed</span><b>{units(totalPreparednessMaterial)}</b><small>material units</small></div>
        <div><span>Prep crew utilized</span><b>{units(totalPreparednessCrew)}</b><small>crew units</small></div>
        <div><span>Impact absorbed</span><b>{(totalAbsorbed * 100).toFixed(2)}</b><small>service-point days · peak day {peakAbsorption.day}</small></div>
        <div><span>Shock days observed</span><b>{shockDays}</b><small>same recorded tape</small></div>
      </div>
      <svg
        className="decision-analysis__sustainability-chart"
        viewBox="0 0 620 150"
        role="img"
        aria-labelledby={`${headingId}-chart-title ${headingId}-chart-description`}
      >
        <title id={`${headingId}-chart-title`}>Preparedness and realized shock absorption by day</title>
        <desc id={`${headingId}-chart-description`}>
          Thirty groups of three bars. Blue is preparedness material consumed, amber is
          preparedness crew utilized, and green is realized shock absorption. Each series is
          independently scaled; exact values follow in the table.
        </desc>
        <line x1="12" y1="126" x2="608" y2="126" />
        {points.map((point, index) => {
          const x = 18 + index * 19.5
          const materialHeight = 94 * point.preparednessMaterial / maxPreparednessMaterial
          const crewHeight = 94 * point.preparednessCrew / maxPreparednessCrew
          const absorptionHeight = 94 * point.absorbedServicePoints / maxAbsorption
          return (
            <g key={point.day}>
              <rect className="decision-analysis__prep-material-bar" x={x} y={126 - materialHeight} width="4" height={materialHeight} />
              <rect className="decision-analysis__prep-crew-bar" x={x + 5} y={126 - crewHeight} width="4" height={crewHeight} />
              <rect className="decision-analysis__absorb-bar" x={x + 10} y={126 - absorptionHeight} width="4" height={absorptionHeight} />
              {(point.day === 1 || point.day % 5 === 0) ? <text x={x + 6} y="142" textAnchor="middle">{point.day}</text> : null}
            </g>
          )
        })}
      </svg>
      <div className="decision-analysis__legend" aria-hidden="true">
        <span><i className="decision-analysis__prep-material-key" /> Prep material consumed</span>
        <span><i className="decision-analysis__prep-crew-key" /> Prep crew utilized</span>
        <span><i className="decision-analysis__absorb-key" /> Realized absorption</span>
      </div>
      <details>
        <summary>Inspect all 30 daily sustainability measurements</summary>
        <div className="decision-analysis__table-scroll" role="region" tabIndex={0} aria-label="Thirty-day sustainability evidence table">
          <table>
            <thead><tr><th scope="col">Day</th><th scope="col">Shock</th><th scope="col">Prep material consumed</th><th scope="col">Prep crew utilized</th><th scope="col">Service impact absorbed</th></tr></thead>
            <tbody>
              {points.map((point) => (
                <tr key={point.day}>
                  <th scope="row">{point.day}</th>
                  <td>{point.shockType === null ? 'Clear' : humanize(point.shockType)}</td>
                  <td>{units(point.preparednessMaterial)} material units</td>
                  <td>{units(point.preparednessCrew)} crew units</td>
                  <td>{(point.absorbedServicePoints * 100).toFixed(3)} service points</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <p className="decision-analysis__method-note">
        Preparedness investment uses the persisted material actually consumed and crew actually
        utilized, reported separately because they are different resources.{' '}
        Realized absorption is the persisted post-shock service level minus the same day’s
        unprotected service projection. It is measured from trajectory evidence, not estimated
        from a copied mitigation constant.
      </p>
    </section>
  )
}

function ExportPanel({ resultId }: { resultId: string }) {
  return (
    <section className="decision-analysis__exports" aria-labelledby="recovery-plan-export-heading">
      <header>
        <div>
          <p className="decision-analysis__eyebrow">Persisted evidence export</p>
          <h4 id="recovery-plan-export-heading">Download recovery plans</h4>
        </div>
        <Download size={20} aria-hidden="true" />
      </header>
      <div className="decision-analysis__export-groups">
        {(['candidate', 'baseline'] as const).map((planner) => (
          <div key={planner}>
            <span>{planner === 'candidate' ? 'PPO candidate' : 'Reactive heuristic'}</span>
            <a href={recoveryPlanUrl(resultId, planner, 'csv')} download>
              <Download size={14} aria-hidden="true" /> Download CSV
            </a>
            <a href={recoveryPlanUrl(resultId, planner, 'pdf')} download>
              <Download size={14} aria-hidden="true" /> Download PDF
            </a>
          </div>
        ))}
      </div>
      <p>CSV contains every day × service ledger row; PDF is the matching 30-day evidence brief.</p>
    </section>
  )
}

export function DecisionAnalysis({
  result,
  resultId,
  selectedDay,
  planner,
  onSelectedDayChange,
  onPlannerChange,
}: DecisionAnalysisProps) {
  const daySelectId = useId()
  const validDay = Number.isInteger(selectedDay) && selectedDay >= 1 && selectedDay <= 30
  const activeDay = validDay ? selectedDay : 1
  const identityMatches = resultId === result.result_id
  const selectedPlannerDay = result[planner].trajectory[activeDay - 1]
  const [explanations, setExplanations] = useState<ExplanationResponse | null>(null)
  const [explanationLoading, setExplanationLoading] = useState(false)
  const [explanationError, setExplanationError] = useState<string | null>(null)
  const [retryCount, setRetryCount] = useState(0)

  useEffect(() => {
    if (planner !== 'candidate' || !identityMatches) return undefined
    const controller = new AbortController()
    setExplanationLoading(true)
    setExplanationError(null)
    setExplanations(null)
    void fetchDecisionExplanations(result, controller.signal)
      .then((response) => {
        if (!controller.signal.aborted) setExplanations(response)
      })
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return
        const message = caught instanceof AnalysisApiError || caught instanceof Error
          ? caught.message
          : 'The explanation replay failed.'
        setExplanationError(message)
      })
      .finally(() => {
        if (!controller.signal.aborted) setExplanationLoading(false)
      })
    return () => controller.abort()
  }, [identityMatches, planner, result, resultId, retryCount])

  const dailyTrail = useMemo(() => result[planner].trajectory.map((day, index) => ({
    day,
    signal: planner === 'candidate'
      ? strongestChannel(explanations?.days[index])
      : heuristicSignal(day),
  })), [explanations, planner, result])

  if (!identityMatches || !validDay) {
    return (
      <section className="decision-analysis decision-analysis__contract-error" role="alert">
        <AlertTriangle size={20} aria-hidden="true" />
        <div>
          <h3>Decision analysis blocked</h3>
          <p>{!identityMatches
            ? 'The selected result identity does not match the comparison evidence.'
            : 'The selected day must be an integer from 1 to 30.'}</p>
        </div>
      </section>
    )
  }

  return (
    <section className="decision-analysis" aria-labelledby="decision-analysis-heading">
      <header className="decision-analysis__heading">
        <div>
          <p className="decision-analysis__eyebrow">Decision evidence / selected day {selectedDay}</p>
          <h3 id="decision-analysis-heading">Decision log & recovery analysis</h3>
          <span>Inspect what moved the policy, test one allocation, and export the persisted plan.</span>
        </div>
        <PlannerControl planner={planner} onPlannerChange={onPlannerChange} />
      </header>

      <section className="decision-analysis__selected-day" aria-labelledby="selected-decision-heading">
        <header>
          <div>
            <p className="decision-analysis__eyebrow">Day {selectedDay} / executed controls</p>
            <h4 id="selected-decision-heading">
              {selectedPlannerDay.shock.type === null ? 'Clear operations' : humanize(selectedPlannerDay.shock.type)}
            </h4>
          </div>
          <div className="decision-analysis__day-control">
            <b>Reward {selectedPlannerDay.reward.toFixed(4)}</b>
            <label htmlFor={daySelectId}>Selected decision day</label>
            <select
              id={daySelectId}
              value={selectedDay}
              onChange={(event) => onSelectedDayChange(Number(event.target.value))}
            >
              {result[planner].trajectory.map((day) => (
                <option key={day.day} value={day.day}>Day {day.day}</option>
              ))}
            </select>
          </div>
        </header>
        <div className="decision-analysis__execution-summary">
          <div><span>Material used</span><b>{units(selectedPlannerDay.material_used)}</b></div>
          <div><span>Crew used</span><b>{units(selectedPlannerDay.crew_used)}</b></div>
          <div><span>Stock released</span><b>{units(selectedPlannerDay.stock_release.reduce((sum, value) => sum + value, 0))}</b></div>
          <div><span>Preparedness</span><b>{units(selectedPlannerDay.preparedness_investment.reduce((sum, value) => sum + value, 0))}</b></div>
          <div><span>Resilience</span><b>{percent(selectedPlannerDay.resilience, 2)}</b></div>
        </div>
      </section>

      {planner === 'candidate' ? (
        <AttributionPanel
          selectedDay={selectedDay}
          explanations={explanations}
          loading={explanationLoading}
          error={explanationError}
          onRetry={() => setRetryCount((count) => count + 1)}
        />
      ) : <HeuristicEvidence day={selectedPlannerDay} />}

      <section className="decision-analysis__trail" aria-labelledby="decision-trail-heading">
        <header>
          <div>
            <p className="decision-analysis__eyebrow">Full horizon / persisted actions</p>
            <h4 id="decision-trail-heading">Thirty-day decision trail</h4>
          </div>
          <span>{planner === 'candidate' ? 'Strongest local signal per day' : 'Recorded rule evidence per day'}</span>
        </header>
        <div className="decision-analysis__table-scroll" role="region" tabIndex={0} aria-label="Thirty-day decision log">
          <table>
            <thead><tr><th scope="col">Day</th><th scope="col">Event</th><th scope="col">Decision evidence</th><th scope="col">Material</th><th scope="col">Crew</th><th scope="col">Preparedness</th><th scope="col">Reward</th></tr></thead>
            <tbody>
              {dailyTrail.map(({ day, signal }) => (
                <tr key={day.day} data-selected={day.day === selectedDay}>
                  <th scope="row">
                    <button
                      type="button"
                      className="decision-analysis__day-button"
                      aria-current={day.day === selectedDay ? 'true' : undefined}
                      aria-label={`Inspect decision evidence for day ${day.day}`}
                      onClick={() => onSelectedDayChange(day.day)}
                    >
                      {day.day}
                    </button>
                  </th>
                  <td>{day.shock.type === null ? 'Clear' : humanize(day.shock.type)}</td>
                  <td>{signal}</td>
                  <td>{units(day.material_used)}</td>
                  <td>{units(day.crew_used)}</td>
                  <td>{units(day.preparedness_investment.reduce((sum, value) => sum + value, 0))}</td>
                  <td>{day.reward.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="decision-analysis__lower-grid">
        <CounterfactualPanel result={result} selectedDay={selectedDay} />
        <div className="decision-analysis__side-stack">
          <SustainabilityPanel result={result} planner={planner} />
          <ExportPanel resultId={resultId} />
        </div>
      </div>
    </section>
  )
}

export default DecisionAnalysis
