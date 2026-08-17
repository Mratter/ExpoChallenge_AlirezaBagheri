import {
  useCallback,
  useEffect,
  lazy,
  useMemo,
  useRef,
  useState,
  Suspense,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import {
  Activity,
  AlertTriangle,
  ArchiveRestore,
  ArrowUpRight,
  Boxes,
  Building2,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Cpu,
  Database,
  PackageOpen,
  Play,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Users,
  XCircle,
} from 'lucide-react'
import { dayFromChartPointer } from './chartInteraction'
import {
  ComparisonError,
  fetchMetadata,
  listSimulations,
  loadSimulation,
  runComparison,
} from './api'
import { DecisionAnalysis } from './DecisionAnalysis'
import { toolboxPresets } from './generated/toolboxPresets'
import { LandingPage } from './LandingPage'
import { ppoName, reactiveHeuristicName, reactiveHeuristicShortName } from './plannerNames'
import { hashForRoute, routeFromHashValue, titleForRoute, type AppRoute } from './routes'
import { defaultScenario, defaultSeed, scenariosMatch } from './scenarios'
import { shockDisplayName } from './shockPresentation'
import {
  environmentContract,
  observationOrder,
  actionOrder,
  requestLimits,
  services,
  shockTypes,
  type CompareResponse,
  type Metadata,
  type OfficialOutcome,
  type SavedResultSummary,
  type Scenario,
  type Service,
  type Vector5,
} from './types'
import {
  actionEntriesForDay,
  observationEntriesForDay,
  outcomeReasonLabel,
  scenarioIssue,
  serviceLabel,
  tailStartDay,
  type PlannerKey,
} from './viewModel'

type ViewMode = 'trajectory' | 'audit' | 'dispatch' | 'decisions'
type LoadFailure = { code: string; message: string }
const CityGame = lazy(() => import('./game/CityGame').then((module) => ({ default: module.CityGame })))

const viewModes: ViewMode[] = ['trajectory', 'audit', 'dispatch', 'decisions']

/** Grouped so the picker reads as a difficulty ladder rather than a flat list. */
const presetGroups = [
  { outcome: 'both', heading: 'Both planners solve' },
  { outcome: 'ppo_only', heading: 'Only PPO solves' },
  { outcome: 'neither', heading: 'Neither planner solves' },
] as const

const serviceCodes: Record<Service, string> = {
  transport: 'TR',
  housing: 'HO',
  food: 'FD',
  healthcare: 'HC',
  public_services: 'PS',
}

function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`
}

function units(value: number): string {
  return value.toFixed(1)
}

function shortHash(value: string): string {
  return `${value.slice(0, 12)}…${value.slice(-6)}`
}

function updateVector(vector: Vector5, index: number, value: number): Vector5 {
  const next = [...vector] as Vector5
  next[index] = value
  return next
}

function plannerTitle(planner: PlannerKey): string {
  return planner === 'candidate' ? ppoName : reactiveHeuristicName
}

function RuntimeRail({ metadata, failure }: { metadata: Metadata | null; failure: LoadFailure | null }) {
  if (!metadata) {
    return (
      <div className="benchmark-rail benchmark-pending" role="status">
        <span><CircleDot size={14} />Runtime policy</span>
        <b>Waiting for configured model</b>
        <small>{failure?.code === 'DEPENDENCY_NOT_READY' ? 'Configure a valid ONNX policy to enable comparisons.' : 'Runtime metadata is unavailable.'}</small>
      </div>
    )
  }
  return (
    <div className="benchmark-rail benchmark-ready" aria-label="Configured runtime contract">
      <span><ShieldCheck size={14} />Runtime contract ready</span>
      <b>{metadata.model.id}</b>
      <i aria-hidden="true" />
      <b>{metadata.environment.observation_count} inputs · {metadata.environment.action_count} actions</b>
      <small>Explicit ONNX artifact · deterministic CPU inference</small>
    </div>
  )
}

function ScenarioEditor({
  draft,
  seed,
  busy,
  runReady,
  savedRuns,
  onDraft,
  onSeed,
  onRun,
  onReset,
  onRestore,
  onPreset,
  onRecheck,
}: {
  draft: Scenario
  seed: number
  busy: boolean
  runReady: boolean
  savedRuns: SavedResultSummary[]
  onDraft: (value: Scenario) => void
  onSeed: (value: number) => void
  onRun: () => void
  onReset: () => void
  onRestore: (id: string) => void
  onPreset: (id: string) => void
  onRecheck: () => void
}) {
  const issue = scenarioIssue(draft, seed)
  const tailStart = tailStartDay(draft)
  const forced = draft.forced_shock
  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (!busy && runReady && !issue) onRun()
  }

  return (
    <aside className="scenario-panel" aria-labelledby="scenario-heading">
      <form onSubmit={submit}>
        <div className="scenario-scroll">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Scenario control</p>
              <h2 id="scenario-heading">{requestLimits.horizonDays.constant}-day exercise</h2>
            </div>
            <button className="icon-button" type="button" onClick={onReset} disabled={busy} aria-label="Reset scenario">
              <RotateCcw size={16} />
            </button>
          </div>

          <label className="field full-field">
            <span>Exercise name</span>
            <input value={draft.name} minLength={1} maxLength={64} required onChange={(event) => onDraft({ ...draft, name: event.target.value })} />
          </label>

          <label className="field full-field preset-field">
            <span><Sparkles size={13} />Preset scenarios</span>
            <select value="" disabled={busy} onChange={(event) => event.target.value && onPreset(event.target.value)}>
              <option value="">Load a preset scenario…</option>
              {presetGroups.map(({ outcome, heading }) => (
                <optgroup key={outcome} label={heading}>
                  {toolboxPresets.presets.filter((preset) => preset.outcome === outcome).map((preset) => (
                    <option key={preset.id} value={preset.id}>{preset.label}</option>
                  ))}
                </optgroup>
              ))}
            </select>
            <small>
              {toolboxPresets.summary.ppoSolved} of {toolboxPresets.summary.total} solved by PPO,
              {' '}{toolboxPresets.summary.heuristicSolved} by the {reactiveHeuristicShortName.toLowerCase()},
              {' '}{toolboxPresets.summary.neitherSolved} by neither.
            </small>
          </label>

          <label className="field full-field saved-field">
            <span><ArchiveRestore size={13} />Saved runs</span>
            <select value="" disabled={busy || savedRuns.length === 0} onChange={(event) => event.target.value && onRestore(event.target.value)}>
              <option value="">{savedRuns.length ? 'Load a saved run…' : 'No saved runs yet'}</option>
              {savedRuns.map((saved) => (
                <option key={saved.result_id} value={saved.result_id}>
                  {saved.scenario_name} · PPO {saved.candidate_solved ? 'solved' : 'failed'} · {percent(saved.candidate_rauc)} resilience
                </option>
              ))}
            </select>
          </label>

          <div className="protocol-locks" aria-label="Scenario protocol">
            <div><span>Horizon</span><b>{requestLimits.horizonDays.constant} days</b></div>
            <div><span>Assessment</span><b>Days {tailStart}–{requestLimits.horizonDays.constant}</b></div>
            <div><span>Tail shocks</span><b>Blocked</b></div>
          </div>

          <div className="field-grid">
            <label className="field"><span>Seed</span><input type="number" min={requestLimits.seed.minimum} max={requestLimits.seed.maximum} step="1" value={seed} onChange={(event) => onSeed(Number(event.target.value))} /></label>
            <label className="field"><span>Material / day</span><input type="number" min={requestLimits.dailyBudget.minimum} max={requestLimits.dailyBudget.maximum} step="1" value={draft.daily_budget} onChange={(event) => onDraft({ ...draft, daily_budget: Number(event.target.value) })} /></label>
            <label className="field"><span>Crew / day</span><input type="number" min={requestLimits.dailyCrewPool.minimum} max={requestLimits.dailyCrewPool.maximum} step="1" value={draft.daily_crew_pool} onChange={(event) => onDraft({ ...draft, daily_crew_pool: Number(event.target.value) })} /></label>
            <label className="field"><span>Shock chance</span><div className="input-suffix"><input type="number" min={requestLimits.shockProbability.minimum * 100} max={requestLimits.shockProbability.maximum * 100} step="1" value={Math.round(draft.shock_probability * 100)} onChange={(event) => onDraft({ ...draft, shock_probability: Number(event.target.value) / 100 })} /><b>%</b></div></label>
            <label className="field"><span>Severity min</span><div className="input-suffix"><input type="number" min={requestLimits.severityMin.minimum * 100} max={requestLimits.severityMin.maximum * 100} value={Math.round(draft.severity_min * 100)} onChange={(event) => onDraft({ ...draft, severity_min: Number(event.target.value) / 100 })} /><b>%</b></div></label>
            <label className="field"><span>Severity max</span><div className="input-suffix"><input type="number" min={requestLimits.severityMax.minimum * 100} max={requestLimits.severityMax.maximum * 100} value={Math.round(draft.severity_max * 100)} onChange={(event) => onDraft({ ...draft, severity_max: Number(event.target.value) / 100 })} /><b>%</b></div></label>
          </div>

          <fieldset className="service-editor">
            <legend>Starting state · priority · solved target</legend>
            <div className="service-editor-head"><span>Service</span><span>Start</span><span>Weight</span><span>Target</span></div>
            {services.map((service, index) => (
              <div className="service-input-row" key={service}>
                <span><b>{serviceCodes[service]}</b>{serviceLabel(service)}</span>
                <input aria-label={`${serviceLabel(service)} starting state`} type="number" min={requestLimits.initialServices.minimum * 100} max={requestLimits.initialServices.maximum * 100} value={Math.round(draft.initial_services[index] * 100)} onChange={(event) => onDraft({ ...draft, initial_services: updateVector(draft.initial_services, index, Number(event.target.value) / 100) })} />
                <input aria-label={`${serviceLabel(service)} priority`} type="number" min={requestLimits.priorities.minimum} max={requestLimits.priorities.maximum} step="0.1" value={draft.priorities[index]} onChange={(event) => onDraft({ ...draft, priorities: updateVector(draft.priorities, index, Number(event.target.value)) })} />
                <input aria-label={`${serviceLabel(service)} solved target`} type="number" min={requestLimits.recoveryTargets.minimum * 100} max={requestLimits.recoveryTargets.maximum * 100} value={Math.round(draft.recovery_targets[index] * 100)} onChange={(event) => onDraft({ ...draft, recovery_targets: updateVector(draft.recovery_targets, index, Number(event.target.value) / 100) })} />
              </div>
            ))}
          </fieldset>

          <fieldset className="forced-editor">
            <legend>Controlled incident</legend>
            <label className="toggle-row">
              <input type="checkbox" checked={forced !== null} onChange={(event) => onDraft({ ...draft, forced_shock: event.target.checked ? { day: 5, type: 'utility', severity: 0.26 } : null })} />
              <span><b>Schedule one forced shock</b><small>Always rejected on assessment days {tailStart}–{requestLimits.horizonDays.constant}.</small></span>
            </label>
            {forced ? (
              <div className="forced-fields">
                <label className="field"><span>Day</span><input type="number" min={requestLimits.forcedShockDay.minimum} max={tailStart - 1} value={forced.day} onChange={(event) => onDraft({ ...draft, forced_shock: { ...forced, day: Number(event.target.value) } })} /></label>
                <label className="field"><span>Type</span><select value={forced.type} onChange={(event) => onDraft({ ...draft, forced_shock: { ...forced, type: event.target.value as typeof forced.type } })}>{shockTypes.map((type) => <option value={type} key={type}>{shockDisplayName(type)}</option>)}</select></label>
                <label className="field"><span>Severity</span><div className="input-suffix"><input type="number" min={requestLimits.forcedShockSeverity.minimum * 100} max={requestLimits.forcedShockSeverity.maximum * 100} value={Math.round(forced.severity * 100)} onChange={(event) => onDraft({ ...draft, forced_shock: { ...forced, severity: Number(event.target.value) / 100 } })} /><b>%</b></div></label>
              </div>
            ) : null}
            {draft.forced_shocks.length ? (
              <div className="injected-shocks"><b>{draft.forced_shocks.length} city-view incident{draft.forced_shocks.length === 1 ? '' : 's'} attached</b><button type="button" onClick={() => onDraft({ ...draft, forced_shocks: [] })}>Clear</button></div>
            ) : null}
          </fieldset>
        </div>

        <div className="scenario-dock">
          {issue ? <p className="scenario-issue"><AlertTriangle size={13} />{issue}</p> : null}
          {!runReady ? <div className="release-recheck"><p className="scenario-issue release-note"><CircleDot size={13} />Run disabled until a valid policy is configured.</p><button type="button" onClick={onRecheck} disabled={busy}>Recheck runtime</button></div> : null}
          <button className="run-button" type="submit" disabled={busy || !runReady || Boolean(issue)}>
            {busy ? <Activity className="spin" size={17} /> : <Play size={17} fill="currentColor" />}
            {busy ? 'Resolving paired trace' : `Run paired ${requestLimits.horizonDays.constant}-day trace`}
          </button>
        </div>
      </form>
    </aside>
  )
}

function OutcomeChecks({ outcome }: { outcome: OfficialOutcome }) {
  return (
    <ul className="outcome-checks">
      {Object.entries(outcome.checks).map(([name, passed]) => (
        <li key={name} data-pass={passed}>
          {passed ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
          <span>{outcomeReasonLabel(name)}</span>
        </li>
      ))}
    </ul>
  )
}

function VerdictPair({ result }: { result: CompareResponse }) {
  const candidate = result.candidate.absolute_outcome
  const baseline = result.baseline.absolute_outcome
  return (
    <section className="verdict-pair" aria-label="Independent official disaster outcomes">
      <article className="verdict-card candidate-verdict" data-solved={candidate.solved}>
        <header><span>PPO / ONNX</span><b>{candidate.solved ? 'SOLVED' : 'FAILED'}</b></header>
        <p>Independent absolute outcome</p>
        <strong>{percent(candidate.resilience_auc)}</strong><small>resilience AUC · floor {percent(candidate.resilience_auc_floor)}</small>
        <OutcomeChecks outcome={candidate} />
      </article>
      <div className="verdict-divider"><span>same tape</span><i /><b>{result.comparison.absolute_outcome_pair.replaceAll('_', ' ')}</b></div>
      <article className="verdict-card baseline-verdict" data-solved={baseline.solved}>
        <header><span>{reactiveHeuristicName}</span><b>{baseline.solved ? 'SOLVED' : 'FAILED'}</b></header>
        <p>Independent absolute outcome</p>
        <strong>{percent(baseline.resilience_auc)}</strong><small>resilience AUC · floor {percent(baseline.resilience_auc_floor)}</small>
        <OutcomeChecks outcome={baseline} />
      </article>
    </section>
  )
}

function linePath(values: number[]): string {
  const width = 736
  const height = 176
  return values.map((value, index) => {
    const x = (index / Math.max(values.length - 1, 1)) * width
    const y = height - value * height
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
  }).join(' ')
}

function PairedTrace({ result, selectedDay, onDay }: { result: CompareResponse; selectedDay: number; onDay: (day: number) => void }) {
  const candidate = result.candidate.trajectory.map((day) => day.resilience)
  const baseline = result.baseline.trajectory.map((day) => day.resilience)
  const finalDayIndex = result.scenario.horizon_days - 1
  const selectedX = ((selectedDay - 1) / finalDayIndex) * 736
  const tailX = ((tailStartDay(result.scenario) - 1) / finalDayIndex) * 736
  const selectedShock = result.shock_schedule[selectedDay - 1]

  function inspectAt(clientX: number, svg: SVGSVGElement) {
    const bounds = svg.getBoundingClientRect()
    onDay(dayFromChartPointer({
      boundsLeft: bounds.left,
      boundsWidth: bounds.width,
      clientX,
      dayEnd: result.scenario.horizon_days,
      dayStart: 1,
      plotLeft: 52,
      plotWidth: 736,
      viewWidth: 824,
    }))
  }

  return (
    <figure className="paired-trace" aria-labelledby="trace-title">
      <figcaption>
        <div><p className="eyebrow">Signature evidence</p><h3 id="trace-title">Paired {result.scenario.horizon_days}-day recovery trace</h3></div>
        <div className="trace-legend"><span className="legend-ppo">PPO policy</span><span className="legend-heuristic">{reactiveHeuristicShortName}</span><span className="legend-tail">Assessment tail</span></div>
      </figcaption>
      <svg
        viewBox="0 0 824 246"
        role="img"
        aria-label={`Both planners on the same ${result.scenario.horizon_days}-day shock tape; day ${selectedDay} selected`}
        onPointerDown={(event) => inspectAt(event.clientX, event.currentTarget)}
        onPointerMove={(event) => {
          if (event.pointerType === 'mouse') inspectAt(event.clientX, event.currentTarget)
        }}
      >
        <g transform="translate(52 24)">
          <rect x={tailX} y="0" width={736 - tailX} height="176" className="tail-zone" />
          {[0.25, 0.5, 0.75, 1].map((tick) => (
            <g key={tick}><line x1="0" x2="736" y1={176 - tick * 176} y2={176 - tick * 176} className="trace-grid" /><text x="-10" y={180 - tick * 176} textAnchor="end">{Math.round(tick * 100)}</text></g>
          ))}
          {result.shock_schedule.map((shock, index) => shock.type ? (
            <g key={`${shock.day}-${shock.type}`}>
              <line x1={(index / finalDayIndex) * 736} x2={(index / finalDayIndex) * 736} y1="0" y2="176" className={shock.forced ? 'shock-guide forced' : 'shock-guide'} />
              <circle cx={(index / finalDayIndex) * 736} cy="5" r={shock.forced ? 4 : 3} className={shock.forced ? 'shock-dot forced' : 'shock-dot'}><title>Day {shock.day}: {shockDisplayName(shock.type)} · {percent(shock.severity)}</title></circle>
            </g>
          ) : null)}
          <line x1={selectedX} x2={selectedX} y1="0" y2="176" className="selected-guide" />
          <g className="trace-lines">
            <path d={linePath(baseline)} className="trace-line trace-baseline" />
            <path d={linePath(candidate)} className="trace-line trace-candidate" />
          </g>
          <circle cx={selectedX} cy={176 - candidate[selectedDay - 1] * 176} r="4.5" className="selected-ppo" />
          <circle cx={selectedX} cy={176 - baseline[selectedDay - 1] * 176} r="4" className="selected-heuristic" />
          <text x="0" y="203">Day 1</text><text x={tailX + 6} y="17" className="tail-label">NO INJECTED SHOCKS</text><text x="736" y="203" textAnchor="end">Day {result.scenario.horizon_days}</text>
        </g>
      </svg>
      <div className="trace-readout">
        <span>Day {selectedDay} · {selectedShock?.type ? shockDisplayName(selectedShock.type) : 'Clear operations'}</span>
        <b className="trace-readout-ppo">PPO {percent(candidate[selectedDay - 1], 2)}</b>
        <b className="trace-readout-heuristic">{reactiveHeuristicShortName} {percent(baseline[selectedDay - 1], 2)}</b>
      </div>
      <label className="trace-scrubber"><span>Inspect day {selectedDay}</span><input aria-valuetext={`Day ${selectedDay}; ${selectedShock?.type ? shockDisplayName(selectedShock.type) : 'clear operations'}; PPO ${percent(candidate[selectedDay - 1], 2)}; heuristic ${percent(baseline[selectedDay - 1], 2)}`} type="range" min="1" max={result.scenario.horizon_days} value={selectedDay} onChange={(event) => onDay(Number(event.target.value))} /></label>
    </figure>
  )
}

function PlannerToggle({ planner, onPlanner }: { planner: PlannerKey; onPlanner: (value: PlannerKey) => void }) {
  return (
    <div className="planner-toggle" role="group" aria-label="Planner to inspect">
      <button type="button" className={planner === 'candidate' ? 'active' : ''} aria-pressed={planner === 'candidate'} onClick={() => onPlanner('candidate')}>PPO policy</button>
      <button type="button" className={planner === 'baseline' ? 'active' : ''} aria-pressed={planner === 'baseline'} onClick={() => onPlanner('baseline')}>Heuristic</button>
    </div>
  )
}

function DayInspector({ result, selectedDay, planner, onPlanner }: { result: CompareResponse; selectedDay: number; planner: PlannerKey; onPlanner: (value: PlannerKey) => void }) {
  const dayIndex = selectedDay - 1
  const day = result[planner].trajectory[dayIndex]
  const observations = useMemo(() => observationEntriesForDay(result, planner, dayIndex), [dayIndex, planner, result])
  const actions = useMemo(() => actionEntriesForDay(result, planner, dayIndex), [dayIndex, planner, result])
  return (
    <section className="day-inspector" aria-labelledby="day-inspector-title">
      <header className="inspector-heading">
        <div><p className="eyebrow">Day {selectedDay} command receipt</p><h3 id="day-inspector-title">{plannerTitle(planner)}</h3></div>
        <PlannerToggle planner={planner} onPlanner={onPlanner} />
      </header>
      <div className="day-facts">
        <div><PackageOpen size={15} /><span>Material</span><b>{units(day.material_used)} / {units(day.available_budget)}</b><small>{units(day.material_unspent)} unspent</small></div>
        <div><Users size={15} /><span>Crew</span><b>{units(day.crew_used)} / {units(day.available_crew)}</b><small>{units(day.crew_idle)} idle</small></div>
        <div><Boxes size={15} /><span>Stock released</span><b>{units(day.stock_release.reduce((sum, value) => sum + value, 0))}</b><small>five depots</small></div>
        <div><ShieldCheck size={15} /><span>Preparedness</span><b>{units(day.preparedness_investment.reduce((sum, value) => sum + value, 0))}</b><small>effective investment</small></div>
      </div>
      <div className="service-ledger" role="table" aria-label={`Day ${selectedDay} intervention ledger`}>
        <div className="service-ledger-head" role="row"><span>Service</span><span>End / target</span><span>Material</span><span>Crew</span><span>Release</span><span>Preparedness</span></div>
        {services.map((service, index) => (
          <div className="service-ledger-row" data-service={service} role="row" key={service}>
            <span className="service-key"><b>{serviceCodes[service]}</b>{serviceLabel(service)}</span>
            <span className="state-cell"><b>{percent(day.services_end[index])}</b><small>target {percent(result.scenario.recovery_targets[index])}</small><i><em style={{ width: `${day.services_end[index] * 100}%` }} /></i></span>
            <strong>{units(day.material_allocation[index])}</strong>
            <strong>{units(day.crew_allocation[index])}</strong>
            <strong>{units(day.stock_release[index])}</strong>
            <strong>{units(day.preparedness_investment[index])}</strong>
          </div>
        ))}
      </div>
      <div className="projection-receipt">
        <span>Material projection <b>{day.projection.distance.toFixed(3)}</b></span>
        <span>Crew projection <b>{day.crew_projection.distance.toFixed(3)}</b></span>
        <span>Hard violations <b>{day.hard_violation_count}</b></span>
        <span>Conservation residual <b>{Math.max(...day.logistics.conservation_residual.map(Math.abs)).toExponential(2)}</b></span>
      </div>
      <div className="vector-pair">
        <details>
          <summary><span><Database size={15} />Public observation</span><b>{observations.length} inputs</b><ChevronRight size={15} /></summary>
          <div className="vector-grid">{observations.map((entry, index) => <div key={entry.name}><span>{String(index + 1).padStart(2, '0')}</span><code>{entry.name}</code><b>{entry.value.toFixed(5)}</b></div>)}</div>
        </details>
        <details>
          <summary><span><SlidersHorizontal size={15} />Policy output</span><b>{actions.length} actions</b><ChevronRight size={15} /></summary>
          <div className="vector-grid action-vector">{actions.map((entry, index) => <div key={entry.name}><span>{String(index + 1).padStart(2, '0')}</span><code>{entry.name}</code><b>{entry.value.toFixed(5)}</b></div>)}</div>
        </details>
      </div>
    </section>
  )
}

function AuditView({ result }: { result: CompareResponse }) {
  return (
    <section className="table-view" aria-labelledby="audit-heading">
      <header className="table-heading"><div><p className="eyebrow">All returned days</p><h3 id="audit-heading">Daily paired audit</h3></div><p>Both planners consume the same recorded shock schedule. Values are engine receipts.</p></header>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Daily paired audit table">
        <table><thead><tr><th>Day</th><th>Event</th><th>PPO resilience</th><th>Heuristic resilience</th><th>Δ RAUC signal</th><th>PPO material</th><th>PPO crew</th><th>PPO prep</th><th>Tail</th></tr></thead>
          <tbody>{result.candidate.trajectory.map((day, index) => {
            const baseline = result.baseline.trajectory[index]
            return <tr key={day.day}><td>{day.day}</td><td>{day.shock.type ? shockDisplayName(day.shock.type) : 'Clear'}{day.shock.forced ? <small>forced</small> : null}</td><td>{percent(day.resilience)}</td><td>{percent(baseline.resilience)}</td><td className={day.resilience >= baseline.resilience ? 'positive' : 'negative'}>{((day.resilience - baseline.resilience) * 100).toFixed(2)} pp</td><td>{units(day.material_used)}</td><td>{units(day.crew_used)}</td><td>{units(day.preparedness_investment.reduce((sum, value) => sum + value, 0))}</td><td>{day.shock.assessment_tail ? 'Assessment' : 'Operations'}</td></tr>
          })}</tbody>
        </table>
      </div>
    </section>
  )
}

function DispatchView({ result, selectedDay, planner, onPlanner }: { result: CompareResponse; selectedDay: number; planner: PlannerKey; onPlanner: (value: PlannerKey) => void }) {
  const day = result[planner].trajectory[selectedDay - 1]
  const residual = Math.max(...day.logistics.conservation_residual.map(Math.abs))
  return (
    <section className="table-view dispatch-view" aria-labelledby="dispatch-heading">
      <header className="table-heading"><div><p className="eyebrow">Day {selectedDay} / physical ledger</p><h3 id="dispatch-heading">Dispatch manifest</h3></div><PlannerToggle planner={planner} onPlanner={onPlanner} /></header>
      <div className="manifest-summary"><div><span>Road capacity</span><b>{percent(day.logistics.road_capacity)}</b></div><div><span>Maximum residual</span><b>{residual.toExponential(3)}</b></div><div><span>Pending next day</span><b>{units(day.logistics.pending_next_day.reduce((sum, value) => sum + value, 0))}</b></div><div><span>Repair supply</span><b>{units(day.logistics.repair_supply.reduce((sum, value) => sum + value, 0))}</b></div></div>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Depot dispatch manifest">
        <table><thead><tr><th>Depot</th><th>Stock before</th><th>Arrivals landed</th><th>Preparedness consumed</th><th>Repair committed</th><th>Crew assigned</th><th>Effective repair</th><th>Stock end</th><th>Pending</th></tr></thead>
          <tbody>{services.map((service, index) => <tr key={service}><td><b>{serviceCodes[service]}</b><small>{serviceLabel(service)}</small></td><td>{units(day.logistics.depot_stock_before[index])}</td><td>{units(day.logistics.pending_arrivals_landed[index] + day.logistics.same_day_delivery_landed[index])}</td><td>{units(day.logistics.preparedness_material_consumed[index])}</td><td>{units(day.logistics.repair_material_committed[index])}</td><td>{units(day.logistics.preparedness_crew_assigned[index] + day.logistics.repair_crew_assigned[index])}</td><td>{units(day.logistics.repair_supply[index])}</td><td>{units(day.logistics.depot_stock_end[index])}</td><td>{units(day.logistics.pending_next_day[index])}</td></tr>)}</tbody>
        </table>
      </div>
      {day.logistics.mutual_aid_transfers.length ? <div className="transfer-list"><h4>Recorded mutual aid</h4>{day.logistics.mutual_aid_transfers.map((transfer, index) => <span key={`${transfer.from_service}-${transfer.to_service}-${index}`}>{serviceCodes[transfer.from_service]} → {serviceCodes[transfer.to_service]} <b>{units(transfer.units)}</b></span>)}</div> : <p className="quiet-receipt">No inter-depot mutual-aid transfer was recorded on this day.</p>}
    </section>
  )
}

function DecisionView({
  result,
  selectedDay,
  planner,
  onPlanner,
  onDay,
}: {
  result: CompareResponse
  selectedDay: number
  planner: PlannerKey
  onPlanner: (value: PlannerKey) => void
  onDay: (day: number) => void
}) {
  return (
    <section className="decision-view" aria-labelledby="decision-heading">
      <h3 id="decision-heading" className="sr-only">Decision log</h3>
      <DecisionAnalysis
        result={result}
        resultId={result.result_id}
        selectedDay={selectedDay}
        planner={planner}
        onPlannerChange={onPlanner}
        onSelectedDayChange={onDay}
      />
    </section>
  )
}

function ArchitecturePanel({ metadata, result, failure }: { metadata: Metadata | null; result: CompareResponse | null; failure: LoadFailure | null }) {
  const observations = metadata?.model.observation_order ?? result?.observation_order ?? []
  const actions = metadata?.model.action_order ?? result?.action_order ?? []
  return (
    <section className="architecture-panel" aria-labelledby="architecture-heading">
      <header><div><p className="eyebrow">Current runtime / loaded contract</p><h2 id="architecture-heading">Policy architecture & interface</h2></div><span className={metadata ? 'seal-ready' : 'seal-pending'}>{metadata ? <ShieldCheck size={15} /> : <CircleDot size={15} />}{metadata ? 'Runtime contract ready' : 'Runtime contract unavailable'}</span></header>
      <div className="architecture-flow">
        <article><Database size={18} /><span>Public observation</span><strong>{observations.length || '—'}</strong><small>causal channels</small></article><ChevronRight size={18} />
        <article><Cpu size={18} /><span>ONNX policy</span><strong>{metadata ? 'CPU' : '—'}</strong><small>{metadata ? 'deterministic inference' : 'runtime unavailable'}</small></article><ChevronRight size={18} />
        <article><SlidersHorizontal size={18} /><span>Continuous control</span><strong>{actions.length || '—'}</strong><small>bounded actions</small></article><ChevronRight size={18} />
        <article><Building2 size={18} /><span>Exact solver</span><strong>2×</strong><small>material + crew projections</small></article>
      </div>
      {metadata ? (
        <div className="architecture-facts">
          <div><span>Model</span><b>{metadata.model.id}</b><small>{metadata.model.artifact_type}</small></div>
          <div><span>Runtime</span><b>{metadata.model.runtime}</b><small>{metadata.model.observation_contract.normalization.replaceAll('_', ' ')}</small></div>
          <div><span>Runtime artifact</span><code>{shortHash(metadata.model.sha256)}</code><small>ONNX SHA-256</small></div>
          <div><span>Environment</span><code>{shortHash(metadata.environment.spec_sha256)}</code><small>{metadata.environment.id} · {metadata.environment.version}</small></div>
        </div>
      ) : <p className="architecture-withheld"><AlertTriangle size={15} />{failure?.message ?? 'Configure a compatible ONNX policy to expose the runtime contract.'}</p>}
      {observations.length && actions.length ? (
        <div className="contract-columns">
          <details><summary><span>Observation contract</span><b>{observations.length} ordered channels</b><ChevronRight size={15} /></summary><ol>{observations.map((name) => <li key={name}><code>{name}</code></li>)}</ol></details>
          <details><summary><span>Action contract</span><b>{actions.length} ordered controls</b><ChevronRight size={15} /></summary><ol>{actions.map((name) => <li key={name}><code>{name}</code></li>)}</ol></details>
        </div>
      ) : null}
    </section>
  )
}

function AnalystToolbox({ initialResult, onOpenGame, onOpenHome }: { initialResult?: CompareResponse | null; onOpenGame: (result: CompareResponse | null) => void; onOpenHome: () => void }) {
  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [metadataFailure, setMetadataFailure] = useState<LoadFailure | null>(null)
  const [draft, setDraft] = useState<Scenario>(initialResult?.scenario ?? defaultScenario)
  const [seed, setSeed] = useState(initialResult?.seed ?? defaultSeed)
  const [result, setResult] = useState<CompareResponse | null>(initialResult ?? null)
  const [savedRuns, setSavedRuns] = useState<SavedResultSummary[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<LoadFailure | null>(null)
  const [selectedDay, setSelectedDay] = useState(initialResult ? 5 : 5)
  const [planner, setPlanner] = useState<PlannerKey>('candidate')
  const [view, setView] = useState<ViewMode>('trajectory')
  const errorRef = useRef<HTMLDivElement>(null)

  const refreshMetadata = useCallback(async (signal?: AbortSignal) => {
    try {
      const payload = await fetchMetadata(signal)
      setMetadata(payload)
      setMetadataFailure(null)
    } catch (caught) {
      if (signal?.aborted) return
      setMetadata(null)
      setMetadataFailure({
        code: caught instanceof ComparisonError ? caught.code : 'METADATA_UNAVAILABLE',
        message: caught instanceof Error ? caught.message : 'Runtime metadata is unavailable.',
      })
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void refreshMetadata(controller.signal)
    void listSimulations(controller.signal).then(setSavedRuns).catch(() => undefined)
    return () => controller.abort()
  }, [refreshMetadata])

  useEffect(() => {
    if (!error) return
    errorRef.current?.focus()
  }, [error])

  const execute = useCallback(async () => {
    if (!metadata || scenarioIssue(draft, seed)) return
    setBusy(true)
    setError(null)
    try {
      const response = await runComparison(seed, draft)
      setResult(response)
      setSelectedDay(5)
      setPlanner('candidate')
      setView('trajectory')
      setSavedRuns(await listSimulations())
    } catch (caught) {
      setError({
        code: caught instanceof ComparisonError ? caught.code : 'RUNTIME_UNREACHABLE',
        message: caught instanceof Error ? caught.message : 'The paired comparison could not be completed.',
      })
    } finally {
      setBusy(false)
    }
  }, [draft, metadata, seed])

  const restore = useCallback(async (resultId: string) => {
    setBusy(true)
    setError(null)
    try {
      const response = await loadSimulation(resultId)
      setResult(response)
      setDraft(response.scenario)
      setSeed(response.seed)
      setSelectedDay(5)
      setPlanner('candidate')
      setView('trajectory')
    } catch (caught) {
      setError({
        code: caught instanceof ComparisonError ? caught.code : 'PERSISTENCE_FAILED',
        message: caught instanceof Error ? caught.message : 'The saved result could not be restored.',
      })
    } finally {
      setBusy(false)
    }
  }, [])

  /** Load a preset's frozen scenario and run it, so the stated outcome is shown, not asserted. */
  const applyPreset = useCallback(async (presetId: string) => {
    const preset = toolboxPresets.presets.find((candidate) => candidate.id === presetId)
    if (!preset) return
    const scenario = preset.scenario as unknown as Scenario
    setDraft(scenario)
    setSeed(preset.seed)
    setError(null)
    if (!metadata) return
    setBusy(true)
    try {
      const response = await runComparison(preset.seed, scenario)
      setResult(response)
      setSelectedDay(5)
      setPlanner('candidate')
      setView('trajectory')
      setSavedRuns(await listSimulations())
    } catch (caught) {
      setError({
        code: caught instanceof ComparisonError ? caught.code : 'RUNTIME_UNREACHABLE',
        message: caught instanceof Error ? caught.message : 'The preset scenario could not be run.',
      })
    } finally {
      setBusy(false)
    }
  }, [metadata])

  const reset = () => {
    setDraft(defaultScenario)
    setSeed(defaultSeed)
    setError(null)
  }

  const draftChanged = result ? seed !== result.seed || !scenariosMatch(draft, result.scenario) : false
  const openCity = async () => {
    if (busy) return
    if (result && !draftChanged) {
      onOpenGame(result)
      return
    }
    if (!metadata) {
      onOpenGame(null)
      return
    }
    const issue = scenarioIssue(draft, seed)
    if (issue) {
      setError({ code: 'INVALID_SCENARIO', message: issue })
      return
    }
    setBusy(true)
    setError(null)
    try {
      const response = await runComparison(seed, draft)
      setResult(response)
      onOpenGame(response)
    } catch (caught) {
      setError({
        code: caught instanceof ComparisonError ? caught.code : 'RUNTIME_UNREACHABLE',
        message: caught instanceof Error ? caught.message : 'The 3D recovery run could not be prepared.',
      })
    } finally {
      setBusy(false)
    }
  }
  const handleTabKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    const index = viewModes.indexOf(view)
    let next: ViewMode | null = null
    if (event.key === 'ArrowRight') next = viewModes[(index + 1) % viewModes.length]
    if (event.key === 'ArrowLeft') next = viewModes[(index - 1 + viewModes.length) % viewModes.length]
    if (event.key === 'Home') next = viewModes[0]
    if (event.key === 'End') next = viewModes.at(-1) ?? null
    if (!next) return
    event.preventDefault()
    setView(next)
    document.getElementById(`tab-${next}`)?.focus()
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand-block brand-home" type="button" onClick={onOpenHome} aria-label="RELAY municipal recovery lab / Analyst Toolbox — return to evidence home"><span className="brand-seal" aria-hidden="true"><i /><i /><i /></span><span><span className="brand-eyebrow">Municipal recovery lab</span><strong>RELAY / Analyst Toolbox</strong></span></button>
        <RuntimeRail metadata={metadata} failure={metadataFailure} />
        <button className="city-switch" type="button" disabled={busy} onClick={() => void openCity()}><Building2 size={15} />{result && !draftChanged ? 'Open current run in 3D' : metadata ? 'Run & open 3D city' : 'Open 3D city setup'}<ArrowUpRight size={14} /></button>
      </header>

      <main className="workspace">
        <ScenarioEditor draft={draft} seed={seed} busy={busy} runReady={metadata !== null} savedRuns={savedRuns} onDraft={setDraft} onSeed={setSeed} onRun={() => void execute()} onReset={reset} onRestore={(id) => void restore(id)} onPreset={(id) => void applyPreset(id)} onRecheck={() => void refreshMetadata()} />
        <section className="results-panel" aria-live="polite" aria-busy={busy}>
          {error ? <div className="blocking-error" ref={errorRef} tabIndex={-1} role="alert"><AlertTriangle size={20} /><div><p>{error.code}</p><h2>Comparison blocked</h2><span>{error.message}</span></div><button type="button" onClick={() => setError(null)}>Dismiss</button></div> : null}
          {busy ? <div className="recompute-bar" role="status"><Activity className="spin" size={16} />Resolving two planners against one shared 30-day shock tape…</div> : null}

          {!result ? (
            <section className="empty-state">
              <div className="empty-grid" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /><i /></div>
              <p className="eyebrow">Technical judge workbench</p>
              <h2>One scenario. Two planners.<br />Two independent verdicts.</h2>
              <p>Configure public starting conditions, material, crew, targets, and hazards. The backend runs the PPO policy and fine tuned reactive heuristic on the same tape, then returns the official 30-day solved checks.</p>
              <div className="empty-protocol"><span><b>{requestLimits.horizonDays.constant}</b> days</span><span><b>{observationOrder.length}</b> public inputs</span><span><b>{actionOrder.length}</b> actions</span><span><b>{environmentContract.assessmentTailDays}</b> assessment days</span></div>
              <small>{metadata ? 'Runtime ready. Run the paired trace from the scenario panel.' : 'Run remains disabled until a compatible policy is configured.'}</small>
            </section>
          ) : (
            <>
              <header className="results-heading">
                <div><p className="eyebrow">Result {result.result_id.slice(0, 10)} / seed {result.seed}</p><h2>{result.scenario.name}</h2><span>Official outcome definition <code>{shortHash(result.outcome_definition_sha256)}</code></span></div>
                <div className="result-receipts"><span data-ok={result.candidate.hard_violation_count === 0 && result.baseline.hard_violation_count === 0}><ShieldCheck size={15} />Hard violations {result.candidate.hard_violation_count + result.baseline.hard_violation_count}</span><span><Database size={15} />Tape {shortHash(result.shock_schedule_sha256)}</span>{draftChanged ? <b>Draft changed · run to refresh</b> : null}</div>
              </header>

              <VerdictPair result={result} />

              <nav className="view-tabs" role="tablist" aria-label="Result evidence views">
                {viewModes.map((mode) => <button key={mode} id={`tab-${mode}`} type="button" role="tab" tabIndex={view === mode ? 0 : -1} aria-selected={view === mode} className={view === mode ? 'active' : ''} onClick={() => setView(mode)} onKeyDown={handleTabKey}>{mode === 'audit' ? 'Daily audit' : mode === 'dispatch' ? 'Dispatch manifest' : mode === 'decisions' ? 'Decision log' : 'Trajectory'}</button>)}
              </nav>

              <div role="tabpanel" aria-labelledby="tab-trajectory" hidden={view !== 'trajectory'}>
                <PairedTrace result={result} selectedDay={selectedDay} onDay={setSelectedDay} />
                <DayInspector result={result} selectedDay={selectedDay} planner={planner} onPlanner={setPlanner} />
              </div>
              <div role="tabpanel" aria-labelledby="tab-audit" hidden={view !== 'audit'}><AuditView result={result} /></div>
              <div role="tabpanel" aria-labelledby="tab-dispatch" hidden={view !== 'dispatch'}><DispatchView result={result} selectedDay={selectedDay} planner={planner} onPlanner={setPlanner} /></div>
              <div role="tabpanel" aria-labelledby="tab-decisions" hidden={view !== 'decisions'}><DecisionView result={result} selectedDay={selectedDay} planner={planner} onPlanner={setPlanner} onDay={setSelectedDay} /></div>

              <footer className="evidence-footer"><div><span>Policy</span><code>{shortHash(result.policy.sha256)}</code></div><div><span>Candidate trajectory</span><code>{shortHash(result.candidate.trajectory_sha256)}</code></div><div><span>Heuristic trajectory</span><code>{shortHash(result.baseline.trajectory_sha256)}</code></div></footer>
            </>
          )}

          <ArchitecturePanel metadata={metadata} result={result} failure={metadataFailure} />
        </section>
      </main>
    </div>
  )
}

function routeFromHash(): AppRoute {
  return routeFromHashValue(window.location.hash)
}

function App() {
  const [route, setRoute] = useState<AppRoute>(routeFromHash)
  const [latestResult, setLatestResult] = useState<CompareResponse | null>(null)
  const [gameLaunchResult, setGameLaunchResult] = useState<CompareResponse | null>(null)
  const [toolboxLaunchResult, setToolboxLaunchResult] = useState<CompareResponse | null>(null)

  useEffect(() => {
    const sync = () => {
      const next = routeFromHash()
      if (next === 'landing' && window.location.hash !== '#/') window.history.replaceState(null, '', '#/')
      setRoute(next)
    }
    sync()
    window.addEventListener('hashchange', sync)
    return () => window.removeEventListener('hashchange', sync)
  }, [])

  useEffect(() => {
    document.title = titleForRoute(route)
  }, [route])

  const navigate = (next: AppRoute) => {
    window.location.hash = hashForRoute(next)
    setRoute(next)
  }

  if (route === 'landing') return <LandingPage />

  return route === 'toolbox' ? (
    <AnalystToolbox initialResult={toolboxLaunchResult} onOpenHome={() => navigate('landing')} onOpenGame={(result) => { if (result) setLatestResult(result); setGameLaunchResult(result); setToolboxLaunchResult(null); navigate('game') }} />
  ) : (
    <Suspense fallback={<div className="route-loading" role="status"><Activity className="spin" size={22} /><b>Loading the 3D city renderer</b></div>}>
      <CityGame initialResult={gameLaunchResult} onResult={(result) => { setLatestResult(result); setGameLaunchResult(null) }} onOpenHome={() => navigate('landing')} onOpenToolbox={() => { setToolboxLaunchResult(latestResult ?? gameLaunchResult); navigate('toolbox') }} />
    </Suspense>
  )
}

export default App
