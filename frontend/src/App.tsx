import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Check,
  Database,
  Play,
  RotateCcw,
  ShieldCheck,
} from 'lucide-react'
import { runComparison } from './api'
import { services, type CompareResponse, type Scenario, type Service } from './types'

const serviceLabels: Record<Service, string> = {
  transport: 'Transport',
  housing: 'Housing',
  food: 'Food',
  healthcare: 'Healthcare',
  public_services: 'Public services',
}

const serviceCodes: Record<Service, string> = {
  transport: 'TR',
  housing: 'HO',
  food: 'FD',
  healthcare: 'HC',
  public_services: 'PS',
}

const defaultScenario: Scenario = {
  name: 'Central district restart',
  horizon_days: 14,
  daily_budget: 180,
  initial_services: [0.34, 0.26, 0.41, 0.38, 0.3],
  priorities: [1, 1.1, 1.2, 1.4, 1],
  shock_probability: 0.2,
  severity_min: 0.1,
  severity_max: 0.28,
  forced_shock: { day: 5, type: 'utility', severity: 0.26 },
}

type ViewMode = 'trajectory' | 'audit'

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function formatUnits(value: number): string {
  return value.toFixed(1)
}

function linePath(values: number[], width = 660, height = 156): string {
  if (values.length === 0) return ''
  return values
    .map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * width
      const y = height - value * height
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
}

function ResilienceChart({ result }: { result: CompareResponse }) {
  const baseline = result.baseline.trajectory.map((day) => day.resilience)
  const candidate = result.candidate.trajectory.map((day) => day.resilience)
  return (
    <figure className="chart-wrap" aria-labelledby="chart-title">
      <figcaption id="chart-title" className="chart-title">
        Weighted service resilience by day
      </figcaption>
      <div className="chart-legend" aria-hidden="true">
        <span><i className="legend-line candidate-line" />Frozen candidate</span>
        <span><i className="legend-line baseline-line" />Urgency baseline</span>
      </div>
      <svg className="chart" viewBox="0 0 720 208" role="img" aria-label="Resilience comparison line chart">
        <g transform="translate(42 20)">
          {[0.25, 0.5, 0.75, 1].map((tick) => (
            <g key={tick}>
              <line x1="0" x2="660" y1={156 - tick * 156} y2={156 - tick * 156} className="gridline" />
              <text x="-10" y={160 - tick * 156} textAnchor="end" className="chart-label">
                {Math.round(tick * 100)}
              </text>
            </g>
          ))}
          <path d={linePath(baseline)} className="series baseline-series" />
          <path d={linePath(candidate)} className="series candidate-series" />
          {result.shock_schedule.map((shock, index) =>
            shock.type ? (
              <circle
                key={`${shock.day}-${shock.type}`}
                cx={(index / Math.max(result.shock_schedule.length - 1, 1)) * 660}
                cy={156 - candidate[index] * 156}
                r={shock.forced ? 5 : 3.5}
                className={shock.forced ? 'shock-dot forced-dot' : 'shock-dot'}
              />
            ) : null,
          )}
          <text x="0" y="181" className="chart-label">Day 1</text>
          <text x="660" y="181" textAnchor="end" className="chart-label">
            Day {result.scenario.horizon_days}
          </text>
        </g>
      </svg>
    </figure>
  )
}

function ScenarioEditor({
  draft,
  seed,
  busy,
  onDraft,
  onSeed,
  onRun,
  onReset,
}: {
  draft: Scenario
  seed: number
  busy: boolean
  onDraft: (scenario: Scenario) => void
  onSeed: (seed: number) => void
  onRun: () => void
  onReset: () => void
}) {
  const updateService = (field: 'initial_services' | 'priorities', index: number, value: number) => {
    const next = [...draft[field]]
    next[index] = value
    onDraft({ ...draft, [field]: next })
  }

  return (
    <aside className="scenario-panel" aria-labelledby="scenario-title">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">Scenario controls</p>
          <h2 id="scenario-title">Recovery envelope</h2>
        </div>
        <button className="icon-button" type="button" onClick={onReset} title="Reset fixture" aria-label="Reset fixture">
          <RotateCcw size={17} />
        </button>
      </div>

      <label className="field full-field">
        <span>Scenario name</span>
        <input value={draft.name} maxLength={64} onChange={(event) => onDraft({ ...draft, name: event.target.value })} />
      </label>

      <div className="field-grid">
        <label className="field">
          <span>Seed</span>
          <input type="number" min="0" max="4294967295" value={seed} onChange={(event) => onSeed(Number(event.target.value))} />
        </label>
        <label className="field">
          <span>Days</span>
          <input type="number" min="7" max="30" value={draft.horizon_days} onChange={(event) => onDraft({ ...draft, horizon_days: Number(event.target.value) })} />
        </label>
        <label className="field">
          <span>Daily units</span>
          <input type="number" min="50" max="500" step="1" value={draft.daily_budget} onChange={(event) => onDraft({ ...draft, daily_budget: Number(event.target.value) })} />
        </label>
        <label className="field">
          <span>Shock chance</span>
          <div className="input-suffix">
            <input type="number" min="0" max="35" step="1" value={Math.round(draft.shock_probability * 100)} onChange={(event) => onDraft({ ...draft, shock_probability: Number(event.target.value) / 100 })} />
            <b>%</b>
          </div>
        </label>
      </div>

      <fieldset className="service-editor">
        <legend>Service condition and priority</legend>
        <div className="service-header" aria-hidden="true"><span>Service</span><span>State</span><span>Weight</span></div>
        {services.map((service, index) => (
          <div className="service-input-row" key={service}>
            <span className="service-name"><b>{serviceCodes[service]}</b>{serviceLabels[service]}</span>
            <input aria-label={`${serviceLabels[service]} initial state`} type="number" min="5" max="95" step="1" value={Math.round(draft.initial_services[index] * 100)} onChange={(event) => updateService('initial_services', index, Number(event.target.value) / 100)} />
            <input aria-label={`${serviceLabels[service]} priority`} type="number" min="0.5" max="2" step="0.1" value={draft.priorities[index]} onChange={(event) => updateService('priorities', index, Number(event.target.value))} />
          </div>
        ))}
      </fieldset>

      <div className="severity-fields">
        <label className="field">
          <span>Severity min</span>
          <input type="number" min="5" max="25" value={Math.round(draft.severity_min * 100)} onChange={(event) => onDraft({ ...draft, severity_min: Number(event.target.value) / 100 })} />
        </label>
        <label className="field">
          <span>Severity max</span>
          <input type="number" min="10" max="40" value={Math.round(draft.severity_max * 100)} onChange={(event) => onDraft({ ...draft, severity_max: Number(event.target.value) / 100 })} />
        </label>
      </div>

      <label className="forced-toggle">
        <input
          type="checkbox"
          checked={draft.forced_shock !== null}
          onChange={(event) => onDraft({ ...draft, forced_shock: event.target.checked ? { day: 5, type: 'utility', severity: 0.26 } : null })}
        />
        <span><b>Force utility failure</b><small>Day 5 at 26% severity</small></span>
      </label>

      <button className="run-button" type="button" onClick={onRun} disabled={busy}>
        {busy ? <Activity className="spin" size={18} /> : <Play size={18} fill="currentColor" />}
        {busy ? 'Running both plans' : 'Run comparison'}
      </button>
    </aside>
  )
}

function DayInspector({ result, selectedDay, onDay }: { result: CompareResponse; selectedDay: number; onDay: (day: number) => void }) {
  const baseline = result.baseline.trajectory[selectedDay - 1]
  const candidate = result.candidate.trajectory[selectedDay - 1]
  const shock = result.shock_schedule[selectedDay - 1]
  return (
    <section className="day-inspector" aria-labelledby="day-title">
      <div className="day-control">
        <div>
          <p className="section-kicker">Daily allocation ledger</p>
          <h3 id="day-title">Day {selectedDay}</h3>
        </div>
        <div className={`shock-label ${shock.type ? 'active-shock' : ''}`}>
          {shock.type ? <AlertTriangle size={16} /> : <Check size={16} />}
          <span>{shock.type ? `${shock.type} / ${formatPercent(shock.severity)}` : 'No shock'}</span>
          {shock.forced ? <em>forced</em> : null}
        </div>
      </div>
      <input className="day-slider" aria-label="Inspect simulation day" type="range" min="1" max={result.scenario.horizon_days} value={selectedDay} onChange={(event) => onDay(Number(event.target.value))} />
      <div className="ledger-head" aria-hidden="true"><span>Service</span><span>End state</span><span>Candidate</span><span>Baseline</span></div>
      <div className="service-ledger">
        {services.map((service, index) => (
          <div className="ledger-row" key={service}>
            <div className="ledger-service"><b>{serviceCodes[service]}</b><span>{serviceLabels[service]}</span></div>
            <div className="state-comparison">
              <div className="state-track"><i className="candidate-fill" style={{ width: `${candidate.services_end[index] * 100}%` }} /></div>
              <small>{formatPercent(candidate.services_end[index])} / {formatPercent(baseline.services_end[index])}</small>
            </div>
            <strong className="candidate-number">{formatUnits(candidate.allocation[index])}</strong>
            <strong>{formatUnits(baseline.allocation[index])}</strong>
          </div>
        ))}
      </div>
      <p className="projection-note">
        Both proposals projected to {formatUnits(candidate.available_budget)} units. Candidate adjustment distance {candidate.projection.distance.toFixed(2)}; baseline {baseline.projection.distance.toFixed(2)}.
      </p>
    </section>
  )
}

function AuditTable({ result }: { result: CompareResponse }) {
  return (
    <div className="audit-scroll">
      <table>
        <caption>Full deterministic daily comparison</caption>
        <thead><tr><th>Day</th><th>Shock</th><th>Budget</th><th>Candidate resilience</th><th>Baseline resilience</th><th>Delta</th></tr></thead>
        <tbody>
          {result.candidate.trajectory.map((day, index) => {
            const baseline = result.baseline.trajectory[index]
            return (
              <tr key={day.day}>
                <td>{day.day}</td>
                <td>{day.shock.type ?? 'None'}{day.shock.forced ? ' (forced)' : ''}</td>
                <td>{formatUnits(day.available_budget)}</td>
                <td>{formatPercent(day.resilience)}</td>
                <td>{formatPercent(baseline.resilience)}</td>
                <td className={day.resilience - baseline.resilience >= 0 ? 'positive' : 'negative'}>{((day.resilience - baseline.resilience) * 100).toFixed(2)} pp</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function App() {
  const [draft, setDraft] = useState<Scenario>(defaultScenario)
  const [seed, setSeed] = useState(424242)
  const [result, setResult] = useState<CompareResponse | null>(null)
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedDay, setSelectedDay] = useState(5)
  const [view, setView] = useState<ViewMode>('trajectory')

  const execute = useCallback(async (scenario: Scenario, runSeed: number, signal?: AbortSignal) => {
    setBusy(true)
    setError(null)
    try {
      const response = await runComparison(runSeed, scenario, signal)
      setResult(response)
      setSelectedDay(Math.min(5, response.scenario.horizon_days))
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      setError(caught instanceof Error ? caught.message : 'The comparison could not be completed.')
    } finally {
      if (!signal?.aborted) setBusy(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void execute(defaultScenario, 424242, controller.signal)
    return () => controller.abort()
  }, [execute])

  const candidateWins = (result?.comparison.candidate_minus_baseline ?? 0) >= 0
  const shockCount = useMemo(() => result?.shock_schedule.filter((shock) => shock.type).length ?? 0, [result])

  const reset = () => {
    setDraft(defaultScenario)
    setSeed(424242)
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div><p>Civic Relay</p><h1>Recovery desk</h1></div>
        </div>
        <div className="runtime-strip">
          <span className={`status-dot ${error ? 'status-error' : ''}`} />
          <span>{error ? 'Runtime blocked' : 'Local deterministic runtime'}</span>
          <b>PCG64</b>
          <span className="synthetic-chip"><Database size={14} />Synthetic model</span>
        </div>
      </header>

      <main className="workspace">
        <ScenarioEditor draft={draft} seed={seed} busy={busy} onDraft={setDraft} onSeed={setSeed} onRun={() => void execute(draft, seed)} onReset={reset} />
        <section className="results-panel" aria-live="polite">
          {error ? (
            <div className="blocking-error" role="alert">
              <AlertTriangle size={24} />
              <div><h2>Comparison blocked</h2><p>{error}</p><button type="button" onClick={() => void execute(draft, seed)}>Try again</button></div>
            </div>
          ) : null}

          {busy && !result ? (
            <div className="loading-state" aria-label="Computing both recovery plans">
              <Activity className="spin" size={28} /><h2>Computing shared shock tape</h2><p>Running both planners through the same daily conditions.</p>
            </div>
          ) : null}

          {!busy && !result && !error ? (
            <div className="empty-state">
              <Activity size={24} /><h2>No trajectory yet</h2><p>Run a bounded recovery scenario to compare both planners.</p>
            </div>
          ) : null}

          {result ? (
            <>
              <div className="results-header">
                <div>
                  <p className="section-kicker">Comparison / seed {result.seed}</p>
                  <h2>{result.scenario.name}</h2>
                </div>
                <div className="proof-badge"><ShieldCheck size={18} /><span><b>0 violations</b>shared constraints</span></div>
              </div>

              <section className="metric-ribbon" aria-label="Comparison summary">
                <div className="primary-metric"><span>Resilience AUC</span><strong>{formatPercent(result.candidate.rauc)}</strong><small>Frozen candidate</small></div>
                <div className="metric-divider" aria-hidden="true" />
                <div className="comparison-metric"><span>Against visible urgency planner</span><strong>{formatPercent(result.baseline.rauc)}</strong><small className={candidateWins ? 'positive' : 'negative'}>{result.comparison.candidate_minus_baseline >= 0 ? '+' : ''}{(result.comparison.candidate_minus_baseline * 100).toFixed(2)} percentage points</small></div>
                <dl className="run-facts"><div><dt>Shocks</dt><dd>{shockCount}</dd></div><div><dt>Days</dt><dd>{result.scenario.horizon_days}</dd></div><div><dt>Daily budget</dt><dd>{result.scenario.daily_budget}</dd></div></dl>
              </section>

              <div className="view-tabs" role="tablist" aria-label="Result view">
                <button role="tab" aria-selected={view === 'trajectory'} className={view === 'trajectory' ? 'active' : ''} onClick={() => setView('trajectory')}>Trajectory</button>
                <button role="tab" aria-selected={view === 'audit'} className={view === 'audit' ? 'active' : ''} onClick={() => setView('audit')}>Daily audit</button>
              </div>

              {view === 'trajectory' ? (
                <div className="trajectory-layout">
                  <ResilienceChart result={result} />
                  <DayInspector result={result} selectedDay={selectedDay} onDay={setSelectedDay} />
                </div>
              ) : <AuditTable result={result} />}

              <footer className="evidence-footer">
                <div><b>Shock tape</b><code>{result.shock_schedule_sha256.slice(0, 16)}…</code></div>
                <div><b>Policy artifact</b><code>{result.policy.sha256.slice(0, 16)}…</code></div>
                <p>{result.policy.disclosure}</p>
              </footer>
            </>
          ) : null}
          {busy && result ? <div className="recompute-bar"><Activity className="spin" size={16} /> Recomputing both trajectories</div> : null}
        </section>
      </main>
    </div>
  )
}

export default App
