import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'
import {
  Activity,
  AlertTriangle,
  ArchiveRestore,
  Check,
  Database,
  Play,
  RotateCcw,
  Scale,
} from 'lucide-react'
import { ComparisonError, listSimulations, loadSimulation, runComparison } from './api'
import { CityGame } from './game/CityGame'
import { relayNarration } from './game/model'
import {
  dawnSitrep,
  decisionRationale,
  depotStatusesForDay,
  LEGACY_V1_DEPOT_DISCLOSURE,
  throughputVehiclesPerDay,
  vehicleManifestsForDay,
} from './game/realism'
import { REALISM_LEDGER } from './game/realismLedger'
import { PlanReviewPanel } from './operator/PlanReviewPanel'
import { defaultScenario, defaultSeed } from './scenarios'
import { shockDisplayName } from './shockPresentation'
import { V5DevelopmentPanel } from './v5/V5DevelopmentPanel'
import {
  services,
  type CompareResponse,
  type SavedResultSummary,
  type Scenario,
  type Service,
} from './types'

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

type ViewMode = 'trajectory' | 'audit' | 'dispatch' | 'decisions' | 'operator' | 'v5' | 'realism'
type ComparisonFailure = { code: string; message: string }

const viewModes: ViewMode[] = [
  'trajectory',
  'audit',
  'dispatch',
  'decisions',
  'operator',
  'v5',
  'realism',
]

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function formatUnits(value: number): string {
  return value.toFixed(1)
}

function measuredViolations(result: CompareResponse, planner: 'candidate' | 'baseline'): number {
  return result[planner].trajectory.reduce(
    (total, day) => total + day.projection.constraint_violations,
    0,
  )
}

function scenariosMatch(left: Scenario, right: Scenario): boolean {
  const leftShock = left.forced_shock
  const rightShock = right.forced_shock
  const leftShocks = left.forced_shocks ?? []
  const rightShocks = right.forced_shocks ?? []

  return left.name === right.name
    && left.horizon_days === right.horizon_days
    && left.daily_budget === right.daily_budget
    && left.shock_probability === right.shock_probability
    && left.severity_min === right.severity_min
    && left.severity_max === right.severity_max
    && left.initial_services.every((value, index) => value === right.initial_services[index])
    && left.priorities.every((value, index) => value === right.priorities[index])
    && leftShocks.length === rightShocks.length
    && leftShocks.every((shock, index) => (
      shock.day === rightShocks[index]?.day
      && shock.type === rightShocks[index]?.type
      && shock.severity === rightShocks[index]?.severity
    ))
    && ((leftShock === null && rightShock === null) || (
      leftShock !== null
      && rightShock !== null
      && leftShock.day === rightShock.day
      && leftShock.type === rightShock.type
      && leftShock.severity === rightShock.severity
    ))
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

function ResilienceChart({ result, selectedDay }: { result: CompareResponse; selectedDay: number }) {
  const baseline = result.baseline.trajectory.map((day) => day.resilience)
  const candidate = result.candidate.trajectory.map((day) => day.resilience)
  const selectedIndex = selectedDay - 1
  const selectedX = (selectedIndex / Math.max(result.scenario.horizon_days - 1, 1)) * 660
  const shockCount = result.shock_schedule.filter((shock) => shock.type).length
  return (
    <figure className="chart-wrap" aria-labelledby="chart-title">
      <figcaption id="chart-title" className="chart-title">
        Weighted service resilience by day
      </figcaption>
      <div className="chart-legend" aria-hidden="true">
        <span><i className="legend-line candidate-line" />SB3 PPO / ONNX</span>
        <span><i className="legend-line baseline-line" />OR-Tools GLOP</span>
        <span><i className="legend-shock-pair"><b /><b /></i>Shared / forced shocks</span>
        <span><i className="legend-selected" />Selected day</span>
      </div>
      <svg
        className="chart"
        viewBox="0 0 720 208"
        role="img"
        aria-label={`Resilience comparison line chart with ${shockCount} shared shocks; day ${selectedDay} selected`}
      >
        <g transform="translate(42 20)">
          {[0.25, 0.5, 0.75, 1].map((tick) => (
            <g key={tick}>
              <line x1="0" x2="660" y1={156 - tick * 156} y2={156 - tick * 156} className="gridline" />
              <text x="-10" y={160 - tick * 156} textAnchor="end" className="chart-label">
                {Math.round(tick * 100)}
              </text>
            </g>
          ))}
          {result.shock_schedule.map((shock, index) => {
            if (!shock.type) return null
            const x = (index / Math.max(result.shock_schedule.length - 1, 1)) * 660
            return (
              <g key={`${shock.day}-${shock.type}`}>
                <line
                  x1={x}
                  x2={x}
                  y1="4"
                  y2="156"
                  className={shock.forced ? 'shared-shock-guide forced-shock-guide' : 'shared-shock-guide'}
                />
                <circle
                  cx={x}
                  cy="4"
                  r={shock.forced ? 4.5 : 3.5}
                  className={shock.forced ? 'shared-shock-dot forced-dot' : 'shared-shock-dot'}
                >
                  <title>{`Day ${shock.day}: ${shockDisplayName(shock.type)}, ${formatPercent(shock.severity)}${shock.forced ? ', forced' : ''}`}</title>
                </circle>
              </g>
            )
          })}
          <line x1={selectedX} x2={selectedX} y1="0" y2="156" className="selected-day-guide" />
          <path d={linePath(baseline)} className="series baseline-series" />
          <path d={linePath(candidate)} className="series candidate-series" />
          <circle cx={selectedX} cy={156 - candidate[selectedIndex] * 156} r="4" className="selected-point candidate-point" />
          <circle cx={selectedX} cy={156 - baseline[selectedIndex] * 156} r="3.5" className="selected-point baseline-point" />
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
  savedRuns,
  onRestore,
}: {
  draft: Scenario
  seed: number
  busy: boolean
  onDraft: (scenario: Scenario) => void
  onSeed: (seed: number) => void
  onRun: () => void
  onReset: () => void
  savedRuns: SavedResultSummary[]
  onRestore: (resultId: string) => void
}) {
  const updateService = (field: 'initial_services' | 'priorities', index: number, value: number) => {
    const next = [...draft[field]]
    next[index] = value
    onDraft({ ...draft, [field]: next })
  }

  return (
    <aside className="scenario-panel" aria-labelledby="scenario-title">
      <div className="scenario-scroll">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">Scenario controls</p>
          <h2 id="scenario-title" tabIndex={-1}>Recovery envelope</h2>
        </div>
        <button
          className="icon-button"
          type="button"
          onClick={() => { if (!busy) onReset() }}
          title="Reset fixture"
          aria-label="Reset fixture"
          aria-disabled={busy}
        >
          <RotateCcw size={17} />
        </button>
      </div>

      <label className="field full-field">
        <span>Scenario name</span>
        <input value={draft.name} required minLength={1} maxLength={64} onChange={(event) => onDraft({ ...draft, name: event.target.value })} />
      </label>

      <label className="field full-field saved-result-field">
        <span><ArchiveRestore size={13} />Saved results</span>
        <select
          aria-label="Restore saved result"
          aria-disabled={busy || savedRuns.length === 0}
          value=""
          disabled={savedRuns.length === 0}
          onChange={(event) => {
            if (!busy && event.target.value) onRestore(event.target.value)
          }}
        >
          <option value="">{savedRuns.length === 0 ? 'No saved results' : `Restore one of ${savedRuns.length}`}</option>
          {savedRuns.map((saved) => (
            <option key={saved.result_id} value={saved.result_id}>
              {saved.scenario_name} · seed {saved.seed} · {saved.result_id.slice(0, 8)}
            </option>
          ))}
        </select>
      </label>

      <div className="field-grid">
        <label className="field">
          <span>Seed</span>
          <input type="number" min="0" max="4294967295" value={seed} onChange={(event) => onSeed(Number(event.target.value))} />
        </label>
        <label className="field">
          <span>Days</span>
          <input id="horizon-days" type="number" min="7" max="30" value={draft.horizon_days} onChange={(event) => onDraft({ ...draft, horizon_days: Number(event.target.value) })} />
        </label>
        <label className="field">
          <span>Daily units</span>
          <input id="daily-budget" type="number" min="50" max="500" step="1" value={draft.daily_budget} onChange={(event) => onDraft({ ...draft, daily_budget: Number(event.target.value) })} />
        </label>
        <label className="field">
          <span>Shock chance</span>
          <div className="input-suffix">
            <input id="shock-probability" type="number" min="0" max="35" step="1" value={Math.round(draft.shock_probability * 100)} onChange={(event) => onDraft({ ...draft, shock_probability: Number(event.target.value) / 100 })} />
            <b>%</b>
          </div>
        </label>
      </div>

      <fieldset className="service-editor">
        <legend>Service condition and priority</legend>
        <div className="service-header" aria-hidden="true"><span>Service</span><span>State %</span><span>Weight</span></div>
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
          <span>Severity min %</span>
          <input id="severity-min" type="number" min="5" max="25" value={Math.round(draft.severity_min * 100)} onChange={(event) => onDraft({ ...draft, severity_min: Number(event.target.value) / 100 })} />
        </label>
        <label className="field">
          <span>Severity max %</span>
          <input id="severity-max" type="number" min="10" max="40" value={Math.round(draft.severity_max * 100)} onChange={(event) => onDraft({ ...draft, severity_max: Number(event.target.value) / 100 })} />
        </label>
      </div>

      <label className="forced-toggle">
        <input
          type="checkbox"
          checked={draft.forced_shock !== null}
          onChange={(event) => onDraft({ ...draft, forced_shock: event.target.checked ? { day: 5, type: 'utility', severity: 0.26 } : null })}
        />
        <span className="checkbox-control" aria-hidden="true"><Check size={14} /></span>
        <span className="forced-copy"><b>Force utility failure</b><small>Day 5 at 26% severity</small></span>
      </label>
      </div>

      <div className="scenario-action-dock">
        <button className="run-button" type="button" onClick={() => { if (!busy) onRun() }} aria-disabled={busy}>
          {busy ? <Activity className="spin" size={18} /> : <Play size={18} fill="currentColor" />}
          {busy ? 'Running both plans' : 'Run comparison'}
        </button>
      </div>
    </aside>
  )
}

function DayInspector({ result, selectedDay, onDay }: { result: CompareResponse; selectedDay: number; onDay: (day: number) => void }) {
  const baseline = result.baseline.trajectory[selectedDay - 1]
  const candidate = result.candidate.trajectory[selectedDay - 1]
  const shock = result.shock_schedule[selectedDay - 1]
  return (
    <section className={`day-inspector ${shock.type ? 'shock-day' : 'steady-day'}`} aria-labelledby="day-title">
      <div className="day-control">
        <div>
          <p className="section-kicker">Daily allocation ledger</p>
          <h3 id="day-title">Day {selectedDay}</h3>
        </div>
        <div className={`shock-label ${shock.type ? 'active-shock' : ''}`}>
          {shock.type ? <AlertTriangle size={16} /> : <Check size={16} />}
          <span>{shock.type ? `${shockDisplayName(shock.type)} / ${formatPercent(shock.severity)}` : 'No shock'}</span>
          {shock.forced ? <em>forced</em> : null}
        </div>
      </div>
      <input className="day-slider" aria-label="Inspect simulation day" type="range" min="1" max={result.scenario.horizon_days} value={selectedDay} onChange={(event) => onDay(Number(event.target.value))} />
      <div className="ledger-table" role="table" aria-label={`Day ${selectedDay} service outcomes and allocations`}>
        <div className="ledger-head" role="row">
          <span role="columnheader">Service</span>
          <span role="columnheader">End state</span>
          <span className="candidate-column" role="columnheader">Candidate<small>units</small></span>
          <span className="baseline-column" role="columnheader">Baseline<small>units</small></span>
        </div>
        <div className="service-ledger" role="rowgroup">
          {services.map((service, index) => (
            <div className="ledger-row" role="row" key={service}>
              <div className="ledger-service" role="rowheader" aria-label={`${serviceCodes[service]} ${serviceLabels[service]}`}><b>{serviceCodes[service]}</b><span>{serviceLabels[service]}</span></div>
              <div
                className="state-comparison"
                role="cell"
                aria-label={`End state: candidate ${formatPercent(candidate.services_end[index])}, baseline ${formatPercent(baseline.services_end[index])}`}
              >
                <div className="state-tracks" aria-hidden="true">
                  <span className="state-track"><i className="candidate-fill" style={{ width: `${candidate.services_end[index] * 100}%` }} /></span>
                  <span className="state-track"><i className="baseline-fill" style={{ width: `${baseline.services_end[index] * 100}%` }} /></span>
                </div>
                <small aria-hidden="true"><span className="candidate-state">C {formatPercent(candidate.services_end[index])}</span><span className="baseline-state">B {formatPercent(baseline.services_end[index])}</span></small>
              </div>
              <strong className="candidate-number" role="cell">{formatUnits(candidate.allocation[index])}</strong>
              <strong className="baseline-number" role="cell">{formatUnits(baseline.allocation[index])}</strong>
            </div>
          ))}
        </div>
      </div>
      <p className="projection-note">
        Shared available budget {formatUnits(candidate.available_budget)} units. Projection distance: candidate {candidate.projection.distance.toFixed(2)} / baseline {baseline.projection.distance.toFixed(2)}.
      </p>
    </section>
  )
}

function AuditTable({ result }: { result: CompareResponse }) {
  return (
    <div className="audit-view">
      <p className="audit-cue">Scroll for candidate, baseline, and delta <span aria-hidden="true">→</span></p>
      <div className="audit-scroll" role="region" aria-label="Scrollable daily comparison table" tabIndex={0}>
        <table>
          <caption>Full deterministic daily comparison</caption>
          <thead><tr><th>Day</th><th>Shock</th><th>Budget</th><th>Candidate resilience</th><th>Baseline resilience</th><th>Delta</th></tr></thead>
          <tbody>
            {result.candidate.trajectory.map((day, index) => {
              const baseline = result.baseline.trajectory[index]
              return (
                <tr key={day.day}>
                  <td>{day.day}</td>
                  <td>{day.shock.type ? shockDisplayName(day.shock.type) : 'None'}{day.shock.forced ? ' (forced)' : ''}</td>
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
    </div>
  )
}

function DispatchManifestView({ result, selectedDay }: { result: CompareResponse; selectedDay: number }) {
  const dayIndex = selectedDay - 1
  const day = result.candidate.trajectory[dayIndex]
  const previous = result.candidate.trajectory[dayIndex - 1]
  const depots = depotStatusesForDay(day, result.services)
  const manifests = vehicleManifestsForDay(day, previous, result.services)
  const inbound = manifests.filter((manifest) => manifest.wave === 'inbound')
  const inboundUnits = inbound.reduce((total, manifest) => total + manifest.cargoUnits, 0)
  const allocationUnits = day.allocation.reduce((total, allocation) => total + allocation, 0)

  return (
    <section className="evidence-view dispatch-view" aria-labelledby="dispatch-view-title">
      <header className="evidence-view-heading">
        <div>
          <p className="section-kicker">Complete quantity-derived manifest</p>
          <h3 id="dispatch-view-title">Day {selectedDay} dispatch manifest</h3>
        </div>
        <p><b>1 unit = 1 pallet.</b> Each load partitions the exact returned supply total; no vehicle creates additional supply. The diorama selects a deterministic bounded road subset while this table remains complete. {day.logistics ? 'Schema v3 stock, landed arrivals, held/overflow queue, throughput, mutual aid, repair dispatch, and effective repair supply are engine-recorded; vehicle load equivalents are derived from those quantities.' : LEGACY_V1_DEPOT_DISCLOSURE}</p>
      </header>

      <div className="dispatch-facts" aria-label={`Day ${selectedDay} dispatch totals`}>
        <article>
          <span>Inbound at intake hub</span>
          <strong>{formatUnits(inboundUnits)} <small>units = pallets</small></strong>
          <em>{inbound.length} edge-to-hub loads</em>
        </article>
        <article>
          <span>Allocated onward</span>
          <strong>{formatUnits(allocationUnits)} <small>units = pallets</small></strong>
          <em>full daily allocation, never held back</em>
        </article>
        <article>
          <span>{day.logistics ? 'Quantity-derived load equivalents' : 'Presentation load equivalents'}</span>
          <strong>{throughputVehiclesPerDay(day)} <small>load equivalents / day</small></strong>
          <em>{day.logistics ? 'complete line-haul + mutual aid + last-mile manifest' : 'allocation split into 12-unit line-haul and 4-unit last-mile presentation'}</em>
        </article>
      </div>

      <p className="evidence-scroll-cue">Scroll horizontally for the complete manifest <span aria-hidden="true">→</span></p>
      <div className="evidence-table-scroll" role="region" aria-label="Scrollable depot dispatch summary" tabIndex={0}>
        <table className="dispatch-summary-table">
          <caption>Day {selectedDay} dispatch summary by district depot</caption>
          <thead>
            <tr><th>Point of distribution</th><th>{day.logistics ? 'Allocation / stock' : 'Exact assignment'}</th><th>Line-haul loads</th><th>Last-mile loads</th><th>Throughput</th><th>Constrained queue / inbound window</th></tr>
          </thead>
          <tbody>
            {depots.map((depot) => {
              const lineHaul = manifests.filter((manifest) => manifest.wave === 'line-haul' && manifest.service === depot.service)
              const mutualAid = manifests.filter((manifest) => manifest.wave === 'mutual-aid' && manifest.service === depot.service)
              const lastMile = manifests.filter((manifest) => manifest.wave === 'last-mile' && manifest.service === depot.service)
              return (
                <tr key={depot.service}>
                  <td><b>{depot.district}</b><small>{day.logistics ? `${depot.damage} · ${depot.inboundWindow}` : depot.source}</small></td>
                  <td>{day.logistics
                    ? <>{formatUnits(depot.allocationUnits)} allocated<small>{formatUnits(depot.palletUnits ?? 0)} / {formatUnits(depot.stockCapacity ?? 0)} units in stock</small></>
                    : <>{formatUnits(depot.allocationUnits)} assigned units<small>allocation-backed presentation; depot stock unavailable</small></>}</td>
                  <td><b>{lineHaul.length} load equivalents</b><small>{lineHaul.map((manifest) => formatUnits(manifest.cargoUnits)).join(' + ')} units</small>{mutualAid.length ? <small>{mutualAid.length} quantity-derived mutual-aid loads · {formatUnits(mutualAid.reduce((sum, manifest) => sum + manifest.cargoUnits, 0))} units</small> : null}</td>
                  <td><b>{lastMile.length} load equivalents</b><small>{lastMile.map((manifest) => formatUnits(manifest.cargoUnits)).join(' + ')} units</small></td>
                  <td>{depot.engineTruth ? formatPercent(depot.throughputSignal ?? 0) : 'Unavailable'}<small>{depot.engineTruth ? 'engine-v2 effective throughput' : 'legacy v1 did not record depot throughput'}</small></td>
                  <td>{depot.engineTruth ? depot.dockQueue : 'Unavailable'}<small>{depot.engineTruth ? `${formatUnits(depot.dockQueueUnits ?? 0)} constrained units · ${formatUnits(depot.pendingUnits ?? 0)} scheduled/held next day` : 'legacy v1 did not record depot queues'}</small></td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="manifest-detail-heading">
        <div><p className="section-kicker">Load-level evidence</p><h4>Every quantity-derived mission leg</h4></div>
        <p>Each row names its source stage. Schema v3 separates landed freight, mutual-aid transfers, and effective repair supply; displayed legs are evidence, not additive supply.</p>
      </div>
      <div className="evidence-table-scroll" role="region" aria-label="Scrollable exact vehicle manifest" tabIndex={0}>
        <table className="manifest-detail-table">
          <caption>Exact load-to-vehicle manifest for day {selectedDay}</caption>
          <thead><tr><th>Mission</th><th>Wave</th><th>Fleet</th><th>Origin → destination</th><th>Cargo</th><th>Return leg</th></tr></thead>
          <tbody>
            {manifests.map((manifest) => (
              <tr key={manifest.id}>
                <td><code>{manifest.id}</code></td>
                <td>{manifest.wave}</td>
                <td>{manifest.fleet}</td>
                <td>{manifest.origin} <span aria-hidden="true">→</span> {manifest.destination}{manifest.scheduledStop ? <small>{manifest.scheduledStop}</small> : null}</td>
                <td>{formatUnits(manifest.cargoUnits)} units = {formatUnits(manifest.cargoPallets)} pallets</td>
                <td>{manifest.returnLeg}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function DecisionLogView({ result }: { result: CompareResponse }) {
  return (
    <section className="evidence-view decision-view" aria-labelledby="decision-view-title">
      <header className="evidence-view-heading">
        <div>
          <p className="section-kicker">Candidate / all trajectory days</p>
          <h3 id="decision-view-title">RELAY decision log</h3>
        </div>
        <p>Every line is generated from that day’s returned arrival, service deficits, and allocation. It is an explanation of recorded output, not a hidden forecast.</p>
      </header>
      <p className="evidence-scroll-cue">Scroll horizontally for the five-way allocation <span aria-hidden="true">→</span></p>
      <div className="evidence-table-scroll" role="region" aria-label="Scrollable RELAY decision log" tabIndex={0}>
        <table className="decision-log-table">
          <caption>Daily candidate allocations and derived rationale</caption>
          <thead><tr><th>Day</th><th>Dawn sitrep and rationale</th><th>Five-way allocation</th><th>Allocated total</th></tr></thead>
          <tbody>
            {result.candidate.trajectory.map((day) => (
              <tr key={day.day}>
                <td><b>Day {day.day}</b>{day.shock.type ? <small>{shockDisplayName(day.shock.type)} · {day.shock.severity.toFixed(2)}</small> : <small>clear operations</small>}</td>
                <td>
                  <b>{dawnSitrep(day, result.services)}</b>
                  <small>{decisionRationale(day, result.services, result.scenario.priorities)}</small>
                  <small>Full derived operations note: {relayNarration(result, day.day - 1)}</small>
                </td>
                <td>
                  <span className="allocation-sequence">
                    {result.services.map((service, index) => (
                      <span key={service}><b>{serviceCodes[service]}</b> {formatUnits(day.allocation[index])}</span>
                    ))}
                  </span>
                </td>
                <td>{formatUnits(day.allocation.reduce((total, allocation) => total + allocation, 0))} units<small>{formatUnits(day.available_budget)} arrived</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function RealismLedgerView() {
  return (
    <section className="evidence-view realism-view" aria-labelledby="realism-view-title">
      <header className="evidence-view-heading">
        <div>
          <p className="section-kicker">Anti-fabrication proof</p>
          <h3 id="realism-view-title">Realism ledger</h3>
        </div>
        <p>Every visible behavior family names its source, deterministic derivation, and the claim it explicitly does not make.</p>
      </header>
      <p className="evidence-scroll-cue">Scroll horizontally for derivation limits <span aria-hidden="true">→</span></p>
      <div className="evidence-table-scroll" role="region" aria-label="Scrollable realism ledger" tabIndex={0}>
        <table className="realism-ledger-table">
          <caption>Visible behavior to simulation-data source mapping</caption>
          <thead><tr><th>Behavior</th><th>Items</th><th>Source</th><th>Derivation</th><th>Does not mean</th></tr></thead>
          <tbody>
            {REALISM_LEDGER.map((entry) => (
              <tr key={entry.behavior}>
                <td><b>{entry.behavior}</b></td>
                <td><code>{entry.items}</code></td>
                <td><code>{entry.source}</code></td>
                <td>{entry.derivation}</td>
                <td>{entry.doesNotMean}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function AnalystToolbox({
  onOpenGame,
  initialResult,
}: {
  onOpenGame: (result: CompareResponse | null) => void
  initialResult?: CompareResponse | null
}) {
  const [draft, setDraft] = useState<Scenario>(initialResult?.scenario ?? defaultScenario)
  const [seed, setSeed] = useState(initialResult?.seed ?? defaultSeed)
  const [result, setResult] = useState<CompareResponse | null>(initialResult ?? null)
  const [busy, setBusy] = useState(!initialResult)
  const [error, setError] = useState<ComparisonFailure | null>(null)
  const [selectedDay, setSelectedDay] = useState(initialResult ? Math.min(5, initialResult.scenario.horizon_days) : 5)
  const [view, setView] = useState<ViewMode>('trajectory')
  const [savedRuns, setSavedRuns] = useState<SavedResultSummary[]>([])
  const errorRef = useRef<HTMLDivElement>(null)

  const execute = useCallback(async (scenario: Scenario, runSeed: number, signal?: AbortSignal) => {
    setBusy(true)
    setError(null)
    try {
      const response = await runComparison(runSeed, scenario, signal)
      setResult(response)
      setDraft(response.scenario)
      setSelectedDay(Math.min(5, response.scenario.horizon_days))
      setSavedRuns(await listSimulations(signal))
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      setResult(null)
      setError({
        code: caught instanceof ComparisonError ? caught.code : 'RUNTIME_UNREACHABLE',
        message: caught instanceof Error ? caught.message : 'The comparison could not be completed.',
      })
    } finally {
      if (!signal?.aborted) setBusy(false)
    }
  }, [])

  const restore = useCallback(async (resultId: string) => {
    setBusy(true)
    setResult(null)
    setError(null)
    try {
      const response = await loadSimulation(resultId)
      setResult(response)
      setDraft(response.scenario)
      setSeed(response.seed)
      setSelectedDay(Math.min(5, response.scenario.horizon_days))
      setView('trajectory')
      setSavedRuns(await listSimulations())
    } catch (caught) {
      setError({
        code: caught instanceof ComparisonError ? caught.code : 'PERSISTENCE_FAILED',
        message: caught instanceof Error ? caught.message : 'The saved result could not be restored.',
      })
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    if (initialResult) {
      void listSimulations(controller.signal).then(setSavedRuns).catch(() => undefined)
      return () => controller.abort()
    }
    void execute(defaultScenario, defaultSeed, controller.signal)
    return () => controller.abort()
  }, [execute, initialResult])

  useEffect(() => {
    if (!error) return
    errorRef.current?.focus()
    errorRef.current?.scrollIntoView?.({ block: 'center', behavior: 'auto' })
  }, [error])

  const candidateWins = (result?.comparison.candidate_minus_baseline ?? 0) >= 0
  const shockCount = result?.shock_schedule.filter((shock) => shock.type).length ?? 0
  const runtimeBlocked = error !== null && error.code !== 'INVALID_SCENARIO'
  const candidateViolationTotal = result ? measuredViolations(result, 'candidate') : 0
  const baselineViolationTotal = result ? measuredViolations(result, 'baseline') : 0
  const draftChanged = result !== null && (
    seed !== result.seed || !scenariosMatch(draft, result.scenario)
  )

  const reset = () => {
    setDraft(defaultScenario)
    setSeed(defaultSeed)
    setResult(null)
    setError(null)
    setSelectedDay(5)
    setView('trajectory')
  }

  const handleTabKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    const currentIndex = viewModes.indexOf(view)
    let nextView: ViewMode | null = null
    if (event.key === 'ArrowRight') nextView = viewModes[(currentIndex + 1) % viewModes.length]
    if (event.key === 'ArrowLeft') nextView = viewModes[(currentIndex - 1 + viewModes.length) % viewModes.length]
    if (event.key === 'Home') nextView = viewModes[0]
    if (event.key === 'End') nextView = viewModes[viewModes.length - 1]
    if (nextView === null) return
    event.preventDefault()
    setView(nextView)
    document.getElementById(`tab-${nextView}`)?.focus()
  }

  const reviewScenarioControls = () => {
    const invalidField = document.querySelector<HTMLElement>('input:invalid, select:invalid')
    const fieldHints: [string, string][] = [
      ['severity_min', '#severity-min'],
      ['severity_max', '#severity-max'],
      ['horizon_days', '#horizon-days'],
      ['daily_budget', '#daily-budget'],
      ['shock_probability', '#shock-probability'],
    ]
    const hintedSelector = fieldHints.find(([hint]) => error?.message.includes(hint))?.[1]
    const hintedField = hintedSelector ? document.querySelector<HTMLElement>(hintedSelector) : null
    const target = invalidField ?? hintedField ?? document.getElementById('scenario-title')
    target?.focus()
    target?.scrollIntoView?.({ block: 'center', behavior: 'auto' })
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /><span /><span /></div>
          <div><p>RELAY / evidence room</p><h1>Analyst Toolbox</h1></div>
        </div>
        <div className="toolbox-top-actions">
          <button
            type="button"
            className="city-switch"
            disabled={busy || !result || draftChanged}
            title={draftChanged ? 'Run the current draft before opening Relay City' : undefined}
            onClick={() => onOpenGame(result)}
          >
            <Play size={15} fill="currentColor" />Relay City view
          </button>
          <div className={`runtime-strip ${runtimeBlocked ? 'runtime-blocked' : ''}`}>
            <span className={`status-dot ${runtimeBlocked ? 'status-error' : ''}`} />
            <span className="runtime-label">{runtimeBlocked ? 'Policy blocked' : 'Local deterministic PPO'}</span>
            <span className="runtime-mobile-label" aria-hidden="true">{busy ? 'Running' : 'Local'}</span>
            <b>ONNX / PCG64</b>
            <span className="synthetic-chip"><Database size={14} />Synthetic model</span>
          </div>
        </div>
      </header>

      <main className="workspace">
        <ScenarioEditor
          draft={draft}
          seed={seed}
          busy={busy}
          onDraft={setDraft}
          onSeed={setSeed}
          onRun={() => void execute(draft, seed)}
          onReset={reset}
          savedRuns={savedRuns}
          onRestore={(resultId) => void restore(resultId)}
        />
        <section className="results-panel" aria-live="polite" aria-busy={busy}>
          {error ? (
            <div ref={errorRef} className={`blocking-error ${error.code === 'INVALID_SCENARIO' ? 'invalid-error' : ''}`} role="alert" tabIndex={-1}>
              <AlertTriangle size={24} aria-hidden="true" />
              <div>
                <h2>{error.code === 'INVALID_SCENARIO' ? 'Scenario invalid' : 'Comparison blocked'}</h2>
                <p>{error.message}</p>
                <button type="button" onClick={error.code === 'INVALID_SCENARIO' ? reviewScenarioControls : () => void execute(draft, seed)}>
                  {error.code === 'INVALID_SCENARIO' ? 'Review scenario controls' : 'Try again'}
                </button>
              </div>
            </div>
          ) : null}

          {busy && !result ? (
            <div className="loading-state" role="status" aria-label="Computing both recovery plans">
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
                <div className="result-statuses">
                  {draftChanged ? <div className="draft-status" role="status"><b>Draft changed</b><span>Run to refresh evidence</span></div> : null}
                  <div
                    className="proof-badge"
                    aria-label={`Measured constraint violations: candidate ${candidateViolationTotal}, baseline ${baselineViolationTotal}`}
                  >
                    <Scale size={18} />
                    <span><b>Candidate {candidateViolationTotal} / baseline {baselineViolationTotal}</b>Measured constraint violations</span>
                  </div>
                </div>
              </div>

              <section className="metric-ribbon" aria-label="Comparison summary">
                <div className="primary-metric"><span>Candidate resilience AUC</span><strong>{formatPercent(result.candidate.rauc)}</strong><small>SB3 PPO / ONNX</small></div>
                <div className="metric-divider" aria-hidden="true" />
                <div className="comparison-metric">
                  <span>Baseline resilience AUC</span>
                  <strong>{formatPercent(result.baseline.rauc)}</strong>
                  <small>Visible OR-Tools GLOP</small>
                  <em className={`delta-readout ${candidateWins ? 'positive' : 'negative'}`}>Measured delta {result.comparison.candidate_minus_baseline >= 0 ? '+' : ''}{(result.comparison.candidate_minus_baseline * 100).toFixed(2)} pp</em>
                </div>
                <dl className="run-facts">
                  <div><dt>Shocks</dt><dd>{shockCount}</dd></div>
                  <div>
                    <dt>Recovery days</dt>
                    <dd aria-label={`Candidate ${result.candidate.days_to_pre_shock_recovery_after_largest_loss}, baseline ${result.baseline.days_to_pre_shock_recovery_after_largest_loss}`}>
                      <span className="planner-key candidate-key" aria-hidden="true">C</span> {result.candidate.days_to_pre_shock_recovery_after_largest_loss}
                      <span aria-hidden="true"> / </span>
                      <span className="planner-key baseline-key" aria-hidden="true">B</span> {result.baseline.days_to_pre_shock_recovery_after_largest_loss}
                    </dd>
                  </div>
                  <div><dt>Daily budget</dt><dd>{result.scenario.daily_budget}</dd></div>
                </dl>
              </section>

              <div className="view-tabs" role="tablist" aria-label="Result view">
                <button id="tab-trajectory" type="button" role="tab" aria-selected={view === 'trajectory'} aria-controls="panel-trajectory" tabIndex={view === 'trajectory' ? 0 : -1} className={view === 'trajectory' ? 'active' : ''} onClick={() => setView('trajectory')} onKeyDown={handleTabKey}>Trajectory</button>
                <button id="tab-audit" type="button" role="tab" aria-selected={view === 'audit'} aria-controls="panel-audit" tabIndex={view === 'audit' ? 0 : -1} className={view === 'audit' ? 'active' : ''} onClick={() => setView('audit')} onKeyDown={handleTabKey}>Daily audit</button>
                <button id="tab-dispatch" type="button" role="tab" aria-selected={view === 'dispatch'} aria-controls="panel-dispatch" tabIndex={view === 'dispatch' ? 0 : -1} className={view === 'dispatch' ? 'active' : ''} onClick={() => setView('dispatch')} onKeyDown={handleTabKey}>Dispatch manifest</button>
                <button id="tab-decisions" type="button" role="tab" aria-selected={view === 'decisions'} aria-controls="panel-decisions" tabIndex={view === 'decisions' ? 0 : -1} className={view === 'decisions' ? 'active' : ''} onClick={() => setView('decisions')} onKeyDown={handleTabKey}>Decision log</button>
                <button id="tab-operator" type="button" role="tab" aria-selected={view === 'operator'} aria-controls="panel-operator" tabIndex={view === 'operator' ? 0 : -1} className={view === 'operator' ? 'active' : ''} onClick={() => setView('operator')} onKeyDown={handleTabKey}>Operator review</button>
                <button id="tab-v5" type="button" role="tab" aria-selected={view === 'v5'} aria-controls="panel-v5" tabIndex={view === 'v5' ? 0 : -1} className={view === 'v5' ? 'active' : ''} onClick={() => setView('v5')} onKeyDown={handleTabKey}>v5 preview</button>
                <button id="tab-realism" type="button" role="tab" aria-selected={view === 'realism'} aria-controls="panel-realism" tabIndex={view === 'realism' ? 0 : -1} className={view === 'realism' ? 'active' : ''} onClick={() => setView('realism')} onKeyDown={handleTabKey}>Realism ledger</button>
              </div>

              <div id="panel-trajectory" role="tabpanel" aria-labelledby="tab-trajectory" hidden={view !== 'trajectory'}>
                <div className="trajectory-layout">
                  <ResilienceChart result={result} selectedDay={selectedDay} />
                  <DayInspector result={result} selectedDay={selectedDay} onDay={setSelectedDay} />
                </div>
              </div>
              <div id="panel-audit" role="tabpanel" aria-labelledby="tab-audit" hidden={view !== 'audit'}>
                <AuditTable result={result} />
              </div>
              <div id="panel-dispatch" role="tabpanel" aria-labelledby="tab-dispatch" hidden={view !== 'dispatch'}>
                <DispatchManifestView result={result} selectedDay={selectedDay} />
              </div>
              <div id="panel-decisions" role="tabpanel" aria-labelledby="tab-decisions" hidden={view !== 'decisions'}>
                <DecisionLogView result={result} />
              </div>
              <div id="panel-operator" role="tabpanel" aria-labelledby="tab-operator" hidden={view !== 'operator'}>
                <PlanReviewPanel result={result} />
              </div>
              <div id="panel-v5" role="tabpanel" aria-labelledby="tab-v5" hidden={view !== 'v5'}>
                {view === 'v5' ? <V5DevelopmentPanel /> : null}
              </div>
              <div id="panel-realism" role="tabpanel" aria-labelledby="tab-realism" hidden={view !== 'realism'}>
                <RealismLedgerView />
              </div>

              <footer className="evidence-footer">
                <div><b>Shock tape</b><code>{result.shock_schedule_sha256.slice(0, 16)}…</code></div>
                <div><b>ONNX policy</b><code>{result.policy.sha256.slice(0, 16)}…</code></div>
                <p>{result.policy.disclosure} {result.policy.legacy_candidate.disclosure}</p>
              </footer>
            </>
          ) : null}
          {busy && result ? <div className="recompute-bar" role="status"><Activity className="spin" size={16} /> Recomputing both trajectories</div> : null}
        </section>
      </main>
    </div>
  )
}

type AppRoute = 'game' | 'toolbox'

function routeFromHash(): AppRoute {
  return window.location.hash.toLowerCase().startsWith('#/toolbox') ? 'toolbox' : 'game'
}

function App() {
  const [route, setRoute] = useState<AppRoute>(routeFromHash)
  const [latestResult, setLatestResult] = useState<CompareResponse | null>(null)
  const [gameLaunchResult, setGameLaunchResult] = useState<CompareResponse | null>(null)
  const [toolboxLaunchResult, setToolboxLaunchResult] = useState<CompareResponse | null>(null)

  useEffect(() => {
    if (!window.location.hash) window.history.replaceState(null, '', '#/game')
    const syncRoute = () => setRoute(routeFromHash())
    window.addEventListener('hashchange', syncRoute)
    return () => window.removeEventListener('hashchange', syncRoute)
  }, [])

  const navigate = (next: AppRoute) => {
    window.location.hash = `#/${next}`
    setRoute(next)
  }

  const recordGameResult = useCallback((result: CompareResponse) => {
    setLatestResult(result)
    setGameLaunchResult(null)
  }, [])

  return route === 'toolbox'
    ? (
        <AnalystToolbox
          initialResult={toolboxLaunchResult}
          onOpenGame={(result) => {
            if (!result) return
            setGameLaunchResult(result)
            setLatestResult(result)
            setToolboxLaunchResult(null)
            navigate('game')
          }}
        />
      )
    : (
        <CityGame
          initialResult={gameLaunchResult}
          onResult={recordGameResult}
          onOpenToolbox={() => {
            setToolboxLaunchResult(latestResult ?? gameLaunchResult)
            navigate('toolbox')
          }}
        />
      )
}

export default App
