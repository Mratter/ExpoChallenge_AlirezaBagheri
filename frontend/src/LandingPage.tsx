import { ArrowRight, ArrowUpRight, BarChart3, Building2, Database, FileCheck2, ShieldCheck } from 'lucide-react'
import { useId } from 'react'
import { mariaRetrospective } from './generated/mariaRetrospective'
import type { Service } from './types'
import './landing.css'

type EvidenceSeries = (typeof mariaRetrospective.series)[keyof typeof mariaRetrospective.series]
type SeriesKey = keyof typeof mariaRetrospective.series

const seriesOrder: readonly SeriesKey[] = ['historical', 'v4', 'reactive']
const milestoneDays = mariaRetrospective.display.milestoneDays
const normalizedAxisTicks = [0, 0.25, 0.5, 0.75, 1] as const

function formatIndex(value: number): string {
  const { indexMin, indexMax } = mariaRetrospective.display
  return (indexMin + value * (indexMax - indexMin)).toFixed(1)
}

function formatRate(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value)
}

function shortHash(value: string): string {
  return `${value.slice(0, 10)}…${value.slice(-8)}`
}

function linePath(values: readonly number[], width: number, height: number): string {
  return values.map((value, index) => {
    const x = (index / Math.max(values.length - 1, 1)) * width
    const y = height - Math.max(0, Math.min(1, value)) * height
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
  }).join(' ')
}

function observationUnion(): number[] {
  return [...new Set(
    mariaRetrospective.serviceOrder.flatMap((service) => mariaRetrospective.observationDays[service]),
  )].sort((a, b) => a - b)
}

function seriesForService(series: EvidenceSeries, service?: Service): readonly number[] {
  return service ? series.services[service] : series.total
}

function RecoveryChart({ service, compact = false }: { service?: Service; compact?: boolean }) {
  const titleId = useId()
  const descriptionId = useId()
  const width = compact ? 430 : 760
  const height = compact ? 154 : 276
  const margin = compact
    ? { top: 15, right: 16, bottom: 28, left: 37 }
    : { top: 22, right: 22, bottom: 38, left: 52 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom
  const { horizonStart, dayEnd, indexMin, indexMax } = mariaRetrospective.display
  const horizonSpan = dayEnd - horizonStart
  const observationDays = service ? mariaRetrospective.observationDays[service] : observationUnion()
  const label = service ? mariaRetrospective.serviceLabels[service] : 'Overall recovery'
  const historical = seriesForService(mariaRetrospective.series.historical, service)
  const chartDescription = seriesOrder.map((key) => {
    const values = seriesForService(mariaRetrospective.series[key], service)
    return `${mariaRetrospective.series[key].label} ends at ${formatIndex(values.at(-1) ?? 0)} index points`
  }).join('; ')

  return (
    <svg
      className={compact ? 'recovery-chart recovery-chart-compact' : 'recovery-chart'}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-labelledby={`${titleId} ${descriptionId}`}
    >
      <title id={titleId}>{label}, days {horizonStart} through {dayEnd}</title>
      <desc id={descriptionId}>{chartDescription}. Historical dots mark dates with source observations.</desc>
      <g transform={`translate(${margin.left} ${margin.top})`}>
        {!compact ? <text className="landing-chart-axis-title" x={-39} y={plotHeight / 2} textAnchor="middle" transform={`rotate(-90 -39 ${plotHeight / 2})`}>Derived recovery index</text> : null}
        {normalizedAxisTicks.map((tick) => {
          const y = plotHeight - tick * plotHeight
          return (
            <g key={tick}>
              <line className="landing-chart-grid" x1="0" x2={plotWidth} y1={y} y2={y} />
              <text className="landing-chart-tick" x="-9" y={y + 3} textAnchor="end">{Math.round(indexMin + tick * (indexMax - indexMin))}</text>
            </g>
          )
        })}
        {milestoneDays.map((day) => {
          const x = ((day - horizonStart) / horizonSpan) * plotWidth
          return <text className="landing-chart-tick" key={day} x={x} y={plotHeight + 20} textAnchor={day === horizonStart ? 'start' : day === dayEnd ? 'end' : 'middle'}>D{day}</text>
        })}
        {seriesOrder.map((key) => (
          <path
            className={`landing-chart-line landing-chart-${key}`}
            d={linePath(seriesForService(mariaRetrospective.series[key], service), plotWidth, plotHeight)}
            key={key}
          />
        ))}
        {observationDays.map((day) => (
          <circle
            className="landing-observation-dot"
            cx={((day - horizonStart) / horizonSpan) * plotWidth}
            cy={plotHeight - historical[day - horizonStart] * plotHeight}
            key={day}
            r={compact ? 2.5 : 3.5}
          >
            <title>{mariaRetrospective.dates[day]}: {service ? 'official service observation used in the reconstruction' : 'date anchored by one or more official service observations; overall value remains project-derived'}</title>
          </circle>
        ))}
      </g>
    </svg>
  )
}

function EvidenceLegend() {
  return (
    <div className="landing-legend" aria-label="Chart legend">
      <span className="legend-historical"><i />Project reconstruction <b>observed markers</b></span>
      <span className="legend-v4"><i />Shipped v4 <b>simulation</b></span>
      <span className="legend-reactive"><i />Reactive heuristic <b>simulation</b></span>
    </div>
  )
}

function MilestoneTable() {
  return (
    <div className="landing-table-wrap hero-table-wrap">
      <table className="landing-table milestone-table">
        <caption>Overall recovery index at {milestoneDays.length} milestones</caption>
        <thead>
          <tr><th scope="col">Evidence</th><th scope="col">Type</th>{milestoneDays.map((day) => <th scope="col" key={day}>Day {day}</th>)}</tr>
        </thead>
        <tbody>
          {seriesOrder.map((key) => {
            const series = mariaRetrospective.series[key]
            return (
              <tr key={key}>
                <th scope="row"><i className={`table-key table-key-${key}`} />{series.label}</th>
                <td><span className="evidence-type">{series.evidenceType}</span></td>
                {milestoneDays.map((day) => <td key={day}>{formatIndex(series.total[day])}</td>)}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ServiceCharts() {
  return (
    <section className="landing-section service-evidence" aria-labelledby="services-title">
      <header className="section-heading">
        <div><p className="landing-kicker">Service-level evidence</p><h2 id="services-title">The aggregate is only the first reading.</h2></div>
        <p>Each panel keeps the same {mariaRetrospective.display.indexMin}-to-{mariaRetrospective.display.indexMax} scale, so improvements and gaps can be compared without a shifting axis.</p>
      </header>
      <div className="service-chart-grid">
        {mariaRetrospective.serviceOrder.map((service, index) => (
          <figure className="service-chart-card" data-service={service} key={service}>
            <figcaption><span>0{index + 1}</span><h3>{mariaRetrospective.serviceLabels[service]}</h3><b>{mariaRetrospective.observationDays[service].length ? `${mariaRetrospective.observationDays[service].length} observed dates` : 'project-estimate anchors'}</b></figcaption>
            <RecoveryChart service={service as Service} compact />
          </figure>
        ))}
      </div>
    </section>
  )
}

function BenchmarkTable() {
  return (
    <section className="benchmark-section" aria-labelledby="benchmark-title">
      <div className="landing-section benchmark-inner">
        <header className="section-heading benchmark-heading">
          <div><p className="landing-kicker">Separate synthetic benchmark</p><h2 id="benchmark-title">Held-out final split · all comparators</h2></div>
          <p>These are solved-case results from the synthetic {mariaRetrospective.syntheticBenchmarkCaseCount}-case benchmark. They are not values on the Hurricane Maria recovery index.</p>
        </header>
        <div className="landing-table-wrap benchmark-table-wrap">
          <table className="landing-table benchmark-table">
            <caption>Canonical final-split results for all {mariaRetrospective.benchmarkRows.length} evaluated planners</caption>
            <thead><tr><th scope="col">Planner</th><th scope="col">Status</th><th scope="col">Solved</th><th scope="col">Rate</th><th scope="col">Reading</th></tr></thead>
            <tbody>
              {mariaRetrospective.benchmarkRows.map((row) => (
                <tr data-benchmark={row.id} key={row.id}>
                  <th scope="row">{row.label}</th>
                  <td><span className="benchmark-classification">{row.classification}</span></td>
                  <td>{row.solved}/{row.total}</td>
                  <td>{formatRate(row.rate)}</td>
                  <td>{row.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}

function ReceiptStrip() {
  const hashes = [
    ['Retrospective receipt', mariaRetrospective.receiptSha256],
    ['Source manifest', mariaRetrospective.sourceManifestSha256],
    ['Reconstruction', mariaRetrospective.reconstructionSha256],
    ['Shipped artifact', mariaRetrospective.artifactSha256],
  ] as const

  return (
    <section className="landing-section receipt-section" aria-labelledby="receipt-title">
      <header>
        <div><FileCheck2 aria-hidden="true" size={19} /><p className="landing-kicker">Evidence receipt</p><h2 id="receipt-title">Every line resolves to a frozen input.</h2></div>
        <p>{mariaRetrospective.methodologyLabel}</p>
      </header>
      <dl>
        {hashes.map(([label, hash]) => <div key={label}><dt>{label}</dt><dd title={hash}>{shortHash(hash)}</dd></div>)}
      </dl>
      <div className="method-cards">
        <article><Database aria-hidden="true" size={18} /><span>01</span><h3>Official records first</h3><p>Dated government observations are converted and frozen before either planner is loaded.</p></article>
        <article><ShieldCheck aria-hidden="true" size={18} /><span>02</span><h3>One shared reconstruction</h3><p>The shipped policy and heuristic start from the same state and use the same no-secondary-shock tape.</p></article>
        <article><BarChart3 aria-hidden="true" size={18} /><span>03</span><h3>Two evidence classes</h3><p>Historical reconstruction and simulated alternatives remain labelled; neither is presented as a causal estimate.</p></article>
      </div>
    </section>
  )
}

export function LandingPage() {
  return (
    <div className="landing-shell">
      <a className="skip-link" href="#retrospective-evidence">Skip to evidence</a>
      <header className="landing-rail">
        <a className="landing-brand" href="#/" aria-label="RELAY evidence home">
          <span className="brand-seal landing-seal" aria-hidden="true"><i /><i /><i /></span>
          <span><b>RELAY</b><small>Municipal recovery lab</small></span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#/toolbox">Analyst Toolbox</a>
          <a href="#/game">3D City</a>
        </nav>
      </header>

      <main>
        <section className="landing-hero" id="retrospective-evidence" aria-labelledby="landing-title">
          <div className="hero-intro">
            <p className="landing-kicker">Hurricane Maria · Puerto Rico · {mariaRetrospective.display.dayZeroLabel}–{mariaRetrospective.display.dayEndLabel}</p>
            <h1 id="landing-title">One historical reconstruction.<br /><em>Two simulated recovery paths.</em></h1>
            <p className="hero-summary">A project reconstruction from official records placed beside the shipped recovery policy and a reactive heuristic—without fitting either simulation to the historical line.</p>
            <div className="hero-actions">
              <a className="landing-primary" href="#/toolbox">Open Analyst Toolbox <ArrowRight aria-hidden="true" size={17} /></a>
              <a className="landing-secondary" href="#/game"><Building2 aria-hidden="true" size={16} />Explore the 3D city <ArrowUpRight aria-hidden="true" size={15} /></a>
            </div>
            <div className="hero-contract" aria-label="Retrospective contract">
              <span><b>{mariaRetrospective.display.dayCount}</b> dated points</span><span><b>{mariaRetrospective.serviceOrder.length}</b> service indices</span><span><b>{mariaRetrospective.scenarioCount}</b> frozen scenario</span>
            </div>
          </div>

          <div className="hero-evidence">
            <figure className="hero-chart-card">
              <figcaption><div><span>Figure 01 / derived index</span><h2>Recovery trajectory</h2></div><b>Day {mariaRetrospective.display.horizonStart}–{mariaRetrospective.display.dayEnd} · {mariaRetrospective.display.indexMin}–{mariaRetrospective.display.indexMax}</b></figcaption>
              <RecoveryChart />
              <EvidenceLegend />
            </figure>
            <MilestoneTable />
          </div>

          <p className="landing-disclosure">{mariaRetrospective.caption}</p>
        </section>

        <ServiceCharts />
        <BenchmarkTable />
        <ReceiptStrip />

        <section className="landing-final-cta" aria-labelledby="cta-title">
          <p className="landing-kicker">Run the frozen system</p>
          <h2 id="cta-title">Inspect the decisions behind the line.</h2>
          <p>The Analyst Toolbox exposes the full {mariaRetrospective.interface.observationCount}-input, {mariaRetrospective.interface.actionCount}-action trace, paired shock tape, intervention ledger, and official outcome checks.</p>
          <a href="#/toolbox">Open Analyst Toolbox <ArrowRight aria-hidden="true" size={17} /></a>
        </section>
      </main>

      <footer className="landing-footer"><b>RELAY</b><span>Evidence is labelled by source class. Synthetic results and historical reconstruction are never blended.</span><a href="#/">Back to top</a></footer>
    </div>
  )
}
