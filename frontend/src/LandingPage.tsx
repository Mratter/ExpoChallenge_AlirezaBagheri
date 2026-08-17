import { ArrowRight, ArrowUpRight, BarChart3, Building2, Database, FileCheck2, ShieldCheck } from 'lucide-react'
import { useId, useState } from 'react'
import { dayFromChartPointer } from './chartInteraction'
import { mariaRetrospective } from './generated/mariaRetrospective'
import { benchmarkDisplayLabel, reactiveHeuristicName } from './plannerNames'
import type { Service } from './types'
import './landing.css'

type EvidenceSeries = (typeof mariaRetrospective.series)[keyof typeof mariaRetrospective.series]
type SeriesKey = keyof typeof mariaRetrospective.series

const retrospectiveSeriesOrder = ['historical', 'v4'] as const satisfies readonly SeriesKey[]
const runtimeComparisonIds = ['v4', 'reactive'] as const
const milestoneDays = mariaRetrospective.display.milestoneDays
const normalizedAxisTicks = [0, 0.25, 0.5, 0.75, 1] as const
const landingCaption = 'The historical line is a project-derived index from official records. The shipped v4 line is a simulated alternative in the frozen model, not an observed or causal real-world outcome.'

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

function formatDate(value: string): string {
  const [year, month, day] = value.split('-').map(Number)
  return new Intl.DateTimeFormat('en-US', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(Date.UTC(year, month - 1, day)))
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

type RecoveryChartProps = {
  compact?: boolean
  service?: Service
}

/** Each chart keeps its own selected day so inspecting one never moves the others. */
function RecoveryChart({ service, compact = false }: RecoveryChartProps) {
  const titleId = useId()
  const descriptionId = useId()
  const readoutId = useId()
  const [activeDay, setActiveDay] = useState<number>(mariaRetrospective.display.dayEnd)
  const [inspecting, setInspecting] = useState(false)
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
  const selectedDate = mariaRetrospective.dates[activeDay] ?? mariaRetrospective.dates[0]
  const selectedX = ((activeDay - horizonStart) / horizonSpan) * plotWidth
  const selectedValuesDescription = retrospectiveSeriesOrder.map((key) => {
    const series = mariaRetrospective.series[key]
    return `${series.label} ${formatIndex(seriesForService(series, service)[activeDay])}`
  }).join(', ')
  const chartDescription = retrospectiveSeriesOrder.map((key) => {
    const values = seriesForService(mariaRetrospective.series[key], service)
    return `${mariaRetrospective.series[key].label} ends at ${formatIndex(values.at(-1) ?? 0)} index points`
  }).join('; ')

  function inspectAt(clientX: number, svg: SVGSVGElement) {
    const bounds = svg.getBoundingClientRect()
    setActiveDay(dayFromChartPointer({
      boundsLeft: bounds.left,
      boundsWidth: bounds.width,
      clientX,
      dayEnd,
      dayStart: horizonStart,
      plotLeft: margin.left,
      plotWidth,
      viewWidth: width,
    }))
  }

  return (
    <div className={compact ? 'chart-explorer chart-explorer-compact' : 'chart-explorer'} data-active-day={activeDay} data-interactive-chart={service ?? 'overall'}>
      <svg
        className={compact ? 'recovery-chart recovery-chart-compact' : 'recovery-chart'}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-labelledby={`${titleId} ${descriptionId}`}
        onPointerDown={(event) => inspectAt(event.clientX, event.currentTarget)}
        onPointerMove={(event) => {
          if (event.pointerType !== 'mouse') return
          setInspecting(true)
          inspectAt(event.clientX, event.currentTarget)
        }}
        onPointerLeave={() => setInspecting(false)}
        onPointerCancel={() => setInspecting(false)}
      >
        <title id={titleId}>{`${label}, days ${horizonStart} through ${dayEnd}`}</title>
        <desc id={descriptionId}>{chartDescription}. Historical dots mark dates with source observations. The selected day is shown in the visible readout below.</desc>
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
          <g className="landing-chart-lines">
            {retrospectiveSeriesOrder.map((key) => (
              <path
                className={`landing-chart-line landing-chart-${key}`}
                d={linePath(seriesForService(mariaRetrospective.series[key], service), plotWidth, plotHeight)}
                key={key}
              />
            ))}
          </g>
          {observationDays.map((day, index) => (
            <circle
              className="landing-observation-dot"
              cx={((day - horizonStart) / horizonSpan) * plotWidth}
              cy={plotHeight - historical[day - horizonStart] * plotHeight}
              key={day}
              r={compact ? 2.5 : 3.5}
              style={{ animationDelay: `${170 + Math.min(index, 10) * 18}ms` }}
            >
              <title>{`${mariaRetrospective.dates[day]}: ${service ? 'official service observation used in the reconstruction' : 'date anchored by one or more official service observations; overall value remains project-derived'}`}</title>
            </circle>
          ))}
          <g className="landing-chart-selection" style={{ transform: `translateX(${selectedX}px)` }} aria-hidden="true">
            {/* The guide line tracks the pointer; the dots stay so the readout always has a visible source. */}
            {inspecting ? <line className="landing-chart-crosshair" x1="0" x2="0" y1="0" y2={plotHeight} /> : null}
            {retrospectiveSeriesOrder.map((key) => {
              const values = seriesForService(mariaRetrospective.series[key], service)
              return (
                <circle
                  className={`landing-chart-selected-dot landing-chart-selected-${key}`}
                  cy={plotHeight - values[activeDay] * plotHeight}
                  key={key}
                  r={compact ? 3.8 : 5}
                />
              )
            })}
          </g>
        </g>
      </svg>

      <div className="chart-readout" id={readoutId}>
        <p><span>Selected</span><strong>Day {activeDay}</strong><time dateTime={selectedDate}>{formatDate(selectedDate)}</time></p>
        <dl>
          {retrospectiveSeriesOrder.map((key) => {
            const series = mariaRetrospective.series[key]
            const value = seriesForService(series, service)[activeDay]
            return <div key={key}><dt><i className={`table-key table-key-${key}`} />{series.label}</dt><dd>{formatIndex(value)}</dd></div>
          })}
        </dl>
      </div>

      <div className="chart-scrubber-row">
        <input
          aria-describedby={readoutId}
          aria-label={`Inspect ${label} by day`}
          aria-valuetext={`Day ${activeDay}, ${formatDate(selectedDate)}; ${selectedValuesDescription}`}
          className="chart-scrubber"
          max={dayEnd}
          min={horizonStart}
          onChange={(event) => setActiveDay(Number(event.currentTarget.value))}
          step="1"
          type="range"
          value={activeDay}
        />
        <span aria-hidden="true">Hover, tap, or use ← →</span>
      </div>
    </div>
  )
}

function EvidenceLegend() {
  return (
    <div className="landing-legend" aria-label="Chart legend">
      <span className="legend-historical"><i />Project reconstruction <b>observed markers</b></span>
      <span className="legend-v4"><i />Shipped v4 <b>simulation</b></span>
    </div>
  )
}

function MilestoneTable() {
  return (
    <div className="landing-table-wrap hero-table-wrap">
      <table className="landing-table milestone-table" data-table="maria-milestones">
        <caption>Overall recovery index at {milestoneDays.length} milestones</caption>
        <thead>
          <tr><th scope="col">Evidence</th><th scope="col">Type</th>{milestoneDays.map((day) => <th scope="col" key={day}>Day {day}</th>)}</tr>
        </thead>
        <tbody>
          {retrospectiveSeriesOrder.map((key) => {
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

function BenchmarkTables() {
  const runtimeRows = runtimeComparisonIds.map((id) => {
    const row = mariaRetrospective.benchmarkRows.find((candidate) => candidate.id === id)
    if (!row) throw new Error(`Missing runtime comparison row: ${id}`)
    return row
  })

  return (
    <section className="benchmark-section" aria-labelledby="benchmark-title">
      <div className="landing-section benchmark-inner">
        <header className="section-heading benchmark-heading">
          <div><p className="landing-kicker">Separate simulation performance</p><h2 id="benchmark-title">How the shipped model performs in held-out simulations.</h2></div>
          <p>These are solved-case results from the synthetic {mariaRetrospective.syntheticBenchmarkCaseCount}-case final split. They are not values on the Hurricane Maria recovery index.</p>
        </header>

        <div className="runtime-comparison">
          <header>
            <div><p className="landing-kicker">Selected runtime comparison</p><h3>Shipped v4 versus the fine tuned baseline</h3></div>
            <p>This compact view compares the two planners available in the public runtime.</p>
          </header>
          <div className="landing-table-wrap runtime-table-wrap">
            <table className="landing-table runtime-table" data-table="runtime-comparison">
              <caption>Canonical held-out synthetic results for shipped v4 and the {reactiveHeuristicName.toLowerCase()}</caption>
              <thead><tr><th scope="col">Planner</th><th scope="col">Solved cases</th><th scope="col">Solve rate</th></tr></thead>
              <tbody>
                {runtimeRows.map((row) => (
                  <tr data-runtime-comparison={row.id} key={row.id}>
                    <th scope="row">{benchmarkDisplayLabel(row.label)}</th>
                    <td>{row.solved}/{row.total}</td>
                    <td>{formatRate(row.rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <header className="complete-benchmark-heading">
          <div><p className="landing-kicker">Complete benchmark context</p><h3>All {mariaRetrospective.benchmarkRows.length} recorded planners</h3></div>
          <p>The full table remains visible so the selected runtime comparison is not mistaken for the complete benchmark.</p>
        </header>
        <div className="landing-table-wrap benchmark-table-wrap">
          <table className="landing-table benchmark-table" data-table="complete-benchmark">
            <caption>Canonical final-split results for all {mariaRetrospective.benchmarkRows.length} evaluated planners</caption>
            <thead><tr><th scope="col">Planner</th><th scope="col">Status</th><th scope="col">Solved</th><th scope="col">Rate</th><th scope="col">Reading</th></tr></thead>
            <tbody>
              {mariaRetrospective.benchmarkRows.map((row) => (
                <tr data-benchmark={row.id} key={row.id}>
                  <th scope="row">{benchmarkDisplayLabel(row.label)}</th>
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
        <article><ShieldCheck aria-hidden="true" size={18} /><span>02</span><h3>One frozen reconstruction</h3><p>The displayed shipped-policy simulation uses the documented starting state and no-secondary-shock tape.</p></article>
        <article><BarChart3 aria-hidden="true" size={18} /><span>03</span><h3>Two evidence classes</h3><p>The historical reconstruction and simulated policy alternative remain labelled; neither is presented as a causal estimate.</p></article>
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
        <section className="landing-hero" data-evidence-section="maria-retrospective" id="retrospective-evidence" aria-labelledby="landing-title">
          <div className="hero-intro">
            <p className="landing-kicker">Hurricane Maria · Puerto Rico · {mariaRetrospective.display.dayZeroLabel}–{mariaRetrospective.display.dayEndLabel}</p>
            <h1 id="landing-title">One historical reconstruction.<br /><em>One simulated recovery path.</em></h1>
            <p className="hero-summary">A project reconstruction from official records placed beside the shipped recovery policy—without fitting the simulation to the historical line.</p>
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

          <p className="landing-disclosure">{landingCaption}</p>
        </section>

        <ServiceCharts />
        <BenchmarkTables />
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
