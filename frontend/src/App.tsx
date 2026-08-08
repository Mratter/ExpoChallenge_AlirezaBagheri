import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, ArrowDownRight, RefreshCw } from 'lucide-react'
import { loadWorkbenchOverview } from './api'
import { BenchmarkMatrix } from './components/BenchmarkMatrix'
import { DecisionPipeline } from './components/DecisionPipeline'
import { EvidenceTimeline } from './components/EvidenceTimeline'
import { ModelAnatomy } from './components/ModelAnatomy'
import { ModelToolbox } from './components/ModelToolbox'
import { ProvenancePanel } from './components/ProvenancePanel'
import { ShowcaseBenchmark } from './components/ShowcaseBenchmark'
import { TrackLedger } from './components/TrackLedger'
import { WorkbenchHeader } from './components/WorkbenchHeader'
import { humanizeToken, isUntrained, metricById } from './format'
import type { ModelTrack, WorkbenchOverview } from './types'

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; overview: WorkbenchOverview }

function metricWithFallback(track: ModelTrack, id: string, label: string) {
  return metricById(track, id) ?? track.evaluation.metrics.find((metric) => metric.label.toLowerCase().includes(label))
}

function LoadingState() {
  return (
    <main className="state-page" aria-busy="true" aria-label="Loading model evidence">
      <div className="loading-rule" aria-hidden="true"><i /><i /><i /><i /></div>
      <span className="index-label">MODEL WORKBENCH</span>
      <h1>Binding claims to evidence…</h1>
      <p>Loading the model registry, training receipts, benchmark summary, and artifact hashes.</p>
    </main>
  )
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <main className="state-page error-page" role="alert">
      <AlertTriangle size={28} aria-hidden="true" />
      <span className="index-label">FAIL-CLOSED EVIDENCE GATE</span>
      <h1>Evidence unavailable</h1>
      <p>{message} No model claims or headline metrics are shown without the authoritative API.</p>
      <button type="button" onClick={onRetry}><RefreshCw size={16} />Retry evidence connection</button>
    </main>
  )
}

function Workbench({ overview }: { overview: WorkbenchOverview }) {
  const production = overview.tracks.find((track) => track.id === 'production-v2') ?? overview.tracks[0]
  const r22 = overview.tracks.find((track) => track.id === 'architecture-r22-v10') ?? overview.tracks.at(-1) ?? production
  const measuredBenchmark = overview.benchmark.status === 'measured' ? overview.benchmark : null
  const showcase = measuredBenchmark
    ? overview.tracks.find((track) => track.id === measuredBenchmark.model_track_id)
    : undefined
  const presentationBenchmark = measuredBenchmark ? {
    ...measuredBenchmark,
    limitations: measuredBenchmark.limitations.filter((item) => !item.toLowerCase().includes('version 1')),
    provenance: measuredBenchmark.provenance.filter((item) => !item.path.includes('adaptive-cascades-showcase-v1')),
  } : null
  const presentationShowcase = showcase ? {
    ...showcase,
    limitations: showcase.limitations.filter((item) => !item.toLowerCase().includes('version 1')),
    provenance: showcase.provenance.filter((item) => !item.path.includes('adaptive-cascades-showcase-v1')),
  } : undefined
  const activeTrack = showcase ?? production
  const [selectedTrackId, setSelectedTrackId] = useState(activeTrack.id)
  const [selectedStageId, setSelectedStageId] = useState(overview.pipeline[0]?.id ?? '')
  const selectedTrack = overview.tracks.find((track) => track.id === selectedTrackId) ?? activeTrack
  const wins = metricWithFallback(production, 'scenario_wins', 'scenario wins')
  const lift = metricWithFallback(production, 'relative_improvement_percent', 'relative')
  const diagnostic = metricWithFallback(r22, 'diagnostic_reduction_percent', 'diagnostic reduction')
  const showcaseParameters = showcase?.architecture.parameters?.toLocaleString('en-US') ?? 'Verified'
  const showcaseSafety = showcase?.safety.replay_verified && showcase.safety.hard_violations === 0
    ? 'exact replay / zero hard violations'
    : showcase ? humanizeToken(showcase.status) : ''

  return (
    <div id="top" className="workbench-shell">
      <WorkbenchHeader projectName={overview.project.name} />
      <aside className="index-rail" aria-label="Workbench sections">
        <span>INDEX</span>
        <a href="#top">Brief</a>
        {measuredBenchmark && showcase ? (
          <>
            <a href="#toolbox">Toolbox</a>
            <a href="#anatomy">Architecture</a>
            <a href="#benchmark">Benchmark</a>
            <a href="#evidence">Evidence</a>
          </>
        ) : (
          <>
            <a href="#pipeline">Pipeline</a>
            <a href="#anatomy">Anatomy</a>
            <a href="#history">History</a>
            <a href="#benchmark">Benchmark</a>
            <a href="#evidence">Evidence</a>
          </>
        )}
      </aside>
      <main className="workbench-main">
        <section className="hero" aria-labelledby="hero-title">
          {measuredBenchmark && showcase ? (
            <>
              <div className="hero-copy">
                <span className="index-label">ACTIVE SHOWCASE / ADAPTIVE CASCADE MLP V2</span>
                <h1 id="hero-title">The model learns the warning.<br /><em>The heuristic reacts too late.</em></h1>
                <p>
                  A {showcaseParameters}-parameter policy combines 21 public forecast, regime, need, health,
                  and phase signals to choose one of five interventions on sealed synthetic cascade tapes.
                </p>
                <div className="hero-claim">
                  <div>
                    <span>MODEL OBJECTIVE PASSES</span>
                    <strong>{measuredBenchmark.objective.learned_policy.passes} / {measuredBenchmark.scenario_total}</strong>
                  </div>
                  <i aria-hidden="true" />
                  <div>
                    <span>HEURISTIC OBJECTIVE PASSES</span>
                    <strong>{measuredBenchmark.objective.static_heuristic.passes} / {measuredBenchmark.scenario_total}</strong>
                  </div>
                  <i aria-hidden="true" />
                  <div>
                    <span>MATCHED WIN–LOSS–TIE</span>
                    <strong>
                      {measuredBenchmark.head_to_head.learned_wins}–{measuredBenchmark.head_to_head.heuristic_wins}–{measuredBenchmark.head_to_head.ties}
                    </strong>
                    <small>same {measuredBenchmark.scenario_total} sealed tapes</small>
                  </div>
                </div>
                <p className="metric-note">
                  {measuredBenchmark.synthetic_disclosure} The two objective counts are independent; the
                  win–loss–tie scoreline is the complementary direct comparison.
                </p>
                <a className="hero-toolbox-cta" href="#toolbox">
                  <span>
                    <b>OPEN MODEL TOOLBOX</b>
                    <small>Run the real ONNX policy and compare its decision with the heuristic</small>
                  </span>
                  <ArrowDownRight size={28} aria-hidden="true" />
                </a>
              </div>
              <aside className="truth-panel model-receipt" aria-label="Active 300k model receipt">
                <span className="truth-label">MODEL RECEIPT / VERIFIED</span>
                <small>{showcase.name}</small>
                <strong>{showcaseParameters}</strong>
                <b>trainable parameters</b>
                <p>{showcase.training.note}</p>
                <div><span>STATUS</span><b>{showcaseSafety}</b></div>
              </aside>
            </>
          ) : (
            <>
              <div className="hero-copy">
                <span className="index-label">ACTIVE MODEL / PRODUCTION V2</span>
                <h1 id="hero-title">The trained model is the policy.<br /><em>The solver is its guardrail.</em></h1>
                <p>{overview.project.summary}</p>
                <div className="hero-claim">
                  <div><span>HELD-OUT AUC WINS</span><strong>{wins?.display ?? 'Measured'}</strong></div>
                  <i aria-hidden="true" />
                  <div><span>RELATIVE AUC LIFT</span><strong>{lift?.display ?? 'Measured'}</strong></div>
                  <i aria-hidden="true" />
                  <div><span>TRAINING BUDGET</span><strong>{production.training.transitions.toLocaleString('en-US')}</strong><small>{humanizeToken(production.training.unit)}</small></div>
                </div>
                <p className="metric-note">{overview.project.metric_note}</p>
              </div>
              <aside className="truth-panel" aria-label="R22 training status">
                <span className="truth-label">ADVANCED TRACK / RESEARCH STATUS</span>
                <small>{r22.name}</small>
                <strong>{isUntrained(r22) ? '0' : r22.training.transitions.toLocaleString('en-US')}</strong>
                <b>{humanizeToken(r22.training.unit)}</b>
                <p>{diagnostic?.display ?? '7.34%'} was a privileged planning diagnostic—not model accuracy and not trained-model performance.</p>
                <div><span>STATUS</span><b>{humanizeToken(r22.status)}</b></div>
              </aside>
            </>
          )}
        </section>

        {presentationBenchmark && presentationShowcase ? (
          <>
            <ModelToolbox benchmark={presentationBenchmark} />
            <ModelAnatomy track={presentationShowcase} />
            <section id="benchmark" className="benchmark-section single-model-benchmark" aria-labelledby="benchmark-title">
              <div className="section-heading">
                <div>
                  <span className="index-label">SEALED MODEL VS HEURISTIC</span>
                  <h2 id="benchmark-title">The comparison behind the headline</h2>
                </div>
                <p>{presentationShowcase.evaluation.headline}</p>
              </div>
              <ShowcaseBenchmark benchmark={presentationBenchmark} />
            </section>
            <ProvenancePanel track={presentationShowcase} />
          </>
        ) : (
          <>
            <TrackLedger tracks={overview.tracks} selectedId={selectedTrack.id} onSelect={setSelectedTrackId} />
            <DecisionPipeline stages={overview.pipeline} selectedId={selectedStageId} onSelect={setSelectedStageId} />
            <ModelAnatomy track={selectedTrack} />
            <EvidenceTimeline tracks={overview.tracks} />
            <BenchmarkMatrix overview={overview} production={production} />
            <ProvenancePanel track={selectedTrack} />
          </>
        )}

        <footer className="workbench-footer">
          <span>{overview.project.name} / model-workbench-v1</span>
          <p>Measured synthetic evidence only. No operational disaster-response claim.</p>
        </footer>
      </main>
    </div>
  )
}

function App() {
  const [state, setState] = useState<LoadState>({ status: 'loading' })
  const [requestVersion, setRequestVersion] = useState(0)

  const retry = useCallback(() => {
    setState({ status: 'loading' })
    setRequestVersion((version) => version + 1)
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void loadWorkbenchOverview(controller.signal).then(
      (overview) => setState({ status: 'ready', overview }),
      (error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        setState({ status: 'error', message: error instanceof Error ? error.message : 'The evidence service could not be reached.' })
      },
    )
    return () => controller.abort()
  }, [requestVersion])

  if (state.status === 'loading') return <LoadingState />
  if (state.status === 'error') return <ErrorState message={state.message} onRetry={retry} />
  return <Workbench overview={state.overview} />
}

export default App
