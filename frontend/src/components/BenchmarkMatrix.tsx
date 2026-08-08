import { Check, Scale } from 'lucide-react'
import { humanizeToken, metricById } from '../format'
import type { ModelTrack, WorkbenchOverview } from '../types'
import { ShowcaseBenchmark } from './ShowcaseBenchmark'

function findMetric(track: ModelTrack, ids: string[], labelFragment: string) {
  for (const id of ids) {
    const match = metricById(track, id)
    if (match) return match
  }
  return track.evaluation.metrics.find((metric) => metric.label.toLowerCase().includes(labelFragment))
}

function numericValue(value: number | string | boolean | null | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

export function BenchmarkMatrix({ overview, production }: { overview: WorkbenchOverview; production: ModelTrack }) {
  const winsMetric = findMetric(production, ['scenario_wins', 'scenario_win_count'], 'scenario wins')
  const totalMetric = findMetric(production, ['scenario_total', 'scenario_count'], 'scenario total')
  const baselineWinsMetric = findMetric(production, ['baseline_wins', 'baseline_win_count'], 'baseline wins')
  const candidateAuc = findMetric(production, ['candidate_rauc', 'candidate_mean_rauc'], 'candidate mean')
  const baselineAuc = findMetric(production, ['baseline_rauc', 'baseline_mean_rauc'], 'baseline mean')
  const lift = findMetric(production, ['relative_improvement_percent', 'relative_rauc_improvement_percent'], 'relative')
  const repeats = findMetric(production, ['deterministic_executions', 'determinism_executions'], 'deterministic')
  const wins = numericValue(winsMetric?.value, 0)
  const baselineWins = numericValue(baselineWinsMetric?.value, 0)
  const total = numericValue(totalMetric?.value, wins + baselineWins)
  const cells = Array.from({ length: total }, (_, index) => index < wins)

  return (
    <section id="benchmark" className="benchmark-section" aria-labelledby="benchmark-title">
      <div className="section-heading">
        <div>
          <span className="index-label">MEASURED V2 HOLDOUT</span>
          <h2 id="benchmark-title">Measured against a fixed comparator</h2>
        </div>
        <p>{production.evaluation.headline}</p>
      </div>
      <div className="benchmark-layout">
        <div className="win-board">
          <div className="win-board-heading">
            <div><span>RESILIENCE AUC WINS</span><strong>{winsMetric?.display ?? `${wins} / ${total}`}</strong></div>
            <div className="matrix-key" aria-hidden="true"><span><i className="candidate-cell" />PPO</span><span><i className="baseline-cell" />GLOP</span></div>
          </div>
          <div
            className="win-matrix"
            role="img"
            aria-label={`${wins} candidate wins and ${baselineWins} baseline wins across ${total} held-out scenario units`}
          >
            {cells.map((candidateWon, index) => (
              <i className={candidateWon ? 'candidate-cell' : 'baseline-cell'} key={index} aria-hidden="true" />
            ))}
          </div>
          <p>Aggregate count matrix. Each cell is one matched scenario unit; cell order is not a chronology.</p>
        </div>
        <div className="benchmark-readout">
          <div className="auc-comparison">
            <div><span>PPO / ONNX</span><strong>{candidateAuc?.display ?? '—'}</strong></div>
            <Scale size={21} aria-hidden="true" />
            <div><span>OR-Tools GLOP</span><strong>{baselineAuc?.display ?? '—'}</strong></div>
          </div>
          <div className="lift-readout">
            <span>RELATIVE AUC LIFT</span>
            <strong>{lift?.display ?? '—'}</strong>
            <small>{lift?.note ?? 'Measured under the fixed v2 protocol.'}</small>
          </div>
          <div className="proof-line"><Check size={14} /> {repeats?.display ?? 'Repeated'} deterministic executions · zero recorded candidate violations</div>
        </div>
      </div>
      {overview.benchmark.status === 'measured' ? (
        <ShowcaseBenchmark benchmark={overview.benchmark} />
      ) : (
        <div className="next-benchmark">
          <span>NEW PATTERN-LEARNING BENCHMARK</span>
          <strong>{humanizeToken(overview.benchmark.status)}</strong>
          <p>{overview.benchmark.note}</p>
        </div>
      )}
    </section>
  )
}
