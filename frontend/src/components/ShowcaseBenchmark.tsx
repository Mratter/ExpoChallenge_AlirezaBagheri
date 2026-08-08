import { FileCheck2, FlaskConical, Info, ShieldCheck } from 'lucide-react'
import { compactHash, humanizeToken } from '../format'
import type { MeasuredWorkbenchBenchmark } from '../types'

function signed(value: number): string {
  return `${value > 0 ? '+' : ''}${value.toFixed(3)}`
}

function precise(value: number): string {
  return value.toFixed(13)
}

export function ShowcaseBenchmark({ benchmark }: { benchmark: MeasuredWorkbenchBenchmark }) {
  const { objective, head_to_head: headToHead, secondary, scenario_total: total } = benchmark
  const [ciLow, ciHigh] = headToHead.paired_bootstrap_ci95

  return (
    <section className="showcase-result" aria-labelledby="showcase-title">
      <header className="showcase-heading">
        <div>
          <span className="index-label">MEASURED / SEALED SYNTHETIC SHOWCASE</span>
          <h3 id="showcase-title">{benchmark.name}</h3>
          <p>Measured response to learnable observable early-warning patterns.</p>
        </div>
        <span className="showcase-seal"><ShieldCheck size={14} aria-hidden="true" /> SEALED EVIDENCE</span>
      </header>

      <div className="synthetic-disclosure" role="note" aria-label="Synthetic benchmark disclosure">
        <FlaskConical size={18} aria-hidden="true" />
        <p>{benchmark.synthetic_disclosure}</p>
      </div>

      <div className="objective-ledger">
        <div className="showcase-subheading">
          <div>
            <span>REGISTERED OBJECTIVE</span>
            <h4>{objective.label}</h4>
          </div>
          <small>Independent thresholds—not a complementary scoreline</small>
        </div>
        <div className="objective-counts" role="group" aria-label="Independent objective pass counts">
          <article className="model-objective" aria-label={`Model objective passes: ${objective.learned_policy.passes} of ${total}`}>
            <span>MODEL OBJECTIVE PASSES</span>
            <strong>{objective.learned_policy.passes}<small>/ {total}</small></strong>
            <p>{objective.learned_policy.label}</p>
          </article>
          <article className="heuristic-objective" aria-label={`Static heuristic objective passes: ${objective.static_heuristic.passes} of ${total}`}>
            <span>STATIC HEURISTIC OBJECTIVE PASSES</span>
            <strong>{objective.static_heuristic.passes}<small>/ {total}</small></strong>
            <p>{objective.static_heuristic.label}</p>
          </article>
        </div>
        <p className="independence-note"><Info size={14} aria-hidden="true" /> {benchmark.note}</p>
      </div>

      <div className="showcase-comparisons">
        <section className="head-to-head" aria-labelledby="head-to-head-title">
          <div className="showcase-subheading">
            <div>
              <span>DIRECT MATCHED HEAD-TO-HEAD</span>
              <h4 id="head-to-head-title">Same 40 scenario tapes</h4>
            </div>
            <small>{headToHead.metric.label}</small>
          </div>
          <dl className="head-to-head-counts">
            <div><dt>LEARNED</dt><dd>{headToHead.learned_wins}</dd></div>
            <div><dt>HEURISTIC</dt><dd>{headToHead.heuristic_wins}</dd></div>
            <div><dt>TIES</dt><dd>{headToHead.ties}</dd></div>
          </dl>
          <div
            className="match-bar"
            role="img"
            aria-label={`Direct matched head-to-head: learned model ${headToHead.learned_wins} wins, static heuristic ${headToHead.heuristic_wins} wins, and ${headToHead.ties} ties across ${total} scenarios`}
          >
            <i className="learned-segment" style={{ width: `${(headToHead.learned_wins / total) * 100}%` }} aria-hidden="true" />
            <i className="heuristic-segment" style={{ width: `${(headToHead.heuristic_wins / total) * 100}%` }} aria-hidden="true" />
            <i className="tie-segment" style={{ width: `${(headToHead.ties / total) * 100}%` }} aria-hidden="true" />
          </div>
          <div className="metric-definition">
            <div><span>METRIC DEFINITION</span><p>{objective.definition}</p></div>
            <div><span>DIRECTION</span><p>{humanizeToken(headToHead.metric.direction)}</p></div>
            <div><span>TIE RULE</span><p>{headToHead.metric.tie_rule}</p></div>
          </div>
          <div className="paired-result">
            <div><span>PAIRED MEAN DIFFERENCE</span><strong>{signed(headToHead.paired_mean_difference)}</strong></div>
            <div><span>PAIRED BOOTSTRAP 95% CI</span><strong>[{ciLow.toFixed(3)}, {ciHigh.toFixed(3)}]</strong></div>
            <small>Learned mean {headToHead.learned_mean.toFixed(3)} · heuristic mean {headToHead.heuristic_mean.toFixed(3)}</small>
          </div>
        </section>

        <section className="secondary-result" aria-labelledby="secondary-title">
          <span>SECONDARY CHECK</span>
          <h4 id="secondary-title">{secondary.metric.label}</h4>
          <small>{humanizeToken(secondary.metric.direction)}</small>
          <dl>
            <div><dt>LEARNED MODEL</dt><dd>{precise(secondary.learned_mean)}</dd></div>
            <div><dt>STATIC HEURISTIC</dt><dd>{precise(secondary.heuristic_mean)}</dd></div>
          </dl>
          <p>Lower cumulative deficit is better on this secondary metric.</p>
        </section>
      </div>

      <div className="showcase-evidence">
        <section className="showcase-limitations" aria-labelledby="showcase-limitations-title">
          <h4 id="showcase-limitations-title"><Info size={15} aria-hidden="true" /> Boundaries</h4>
          <ul>{benchmark.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
        </section>
        <section className="showcase-provenance" aria-labelledby="showcase-provenance-title">
          <h4 id="showcase-provenance-title"><FileCheck2 size={15} aria-hidden="true" /> Evidence receipts</h4>
          <ul>
            {benchmark.provenance.map((item) => (
              <li key={`${item.path}-${item.sha256}`}>
                <div><span>{item.label}</span><code>{item.path}</code></div>
                <code title={item.sha256}>{compactHash(item.sha256)}</code>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </section>
  )
}
