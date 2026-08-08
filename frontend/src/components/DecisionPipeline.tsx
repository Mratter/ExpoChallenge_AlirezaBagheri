import { ArrowRight } from 'lucide-react'
import type { PipelineStage } from '../types'

export function DecisionPipeline({
  stages,
  selectedId,
  onSelect,
}: {
  stages: PipelineStage[]
  selectedId: string
  onSelect: (stageId: string) => void
}) {
  const selected = stages.find((stage) => stage.id === selectedId) ?? stages[0]
  return (
    <section id="pipeline" className="pipeline-section" aria-labelledby="pipeline-title">
      <div className="section-heading">
        <div>
          <span className="index-label">DECISION PIPELINE</span>
          <h2 id="pipeline-title">Click through one policy decision</h2>
        </div>
        <p>The neural model proposes strategy. Deterministic code owns feasibility and evidence.</p>
      </div>
      <div className="pipeline-spine" role="group" aria-label="Ordered model decision pipeline">
        {stages.map((stage, index) => (
          <div className="pipeline-node-wrap" key={stage.id}>
            <button
              type="button"
              className={`pipeline-node ${stage.id === selected?.id ? 'active' : ''}`}
              onClick={() => onSelect(stage.id)}
              aria-pressed={stage.id === selected?.id}
            >
              <span>{String(index + 1).padStart(2, '0')}</span>
              <b>{stage.label}</b>
            </button>
            {index < stages.length - 1 ? <ArrowRight className="pipeline-arrow" size={16} aria-hidden="true" /> : null}
          </div>
        ))}
      </div>
      {selected ? (
        <div className="pipeline-detail" role="region" aria-live="polite" aria-label={`${selected.label} stage details`}>
          <span className="signal-mark" aria-hidden="true"><i /><i /><i /></span>
          <div>
            <small>ACTIVE STAGE / {selected.id.toUpperCase()}</small>
            <h3>{selected.label}</h3>
          </div>
          <p>{selected.detail}</p>
        </div>
      ) : null}
    </section>
  )
}
