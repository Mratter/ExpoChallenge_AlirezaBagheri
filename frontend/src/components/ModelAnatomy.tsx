import { Cpu, GitBranch } from 'lucide-react'
import { formatInteger, humanizeToken, isUntrained } from '../format'
import type { ModelTrack } from '../types'

function AnatomyCell({ label, value }: { label: string; value: string }) {
  return <div className="anatomy-cell"><span>{label}</span><strong>{value}</strong></div>
}

export function ModelAnatomy({ track }: { track: ModelTrack }) {
  const untrained = isUntrained(track)
  return (
    <section id="anatomy" className="anatomy-section" aria-labelledby="anatomy-title">
      <div className="anatomy-main">
        <div className="section-heading compact-heading">
          <div>
            <span className="index-label">ARCHITECTURE ANATOMY</span>
            <h2 id="anatomy-title">{track.name}</h2>
          </div>
          <span className={`evidence-pill ${untrained ? 'warning' : 'success'}`}>
            {untrained ? 'architecture only' : 'trained artifact'}
          </span>
        </div>
        <div className="architecture-line" aria-label={`${track.architecture.inputs}, ${track.architecture.family}, ${track.architecture.outputs}`}>
          <div><span>INPUT</span><b>{track.architecture.inputs}</b></div>
          <i aria-hidden="true" />
          <div className="architecture-core"><Cpu size={18} aria-hidden="true" /><span>POLICY</span><b>{track.architecture.family}</b></div>
          <i aria-hidden="true" />
          <div><span>OUTPUT</span><b>{track.architecture.outputs}</b></div>
        </div>
        <div className="anatomy-grid">
          <AnatomyCell label="Parameters" value={track.architecture.parameters === null ? 'Not reported' : formatInteger(track.architecture.parameters)} />
          <AnatomyCell label="Runtime" value={track.architecture.runtime} />
          <AnatomyCell label="Evidence class" value={humanizeToken(track.evidence_class)} />
          <AnatomyCell label="Claim eligible" value={track.claim_eligible ? 'Yes · bounded claim' : 'No'} />
        </div>
      </div>
      <aside className="training-receipt" aria-label={`${track.name} training receipt`}>
        <div className="receipt-heading"><GitBranch size={17} aria-hidden="true" /><span>TRAINING RECEIPT</span></div>
        <dl>
          <div><dt>Started</dt><dd>{track.training.started ? 'Yes' : 'No'}</dd></div>
          <div><dt>Training volume</dt><dd>{formatInteger(track.training.transitions)} {humanizeToken(track.training.unit)}</dd></div>
          <div><dt>Seeds</dt><dd>{track.training.seed_count}</dd></div>
          <div><dt>Hardware</dt><dd>{track.training.hardware}</dd></div>
        </dl>
        <p>{track.training.note}</p>
      </aside>
    </section>
  )
}
