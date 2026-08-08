import { AlertTriangle, FileText, Shield } from 'lucide-react'
import { compactHash } from '../format'
import type { ModelTrack } from '../types'

export function ProvenancePanel({ track }: { track: ModelTrack }) {
  return (
    <section id="evidence" className="provenance-section" aria-labelledby="provenance-title">
      <div className="section-heading">
        <div>
          <span className="index-label">PROVENANCE &amp; LIMITS</span>
          <h2 id="provenance-title">Receipts you can open, boundaries you can defend</h2>
        </div>
        <p>Showing evidence for the selected track: <b>{track.name}</b></p>
      </div>
      <div className="provenance-layout">
        <div className="artifact-list" aria-label="Evidence artifacts">
          {track.provenance.map((item) => (
            <article key={`${item.path}-${item.sha256}`}>
              <FileText size={18} aria-hidden="true" />
              <div><span>{item.label}</span><b>{item.path}</b><small>{item.source_repository}</small></div>
              <code title={item.sha256}>{compactHash(item.sha256)}</code>
            </article>
          ))}
          {track.provenance.length === 0 ? <p className="empty-evidence">No provenance artifact was supplied for this track.</p> : null}
        </div>
        <div className="safety-and-limits">
          <div className="safety-strip">
            <Shield size={18} aria-hidden="true" />
            <div><span>Hard violations</span><b>{track.safety.hard_violations ?? 'Not measured'}</b></div>
            <div><span>Resource violations</span><b>{track.safety.resource_violations ?? 'Not measured'}</b></div>
            <div><span>Exact replay</span><b>{track.safety.replay_verified === null ? 'Not measured' : track.safety.replay_verified ? 'Verified' : 'Failed'}</b></div>
          </div>
          <div className="limitations">
            <h3><AlertTriangle size={16} aria-hidden="true" /> Claim boundaries</h3>
            <ul>{track.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
          </div>
        </div>
      </div>
    </section>
  )
}
