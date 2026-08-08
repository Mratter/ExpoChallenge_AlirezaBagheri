import { humanizeToken } from '../format'
import type { ModelTrack } from '../types'

export function EvidenceTimeline({ tracks }: { tracks: ModelTrack[] }) {
  return (
    <section id="history" className="timeline-section" aria-labelledby="timeline-title">
      <div className="section-heading">
        <div>
          <span className="index-label">TRAINING &amp; EVIDENCE HISTORY</span>
          <h2 id="timeline-title">What changed—and where each track stopped</h2>
        </div>
        <p>Version numbers mark research revisions, not ten successively trained models.</p>
      </div>
      <ol className="evidence-timeline">
        {tracks.map((track, index) => (
          <li key={track.id}>
            <span className="timeline-index">{index + 1}</span>
            <div className="timeline-copy">
              <small>{track.role}</small>
              <h3>{track.name}</h3>
              <p>{track.evaluation.headline}</p>
            </div>
            <span className="timeline-status">{humanizeToken(track.status)}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}
