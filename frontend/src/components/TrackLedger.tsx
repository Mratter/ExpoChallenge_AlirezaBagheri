import { Check, ChevronRight } from 'lucide-react'
import { formatInteger, humanizeToken, isUntrained } from '../format'
import type { ModelTrack } from '../types'

export function TrackLedger({
  tracks,
  selectedId,
  onSelect,
}: {
  tracks: ModelTrack[]
  selectedId: string
  onSelect: (trackId: string) => void
}) {
  return (
    <section className="track-ledger" aria-labelledby="track-ledger-title">
      <div className="section-heading compact-heading">
        <div>
          <span className="index-label">MODEL LEDGER</span>
          <h2 id="track-ledger-title">Research tracks, separated by evidence class</h2>
        </div>
        <p>Select a row to inspect exactly what existed, trained, and passed.</p>
      </div>
      <div className="track-list" role="group" aria-label="Model research tracks">
        {tracks.map((track, index) => {
          const selected = track.id === selectedId
          const untrained = isUntrained(track)
          return (
            <button
              type="button"
              className={`track-row ${selected ? 'selected' : ''}`}
              key={track.id}
              onClick={() => onSelect(track.id)}
              aria-pressed={selected}
            >
              <span className="track-order">{String(index + 1).padStart(2, '0')}</span>
              <span className="track-name"><b>{track.name}</b><small>{track.role}</small></span>
              <span className={`track-state ${untrained ? 'untrained' : 'trained'}`}>
                {untrained ? 'NOT TRAINED' : <><Check size={12} /> TRAINED</>}
              </span>
              <span className="track-budget">
                <b>{formatInteger(track.training.transitions)}</b>
                <small>{humanizeToken(track.training.unit)}</small>
              </span>
              <ChevronRight className="track-chevron" size={17} aria-hidden="true" />
            </button>
          )
        })}
      </div>
    </section>
  )
}
