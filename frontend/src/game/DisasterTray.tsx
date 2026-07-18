import { Activity, CloudRain, Package, Waves, Zap } from 'lucide-react'
import type { DragEvent, ReactNode } from 'react'
import { shockTypes, type ShockType } from '../types'

const labels: Record<ShockType, string> = {
  aftershock: 'Aftershock',
  supply: 'Supply',
  epidemic: 'Epidemic',
  utility: 'Utility',
  weather: 'Weather',
}

function ShockIcon({ type }: { type: ShockType }): ReactNode {
  if (type === 'aftershock') return <Waves size={17} />
  if (type === 'supply') return <Package size={17} />
  if (type === 'epidemic') return <Activity size={17} />
  if (type === 'utility') return <Zap size={17} />
  return <CloudRain size={17} />
}

export function DisasterTray({
  severity,
  aimingType,
  disabled,
  targetLabel,
  onSeverity,
  onAimStart,
  onAimEnd,
}: {
  severity: number
  aimingType: ShockType | null
  disabled: boolean
  targetLabel: string | null
  onSeverity: (severity: number) => void
  onAimStart: (type: ShockType, event: DragEvent<HTMLButtonElement>) => void
  onAimEnd: () => void
}) {
  return (
    <aside className={`disaster-tray ${aimingType ? 'is-aiming' : ''}`} aria-label="Disaster tray">
      <div className="tray-heading">
        <div><span>Disaster tray</span><b>{aimingType ? 'Aim over the city' : 'Drag to strike overnight'}</b></div>
        <output aria-label="Selected disaster severity">{severity.toFixed(2)}</output>
      </div>
      <label className="severity-control">
        <span>Severity</span>
        <input
          type="range"
          min="0.05"
          max="0.40"
          step="0.01"
          value={severity}
          disabled={disabled}
          onChange={(event) => onSeverity(Number(event.target.value))}
        />
        <small>0.05</small><small>0.40</small>
      </label>
      <div className="disaster-cards">
        {shockTypes.map((type) => (
          <button
            key={type}
            type="button"
            draggable={!disabled}
            disabled={disabled}
            className={aimingType === type ? 'active' : ''}
            onDragStart={(event) => onAimStart(type, event)}
            onDragEnd={onAimEnd}
          >
            <ShockIcon type={type} />
            <span>{labels[type]}</span>
          </button>
        ))}
      </div>
      <p className="aim-readout" aria-live="polite">
        {targetLabel ?? (disabled ? 'Run complete — rewind to throw another disaster.' : 'The typed footprint follows engine truth.')}
      </p>
    </aside>
  )
}
