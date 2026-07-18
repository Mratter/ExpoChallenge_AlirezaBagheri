import { Activity, CloudRain, Package, Waves, Zap } from 'lucide-react'
import type { DragEvent, ReactNode } from 'react'
import { shockTypes, type ShockType } from '../types'
import { DISTRICTS, shockImpactFor, type DistrictDefinition } from './model'
import type { GameMode } from './session'

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
  remaining,
  mode,
  aimedDistrict,
  targetLabel,
  onSeverity,
  onArm,
  onAimStart,
  onAimEnd,
  onDistrictSelect,
  onConfirm,
  onCancel,
}: {
  severity: number
  aimingType: ShockType | null
  disabled: boolean
  remaining: number | null
  mode: GameMode
  aimedDistrict: DistrictDefinition | null
  targetLabel: string | null
  onSeverity: (severity: number) => void
  onArm: (type: ShockType) => void
  onAimStart: (type: ShockType, event: DragEvent<HTMLButtonElement>) => void
  onAimEnd: () => void
  onDistrictSelect: (district: DistrictDefinition) => void
  onConfirm: () => void
  onCancel: () => void
}) {
  const hardestType = aimedDistrict
    ? shockTypes.reduce((hardest, type) => (
        shockImpactFor(type, aimedDistrict.service) > shockImpactFor(hardest, aimedDistrict.service)
          ? type
          : hardest
      ))
    : null

  return (
    <aside className={`disaster-tray ${aimingType ? 'is-aiming' : ''}`} aria-label="Disaster tray">
      <div className="tray-heading">
        <div><span>Disaster tray</span><b>{aimingType ? 'Choose a district or drag' : 'Drag or tap to strike overnight'}</b></div>
        <div className="tray-readouts">
          <span aria-label={mode === 'stress' ? `${remaining ?? 0} disasters remaining` : 'Unlimited disasters'}>
            {mode === 'stress' ? `${remaining ?? 0} left` : '∞'}
          </span>
          <output aria-label="Selected disaster severity">{severity.toFixed(2)}</output>
        </div>
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
            className={`${aimingType === type ? 'active' : ''} ${hardestType === type ? 'is-hardest' : ''}`}
            aria-pressed={aimingType === type}
            onClick={() => onArm(type)}
            onDragStart={(event) => onAimStart(type, event)}
            onDragEnd={onAimEnd}
          >
            <ShockIcon type={type} />
            <span>{labels[type]}</span>
          </button>
        ))}
      </div>
      {aimingType ? (
        <fieldset className="district-targets">
          <legend>Choose a district target</legend>
          <div>
            {DISTRICTS.map((district) => (
              <button
                key={district.service}
                type="button"
                disabled={disabled}
                className={aimedDistrict?.service === district.service ? 'active' : ''}
                aria-pressed={aimedDistrict?.service === district.service}
                onClick={() => onDistrictSelect(district)}
              >
                {district.shortLabel}
              </button>
            ))}
          </div>
          {aimedDistrict && hardestType ? (
            <p className="footprint-readout" aria-live="polite">
              <span>{aimedDistrict.shortLabel} is hit hardest by {labels[hardestType]} · {Math.round(shockImpactFor(hardestType, aimedDistrict.service) * 100)}% footprint.</span>
              <small>Selected {labels[aimingType]} footprint · {Math.round(shockImpactFor(aimingType, aimedDistrict.service) * 100)}%.</small>
            </p>
          ) : (
            <p className="footprint-readout">Select one of the five engine districts.</p>
          )}
          <div className="district-target-actions">
            <button type="button" className="cancel-target" onClick={onCancel}>Cancel</button>
            <button
              type="button"
              className="confirm-target"
              disabled={disabled || !aimedDistrict}
              onClick={onConfirm}
            >
              {aimedDistrict ? `Strike ${aimedDistrict.shortLabel} overnight` : 'Choose a district'}
            </button>
          </div>
        </fieldset>
      ) : null}
      <p className="aim-readout" aria-live="polite">
        {targetLabel ?? (
          remaining === 0
            ? 'Stress Test arsenal exhausted — playback continues.'
            : disabled
              ? 'Disasters are unavailable at this day boundary.'
              : 'The typed footprint follows engine truth.'
        )}
      </p>
    </aside>
  )
}
