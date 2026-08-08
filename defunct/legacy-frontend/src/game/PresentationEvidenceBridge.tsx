import { useEffect, useMemo, useRef, useSyncExternalStore } from 'react'
import type { CompareResponse } from '../types'
import { sampleRunPresentation } from './presentation'
import type { IncidentPhase } from './realism'
import type { PresentationClockSnapshot, PresentationClockStore } from './presentationClock'

export type PresentationEvidenceFrame = Readonly<{
  result_id: string
  timestamp_ms: number
  day_index: number
  engine_day: number
  raw_progress: number
  eased_progress: number
  paused: boolean
  speed: number
  incident_phase: IncidentPhase
  incident_segment: 'clear' | 'impact' | 'assessment' | 'recovery'
  services: {
    before: number[]
    after_shock: number[]
    end: number[]
    presented: number[]
  }
  available_budget: {
    start: number
    end: number
    presented: number
  }
  wellbeing: { presented: number }
  depots: null | {
    stock_before: number[]
    stock_end: number[]
    presented: number[]
  }
  lighting: { sun_progress: number; saturation: number }
  shock: null | {
    type: string
    day: number
    telegraph_blend: number
    impact_blend: number
    settling_blend: number
    recovery_blend: number
  }
}>

export type RelayPresentationEvidenceHook = {
  /** Playwright must opt in before application boot; absent in normal use. */
  enabled: true
  onFrame?: (frame: PresentationEvidenceFrame) => void
  read?: () => PresentationEvidenceFrame | null
  seek?: (dayIndex: number, progress: number) => void
  clearSeek?: () => void
}

declare global {
  interface Window {
    __RELAY_PRESENTATION_EVIDENCE__?: RelayPresentationEvidenceHook
  }
}

function useClockSnapshot(clock: PresentationClockStore): PresentationClockSnapshot {
  return useSyncExternalStore(clock.subscribe, clock.getSnapshot, clock.getSnapshot)
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value))
}

export function PresentationEvidenceBridge({
  result,
  clock,
  incidentPhase,
  pendingShock,
  onSeek,
  onClearSeek,
}: {
  result: CompareResponse
  clock: PresentationClockStore
  incidentPhase: IncidentPhase
  pendingShock: { type: string; day: number } | null
  onSeek: (dayIndex: number, progress: number) => void
  onClearSeek: () => void
}) {
  const cursor = useClockSnapshot(clock)
  const sample = useMemo(() => sampleRunPresentation(result, cursor), [cursor, result])
  const frame = useMemo<PresentationEvidenceFrame>(() => {
    const conditionTone = clamp((0.72 - sample.wellbeing) / 0.62)
    const currentShock = sample.recordedDay.shock.type
    const shock = pendingShock ? {
      type: pendingShock.type,
      day: pendingShock.day,
      telegraph_blend: sample.easedProgress,
      impact_blend: 0,
      settling_blend: 0,
      recovery_blend: 0,
    } : currentShock ? {
      type: currentShock,
      day: sample.recordedDay.day,
      telegraph_blend: 0,
      impact_blend: sample.shockImpactProgress,
      settling_blend: sample.incidentSegment === 'recovery'
        ? clamp(sample.recoveryProgress / 0.22)
        : 0,
      recovery_blend: sample.recoveryProgress,
    } : null
    return {
      result_id: result.result_id,
      timestamp_ms: typeof performance === 'undefined' ? 0 : performance.now(),
      day_index: sample.dayIndex,
      engine_day: sample.dayNumber,
      raw_progress: sample.progress,
      eased_progress: sample.easedProgress,
      paused: cursor.paused,
      speed: cursor.speed,
      incident_phase: incidentPhase,
      incident_segment: sample.incidentSegment,
      services: {
        before: [...sample.recordedDay.services_before],
        after_shock: [...sample.servicesAfterShock],
        end: [...sample.recordedDay.services_end],
        presented: [...sample.services],
      },
      available_budget: {
        start: sample.availableBudgetEndpoints.start,
        end: sample.availableBudgetEndpoints.end,
        presented: sample.availableBudget,
      },
      wellbeing: { presented: sample.wellbeing },
      depots: sample.logistics ? {
        stock_before: [...sample.logistics.depotStockEndpoints.start],
        stock_end: [...sample.logistics.depotStockEndpoints.end],
        presented: [...sample.logistics.depotStock],
      } : null,
      lighting: {
        sun_progress: 0.5 + Math.sin((sample.dayIndex + sample.progress) * Math.PI * 0.42 - 0.7) * 0.26,
        saturation: 1 - conditionTone * 0.18,
      },
      shock,
    }
  }, [cursor.paused, cursor.speed, incidentPhase, pendingShock, result.result_id, sample])
  const frameRef = useRef<PresentationEvidenceFrame | null>(null)
  frameRef.current = frame

  useEffect(() => {
    const hook = typeof window === 'undefined' ? undefined : window.__RELAY_PRESENTATION_EVIDENCE__
    if (!hook?.enabled) return
    hook.read = () => frameRef.current
    hook.seek = onSeek
    hook.clearSeek = onClearSeek
    try {
      hook.onFrame?.(frame)
    } catch {
      // Verification must never influence playback or the returned trajectory.
    }
  }, [frame, onClearSeek, onSeek])

  return null
}
