import type { CompareResponse, DayResult, Service } from '../types'

export type DamageState = 'intact' | 'slight' | 'moderate' | 'rubble'
export type BuildingArchetype =
  | 'apartment'
  | 'rowhouse'
  | 'office'
  | 'hospital'
  | 'market'
  | 'warehouse'
  | 'transit'
  | 'civic'

export type DistrictDefinition = {
  service: Service
  label: string
  shortLabel: string
  accent: string
  body: string
  center: [number, number, number]
}

export const SERVICE_LABELS: Record<Service, string> = {
  transport: 'Transport',
  housing: 'Housing',
  food: 'Food',
  healthcare: 'Healthcare',
  public_services: 'Civic',
}

export const DISTRICTS: DistrictDefinition[] = [
  { service: 'housing', label: 'Residential quarter', shortLabel: 'Housing', accent: '#bd6b52', body: '#b98269', center: [-6.7, 0, -5.1] },
  { service: 'healthcare', label: 'Health campus', shortLabel: 'Healthcare', accent: '#e6e2d8', body: '#aab9b4', center: [6.6, 0, -5.1] },
  { service: 'food', label: 'Market quarter', shortLabel: 'Food', accent: '#d49a3d', body: '#c88a4c', center: [-7.1, 0, 5.0] },
  { service: 'transport', label: 'Transit works', shortLabel: 'Transport', accent: '#5a8290', body: '#6e8790', center: [7.0, 0, 5.0] },
  { service: 'public_services', label: 'Civic quarter', shortLabel: 'Civic', accent: '#71866a', body: '#8b9a7f', center: [0, 0, 7.0] },
]

const damageBias = [-0.78, -0.45, -0.12, 0.18, 0.43, 0.72, 0.98]

export function damageStateFor(serviceLevel: number, buildingIndex: number): DamageState {
  const score = Math.max(0, Math.min(3, (1 - serviceLevel) * 3.8 + damageBias[buildingIndex % damageBias.length]))
  if (score < 0.75) return 'intact'
  if (score < 1.5) return 'slight'
  if (score < 2.35) return 'moderate'
  return 'rubble'
}

export function serviceIndex(service: Service): number {
  return ['transport', 'housing', 'food', 'healthcare', 'public_services'].indexOf(service)
}

export function isBuildingRebuilding(
  day: DayResult,
  previous: DayResult | undefined,
  service: Service,
  buildingIndex: number,
): boolean {
  const index = serviceIndex(service)
  const recovered = previous ? day.services_end[index] - previous.services_end[index] : day.gain[index]
  if (recovered <= 0.002 || damageStateFor(day.services_end[index], buildingIndex) === 'intact') return false
  const activeCount = Math.max(1, Math.min(3, Math.round(day.allocation[index] / 28)))
  return (buildingIndex + day.day + index * 2) % 7 < activeCount
}

function strongestImpact(day: DayResult): Service {
  let strongestIndex = 0
  day.shock.impact.forEach((value, index) => {
    if (value > day.shock.impact[strongestIndex]) strongestIndex = index
  })
  return ['transport', 'housing', 'food', 'healthcare', 'public_services'][strongestIndex] as Service
}

function largestValue(values: number[]): number {
  return values.reduce((best, value, index) => value > values[best] ? index : best, 0)
}

export function relayNarration(result: CompareResponse, dayIndex: number): string {
  const day = result.candidate.trajectory[dayIndex]
  const previous = result.candidate.trajectory[dayIndex - 1]
  if (!day) return 'TRAJECTORY STANDING BY.'

  if (day.shock.type) {
    const service = strongestImpact(day)
    return `SHOCK DETECTED — ${SERVICE_LABELS[service].toUpperCase()}. SEVERITY ${day.shock.severity.toFixed(2)}.`
  }

  const criticalIndex = day.services_end.findIndex((value) => value < 0.12)
  if (criticalIndex >= 0) {
    const service = result.services[criticalIndex]
    return `CRITICAL FLOOR BREACHED — ${SERVICE_LABELS[service].toUpperCase()}. PRIORITY OVERRIDE.`
  }

  if (day.services_end[4] < 0.3) {
    return 'CIVIC CAPACITY DEGRADED — RECOVERY EFFICIENCY REDUCED.'
  }

  if (previous) {
    const recovery = day.services_end.map((value, index) => value - previous.services_end[index])
    const recoveryIndex = largestValue(recovery)
    if (recovery[recoveryIndex] > 0.003) {
      const service = result.services[recoveryIndex]
      return `REBUILDING ${SERVICE_LABELS[service].toUpperCase()} DISTRICT — ${day.allocation[recoveryIndex].toFixed(0)} UNITS ROUTED.`
    }

    const allocationChange = day.allocation.map((value, index) => value - previous.allocation[index])
    const changeIndex = largestValue(allocationChange)
    if (allocationChange[changeIndex] > 0.5) {
      const service = result.services[changeIndex]
      return `REROUTING ${allocationChange[changeIndex].toFixed(0)} ADDITIONAL UNITS TO ${SERVICE_LABELS[service].toUpperCase()}.`
    }
  }

  const allocationIndex = largestValue(day.allocation)
  const service = result.services[allocationIndex]
  return `DAY ${day.day} — ROUTING ${day.allocation[allocationIndex].toFixed(0)} UNITS TO ${SERVICE_LABELS[service].toUpperCase()}.`
}

export function weightedWellbeing(day: DayResult): number {
  return day.resilience
}
