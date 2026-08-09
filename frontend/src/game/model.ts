import type { CompareResponse, DayResult, ForcedShock, Scenario, Service, ShockType } from '../types'
import {
  CITY_BUILDING_OFFSETS,
  CITY_DISTRICTS,
  type CityBuildingArchetype,
} from './worldLayout'

export type DamageState = 'intact' | 'slight' | 'moderate' | 'rubble'
export type BuildingArchetype = CityBuildingArchetype

export type CityImpactEvent = {
  id: number
  type: ShockType
  severity: number
  day: number
  point: [number, number, number]
  service: Service
  impact: number[]
  wind?: readonly [number, number]
}

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

const DISTRICT_PRESENTATION: Readonly<Record<Service, Omit<DistrictDefinition, 'service' | 'center'>>> = {
  housing: { label: 'Residential quarter', shortLabel: 'Housing', accent: '#bd6b52', body: '#b98269' },
  healthcare: { label: 'Health campus', shortLabel: 'Healthcare', accent: '#e6e2d8', body: '#aab9b4' },
  food: { label: 'Market quarter', shortLabel: 'Food', accent: '#d49a3d', body: '#c88a4c' },
  transport: { label: 'Transit works', shortLabel: 'Transport', accent: '#5a8290', body: '#6e8790' },
  public_services: { label: 'Civic quarter', shortLabel: 'Civic', accent: '#71866a', body: '#8b9a7f' },
}

export const DISTRICTS: DistrictDefinition[] = CITY_DISTRICTS.map((district) => ({
  service: district.service,
  ...DISTRICT_PRESENTATION[district.service],
  center: [...district.center] as [number, number, number],
}))

export const DISTRICT_BUILDING_OFFSETS: ReadonlyArray<readonly [number, number]> = CITY_BUILDING_OFFSETS

export const SHOCK_IMPACTS: Record<ShockType, [number, number, number, number, number]> = {
  aftershock: [0.65, 1, 0.2, 0.35, 0.45],
  supply: [0.35, 0.05, 1, 0.55, 0.1],
  epidemic: [0.1, 0.2, 0.25, 1, 0.35],
  utility: [0.3, 0.35, 0.45, 0.7, 1],
  weather: [0.75, 0.55, 0.5, 0.4, 0.6],
}

export function shockImpactFor(type: ShockType, service: Service): number {
  return SHOCK_IMPACTS[type][serviceIndex(service)]
}

export function appendForcedShock(scenario: Scenario, shock: ForcedShock): Scenario {
  const tailStart = scenario.horizon_days - scenario.assessment_tail_days + 1
  if (shock.day >= tailStart) {
    throw new RangeError(`Forced shocks are disabled during the assessment tail (days ${tailStart}–${scenario.horizon_days}).`)
  }
  return { ...scenario, forced_shocks: [...(scenario.forced_shocks ?? []), shock] }
}

export function closestDistrict(x: number, z: number): DistrictDefinition {
  return DISTRICTS.reduce((closest, district) => {
    const closestDistance = Math.hypot(x - closest.center[0], z - closest.center[2])
    const distance = Math.hypot(x - district.center[0], z - district.center[2])
    return distance < closestDistance ? district : closest
  })
}

const DAMAGE_BUILDING_COUNT = CITY_BUILDING_OFFSETS.length
const DAMAGE_ORDER_STEP = 23
const DAMAGE_BIAS_MIN = -0.84
const DAMAGE_BIAS_MAX = 1.02

const DAMAGE_SEVERITY: Readonly<Record<DamageState, number>> = {
  intact: 0,
  slight: 1,
  moderate: 2,
  rubble: 3,
}

function normalizedBuildingIndex(buildingIndex: number, buildingCount: number): number {
  return ((buildingIndex % buildingCount) + buildingCount) % buildingCount
}

/**
 * A complete, low-discrepancy permutation of the dense district. The coprime step
 * gives every building one unique threshold while avoiding a visible grid sweep.
 */
export function damageOrderForBuilding(
  buildingIndex: number,
  buildingCount = DAMAGE_BUILDING_COUNT,
): number {
  if (!Number.isInteger(buildingCount) || buildingCount < 1) return 0
  const step = buildingCount === DAMAGE_BUILDING_COUNT ? DAMAGE_ORDER_STEP : buildingCount - 1
  return (normalizedBuildingIndex(buildingIndex, buildingCount) * step) % buildingCount
}

export function damageSeverity(state: DamageState): number {
  return DAMAGE_SEVERITY[state]
}

export function damageStateFor(serviceLevel: number, buildingIndex: number): DamageState {
  const rank = damageOrderForBuilding(buildingIndex)
  const quantile = (rank + 0.5) / DAMAGE_BUILDING_COUNT
  const bias = DAMAGE_BIAS_MIN + quantile * (DAMAGE_BIAS_MAX - DAMAGE_BIAS_MIN)
  const score = Math.max(0, Math.min(3, (1 - serviceLevel) * 3.8 + bias))
  if (score < 0.75) return 'intact'
  if (score < 1.5) return 'slight'
  if (score < 2.35) return 'moderate'
  return 'rubble'
}

export function damageStatesForDistrict(serviceLevel: number): DamageState[] {
  return Array.from(
    { length: DAMAGE_BUILDING_COUNT },
    (_, buildingIndex) => damageStateFor(serviceLevel, buildingIndex),
  )
}

export function serviceIndex(service: Service): number {
  return ['transport', 'housing', 'food', 'healthcare', 'public_services'].indexOf(service)
}

function realizedRecovery(
  day: DayResult,
  previous: DayResult | undefined,
  index: number,
): number {
  return previous
    ? day.services_end[index] - previous.services_end[index]
    : day.gain[index]
}

function cyclicRepairRank(
  buildingIndex: number,
  dayNumber: number,
  serviceOffset: number,
  buildingCount: number,
): number {
  const baseRank = damageOrderForBuilding(buildingIndex, buildingCount)
  const dayShift = (dayNumber * 7 + serviceOffset * 5) % buildingCount
  return (baseRank - dayShift + buildingCount) % buildingCount
}

/**
 * Selects a small active-work cohort from a real positive trajectory change. State
 * transitions are preferred, then the daily low-discrepancy window rotates through
 * remaining damaged buildings so a multi-day recovery reads as distributed work.
 */
export function rebuildingCohortForDay(
  day: DayResult,
  previous: DayResult | undefined,
  service: Service,
  buildingCount = DAMAGE_BUILDING_COUNT,
): number[] {
  const index = serviceIndex(service)
  if (index < 0 || buildingCount < 1) return []
  const recovered = realizedRecovery(day, previous, index)
  if (recovered <= 0.002) return []

  const currentLevel = day.services_end[index]
  const priorLevel = previous?.services_end[index] ?? day.services_after_shock[index]
  const allocation = Math.max(0, day.allocation[index] ?? 0)
  const allocationShare = allocation / Math.max(day.available_budget, 1)
  const recoveryShare = Math.min(1, recovered / 0.05)
  const requestedCount = Math.max(
    1,
    Math.min(6, Math.round(1 + allocationShare * 6 + recoveryShare * 2)),
  )

  return Array.from({ length: buildingCount }, (_, buildingIndex) => buildingIndex)
    .filter((buildingIndex) => damageStateFor(currentLevel, buildingIndex) !== 'intact')
    .sort((left, right) => {
      const leftImproved = damageSeverity(damageStateFor(currentLevel, left))
        < damageSeverity(damageStateFor(priorLevel, left))
      const rightImproved = damageSeverity(damageStateFor(currentLevel, right))
        < damageSeverity(damageStateFor(priorLevel, right))
      if (leftImproved !== rightImproved) return leftImproved ? -1 : 1
      return cyclicRepairRank(left, day.day, index, buildingCount)
        - cyclicRepairRank(right, day.day, index, buildingCount)
    })
    .slice(0, requestedCount)
}

export function isBuildingRebuilding(
  day: DayResult,
  previous: DayResult | undefined,
  service: Service,
  buildingIndex: number,
): boolean {
  return rebuildingCohortForDay(day, previous, service).includes(buildingIndex)
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

const RELAY_SHOCK_LABELS: Readonly<Record<ShockType, string>> = {
  aftershock: 'EARTHQUAKE',
  supply: 'SUPPLY DISRUPTION',
  epidemic: 'EPIDEMIC PRESSURE',
  utility: 'UTILITY FAILURE',
  weather: 'SEVERE WEATHER',
}

function recoveryWindow(result: CompareResponse, dayIndex: number, serviceIndex: number): string {
  const day = result.candidate.trajectory[dayIndex]
  const target = day.services_before[serviceIndex]
  const recoveryIndex = result.candidate.trajectory.findIndex((entry, index) => (
    index >= dayIndex && entry.services_end[serviceIndex] >= target - 1e-7
  ))
  if (recoveryIndex < 0) return 'BEYOND CURRENT HORIZON'
  const days = Math.max(1, recoveryIndex - dayIndex)
  return `${Math.max(1, days - 1)}–${days + 1} DAYS`
}

function weightedDeficitRatio(result: CompareResponse, day: DayResult, primaryIndex: number): number {
  const priorities = result.scenario?.priorities ?? [1, 1, 1, 1, 1]
  const primary = priorities[primaryIndex] * Math.max(0.001, 1 - day.services_after_shock[primaryIndex])
  const alternatives = day.services_after_shock.map((level, index) => (
    index === primaryIndex ? Number.POSITIVE_INFINITY : priorities[index] * Math.max(0.001, 1 - level)
  ))
  return primary / Math.max(0.001, Math.min(...alternatives))
}

function logisticsNarration(
  result: CompareResponse,
  day: DayResult,
  serviceIndex: number,
): string {
  const ledger = day.logistics
  if (!ledger || serviceIndex < 0) return ''
  const capacity = ledger.depot_capacity[serviceIndex] ?? 0
  const stock = ledger.depot_stock_end[serviceIndex] ?? 0
  const throughput = ledger.throughput_factor[serviceIndex] ?? 0
  const pendingLanded = ledger.pending_arrivals_landed[serviceIndex] ?? 0
  const sameDayLanded = ledger.same_day_delivery_landed[serviceIndex] ?? 0
  const landed = pendingLanded + sameDayLanded
  const repairDispatch = ledger.repair_dispatch[serviceIndex] ?? 0
  const usable = ledger.repair_supply[serviceIndex] ?? 0
  const queued = ledger.pending_next_day[serviceIndex] ?? 0
  const damagePenalty = ledger.depot_damage_penalty[serviceIndex] ?? 0
  const damageDays = ledger.depot_damage_days_remaining[serviceIndex] ?? 0
  const transfer = ledger.mutual_aid_transfers
    .filter((event) => {
      const donorIndex = result.services.indexOf(event.from_service)
      const receiverIndex = result.services.indexOf(event.to_service)
      return donorIndex >= 0
        && receiverIndex >= 0
        && Math.abs((ledger.mutual_aid_net[donorIndex] ?? 0) + event.units) <= 1e-6
        && Math.abs((ledger.mutual_aid_net[receiverIndex] ?? 0) - event.units) <= 1e-6
    })
    .map((event) => ` MUTUAL AID: ${event.units.toFixed(1)} UNITS ${SERVICE_LABELS[event.from_service].toUpperCase()} TO ${SERVICE_LABELS[event.to_service].toUpperCase()}.`)
    .join('')
  const foodIndex = result.services.indexOf('food')
  const foodSpoilage = foodIndex < 0 ? 0 : ledger.spoilage[foodIndex] ?? 0
  const spoilage = foodSpoilage > 0.0001
    ? ` ${foodSpoilage.toFixed(2)} FOOD UNITS EXPIRED IN STORAGE.`
    : ''
  const damage = damagePenalty > 1e-7
    ? ` DAMAGE PENALTY ${damagePenalty.toFixed(2)} WITH ${damageDays} DAY${damageDays === 1 ? '' : 'S'} REMAINING;`
    : ''
  return ` DEPOT LEDGER: ${landed.toFixed(1)} UNITS LANDED; ${stock.toFixed(1)}/${capacity.toFixed(0)} STOCK;${damage} ${Math.round(throughput * 100)}% EFFECTIVE THROUGHPUT; ${repairDispatch.toFixed(1)} DISPATCHED, ${usable.toFixed(1)} EFFECTIVE REPAIR-SUPPLY UNITS; ${queued.toFixed(1)} QUEUED FOR DAY ${day.day + 1}.${transfer}${spoilage}`
}

export function relayNarration(result: CompareResponse, dayIndex: number): string {
  const day = result.candidate.trajectory[dayIndex]
  const previous = result.candidate.trajectory[dayIndex - 1]
  if (!day) return 'TRAJECTORY STANDING BY.'

  const criticalIndex = day.services_end.findIndex((value) => value < 0.12)
  if (day.shock.type && criticalIndex >= 0) {
    const impactService = strongestImpact(day)
    const criticalService = result.services[criticalIndex]
    return `${RELAY_SHOCK_LABELS[day.shock.type as ShockType]} — RAW ${day.shock.severity.toFixed(2)}. DAMAGE ASSESSMENT: ${SERVICE_LABELS[impactService].toUpperCase()} FOOTPRINT. CRITICAL FLOOR BREACHED — ${SERVICE_LABELS[criticalService].toUpperCase()}.${logisticsNarration(result, day, criticalIndex)}`
  }

  if (day.shock.type) {
    const service = strongestImpact(day)
    const index = result.services.indexOf(service)
    const nominalBudget = result.scenario?.daily_budget ?? day.available_budget
    const arrivalClause = day.available_budget < nominalBudget - 0.001
      ? `ARRIVALS REDUCED TO ${day.available_budget.toFixed(1)} UNITS — RESEQUENCING.`
      : `FULL ${day.available_budget.toFixed(1)}-UNIT ARRIVAL RECEIVED.`
    const supplyClause = day.shock.type === 'supply' ? ' CRITICAL GOODS FIRST.' : ''
    const cascadeClause = day.shock.type === 'utility'
      ? ` CASCADE NOTE: RETURNED DEPENDENCY SUPPORT FLOOR ${Math.min(...day.support).toFixed(2)}.`
      : ''
    const assignmentClause = day.logistics
      ? `PLANNER ASSIGNED ${day.allocation[index].toFixed(1)} UNITS TO ${SERVICE_LABELS[service].toUpperCase()}.`
      : `STAGING ${day.allocation[index].toFixed(1)} UNITS AT ${SERVICE_LABELS[service].toUpperCase()} POINT OF DISTRIBUTION.`
    return `${RELAY_SHOCK_LABELS[day.shock.type as ShockType]} — RAW ${day.shock.severity.toFixed(2)}. ${arrivalClause}${supplyClause}${cascadeClause} ${assignmentClause} RECOVERY RANGE ${recoveryWindow(result, dayIndex, index)}.${logisticsNarration(result, day, index)}`
  }

  if (criticalIndex >= 0) {
    const service = result.services[criticalIndex]
    const movement = day.logistics ? 'ASSIGNED' : 'STAGED'
    return `CRITICAL FLOOR BREACHED — ${SERVICE_LABELS[service].toUpperCase()}. INCIDENT COMMAND PRIORITY OVERRIDE; ${day.allocation[criticalIndex].toFixed(1)} UNITS ${movement}.${logisticsNarration(result, day, criticalIndex)}`
  }

  const civicIndex = result.services.indexOf('public_services')
  if (civicIndex >= 0 && day.services_end[civicIndex] < 0.3) {
    const minimumSupport = Math.min(...day.support)
    return `CIVIC CAPACITY DEGRADED — RETURNED SUPPORT FLOOR ${minimumSupport.toFixed(2)}; RECOVERY EFFICIENCY REDUCED.${logisticsNarration(result, day, civicIndex)}`
  }

  if (previous) {
    const crossed = (service: Service, threshold: number) => {
      const index = result.services.indexOf(service)
      return index >= 0
        && previous.services_end[index] < threshold
        && day.services_end[index] >= threshold
    }
    const housingIndex = result.services.indexOf('housing')
    const transportIndex = result.services.indexOf('transport')
    if (
      housingIndex >= 0
      && transportIndex >= 0
      && day.services_end[housingIndex] >= 0.65
      && day.services_end[transportIndex] >= 0.65
      && (previous.services_end[housingIndex] < 0.65 || previous.services_end[transportIndex] < 0.65)
    ) {
      return `REOPENING MILESTONE — RETURN TRAFFIC AND REGULAR PICKUP LOOP RESUMED. HOUSING ${day.services_end[housingIndex].toFixed(2)}; TRANSPORT ${day.services_end[transportIndex].toFixed(2)}.${logisticsNarration(result, day, transportIndex)}`
    }
    const milestone = ([
      ['food', 0.58, 'MARKET DISTRIBUTION RESTORED'],
      ['transport', 0.55, 'TRANSIT LOOP RESUMED'],
      ['public_services', 0.60, 'CIVIC COORDINATION FLAG RAISED'],
    ] as const).find(([service, threshold]) => crossed(service, threshold))
    if (milestone) {
      const [service, , label] = milestone
      const index = result.services.indexOf(service)
      return `REOPENING MILESTONE — ${label}. ${SERVICE_LABELS[service].toUpperCase()} STATE ${day.services_end[index].toFixed(2)}.${logisticsNarration(result, day, index)}`
    }

    const recovery = day.services_end.map((value, index) => value - previous.services_end[index])
    const recoveryIndex = largestValue(recovery)
    if (recovery[recoveryIndex] > 0.003) {
      const service = result.services[recoveryIndex]
      const repairSupply = day.logistics?.repair_supply[recoveryIndex]
      const movement = repairSupply === undefined
        ? `${day.allocation[recoveryIndex].toFixed(1)} UNITS MOVING THROUGH THE POINT OF DISTRIBUTION.`
        : `${repairSupply.toFixed(1)} EFFECTIVE REPAIR-SUPPLY UNITS MOVING FROM THE POINT OF DISTRIBUTION.`
      return `RECOVERY WAVE — ${SERVICE_LABELS[service].toUpperCase()} GAIN ${recovery[recoveryIndex].toFixed(3)}. ${movement}${logisticsNarration(result, day, recoveryIndex)}`
    }

    const allocationChange = day.allocation.map((value, index) => value - previous.allocation[index])
    const changeIndex = largestValue(allocationChange)
    if (allocationChange[changeIndex] > 0.5) {
      const service = result.services[changeIndex]
      const deferredIndex = allocationChange.reduce((lowest, value, index, values) => value < values[lowest] ? index : lowest, 0)
      const ratio = weightedDeficitRatio(result, day, changeIndex)
      const movement = day.logistics ? 'ASSIGNED' : 'STAGED'
      return `${SERVICE_LABELS[result.services[deferredIndex]].toUpperCase()} SEQUENCED LATER — ${SERVICE_LABELS[service].toUpperCase()} DEFICIT SIGNAL ${ratio.toFixed(1)}× WEIGHTED. ${allocationChange[changeIndex].toFixed(1)} ADDITIONAL UNITS ${movement}.${logisticsNarration(result, day, changeIndex)}`
    }
  }

  const allocationIndex = largestValue(day.allocation)
  const service = result.services[allocationIndex]
  const leadClause = day.logistics
    ? `ASSIGNING ${day.allocation[allocationIndex].toFixed(1)} UNITS TO ${SERVICE_LABELS[service].toUpperCase()}`
    : `STAGING ${day.allocation[allocationIndex].toFixed(1)} UNITS AT ${SERVICE_LABELS[service].toUpperCase()} POINT OF DISTRIBUTION`
  return `DAY ${day.day} SITREP — ${leadClause}; ALL ${day.available_budget.toFixed(1)} ARRIVAL UNITS ASSIGNED.${logisticsNarration(result, day, allocationIndex)}`
}

export function weightedWellbeing(day: DayResult): number {
  return day.resilience
}
