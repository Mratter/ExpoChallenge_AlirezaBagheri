import type { CompareResponse, DayResult, LogisticsLedger } from '../types'

/**
 * The engine remains day-quantized. These values identify a point inside one
 * returned day without changing, predicting, or stepping the simulation.
 */
export type PresentationCursor = Readonly<{
  dayIndex: number
  progress: number
}>

export type ScalarEndpoints = Readonly<{
  start: number
  end: number
}>

export type VectorEndpoints = Readonly<{
  start: number[]
  end: number[]
}>

export type PresentationLogisticsSample = Readonly<{
  depotCapacity: number[]
  depotStock: number[]
  depotStockEndpoints: VectorEndpoints
  depotDamageFactor: number[]
  depotDamageFactorEndpoints: VectorEndpoints
  depotDamagePenalty: number[]
  depotDamagePenaltyEndpoints: VectorEndpoints
  depotDamageDaysRemaining: number[]
  depotDamageDaysRemainingEndpoints: VectorEndpoints
  throughputFactor: number[]
  throughputFactorEndpoints: VectorEndpoints
  roadCapacity: number
  roadCapacityEndpoints: ScalarEndpoints
  pendingArrivalsLanded: number[]
  sameDayDeliveryLanded: number[]
  landedUnits: number[]
  pendingNextDay: number[]
  pendingNextDayEndpoints: VectorEndpoints
  repairDispatch: number[]
  repairSupply: number[]
  spoilage: number[]
  capacityOverflow: number[]
  constrainedUnits: number[]
}>

export type PresentationSample = Readonly<{
  cursor: PresentationCursor
  dayIndex: number
  dayNumber: number
  dayCount: number
  progress: number
  easedProgress: number
  shockAtBoundary: boolean
  incidentSegment: 'clear' | 'impact' | 'assessment' | 'recovery'
  shockImpactProgress: number
  recoveryProgress: number
  recordedDay: DayResult
  services: number[]
  serviceEndpoints: VectorEndpoints
  servicesAfterShock: number[]
  availableBudget: number
  availableBudgetEndpoints: ScalarEndpoints
  wellbeing: number
  logistics: PresentationLogisticsSample | null
  /**
   * A shallow-cloned day record for view selectors. Daily decisions such as
   * allocations and manifests stay exact; only explicitly visual state fields
   * are replaced with the sampled presentation values.
   */
  visualDay: DayResult
}>

export const PRESENTATION_INTERPOLATION_DISCLOSURE =
  'Visuals interpolate only between returned daily states. Intermediate values are deterministic presentation estimates, not additional simulator steps.'

/** The first 18% of a shock day carries the visible impact into its exact after-shock state. */
export const SHOCK_IMPACT_WINDOW_FRACTION = 0.18
/** A second equal cursor window keeps assessment distinct before response. */
export const SHOCK_ASSESSMENT_WINDOW_FRACTION = SHOCK_IMPACT_WINDOW_FRACTION
export const SHOCK_RESPONSE_START_FRACTION = Math.min(
  1,
  SHOCK_IMPACT_WINDOW_FRACTION + SHOCK_ASSESSMENT_WINDOW_FRACTION,
)
/** Absorbs IEEE-754 round-off when an exact authored boundary is reconstructed by seek. */
const PRESENTATION_BOUNDARY_EPSILON = 1e-9

export type PresentationIncidentStage = 'impact' | 'assessment' | 'response'

/**
 * Pure incident handoff over the shared day cursor. Playback speed and pauses
 * only change how quickly this cursor is reached; they cannot reset a phase.
 */
export function presentationIncidentStage(progress: number): PresentationIncidentStage {
  const normalized = normalizedProgress(progress)
  if (normalized < SHOCK_IMPACT_WINDOW_FRACTION - PRESENTATION_BOUNDARY_EPSILON) return 'impact'
  if (normalized < SHOCK_RESPONSE_START_FRACTION - PRESENTATION_BOUNDARY_EPSILON) return 'assessment'
  return 'response'
}

function finiteOr(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback
}

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, value))
}

function normalizedProgress(progress: number): number {
  return clamp(finiteOr(progress, 0))
}

/** Quintic smoothstep: zero velocity and acceleration at both day boundaries. */
export function presentationEase(progress: number): number {
  const t = normalizedProgress(progress)
  return t * t * t * (t * (t * 6 - 15) + 10)
}

function lerp(start: number, end: number, progress: number): number {
  return start + (end - start) * progress
}

function sameLengthVector(values: readonly number[], length: number, fallback = 0): number[] {
  return Array.from({ length }, (_, index) => finiteOr(values[index] ?? fallback, fallback))
}

function vectorEndpoints(
  startValues: readonly number[],
  endValues: readonly number[],
  length: number,
  fallback = 0,
): VectorEndpoints {
  return {
    start: sameLengthVector(startValues, length, fallback),
    end: sameLengthVector(endValues, length, fallback),
  }
}

function interpolateVector(endpoints: VectorEndpoints, progress: number): number[] {
  return endpoints.start.map((start, index) => lerp(start, endpoints.end[index] ?? start, progress))
}

function adjacentLedgerEndpoints(
  current: LogisticsLedger,
  previous: LogisticsLedger | undefined,
  key: 'depot_damage_factor' | 'depot_damage_penalty' | 'depot_damage_days_remaining' | 'throughput_factor',
  length: number,
): VectorEndpoints {
  const end = sameLengthVector(current[key], length)
  const start = sameLengthVector(previous?.[key] ?? end, length)
  return { start, end }
}

function adjacentDailyVectorEndpoints(
  current: LogisticsLedger,
  previous: LogisticsLedger | undefined,
  key: 'pending_arrivals_landed'
    | 'same_day_delivery_landed'
    | 'pending_arrivals_held'
    | 'same_day_delivery_held'
    | 'pending_next_day'
    | 'repair_dispatch'
    | 'repair_supply'
    | 'spoilage'
    | 'capacity_overflow',
  length: number,
): VectorEndpoints {
  return vectorEndpoints(previous?.[key] ?? [], current[key], length)
}

function sampleLogistics(
  day: DayResult,
  previous: DayResult | undefined,
  serviceCount: number,
  easedProgress: number,
  conditionProgress: number,
): { sample: PresentationLogisticsSample; visualLedger: LogisticsLedger } | null {
  const ledger = day.logistics
  if (!ledger) return null

  const previousLedger = previous?.logistics
  const depotStockEndpoints = vectorEndpoints(
    ledger.depot_stock_before,
    ledger.depot_stock_end,
    serviceCount,
  )
  const depotDamageFactorEndpoints = adjacentLedgerEndpoints(
    ledger,
    previousLedger,
    'depot_damage_factor',
    serviceCount,
  )
  const depotDamagePenaltyEndpoints = adjacentLedgerEndpoints(
    ledger,
    previousLedger,
    'depot_damage_penalty',
    serviceCount,
  )
  const depotDamageDaysRemainingEndpoints = adjacentLedgerEndpoints(
    ledger,
    previousLedger,
    'depot_damage_days_remaining',
    serviceCount,
  )
  const throughputFactorEndpoints = adjacentLedgerEndpoints(
    ledger,
    previousLedger,
    'throughput_factor',
    serviceCount,
  )
  const previousRoadCapacity = previousLedger?.road_capacity ?? ledger.road_capacity
  const roadCapacityEndpoints = {
    start: finiteOr(previousRoadCapacity, ledger.road_capacity),
    end: finiteOr(ledger.road_capacity, previousRoadCapacity),
  }
  // These daily totals are view-layer activity signals. Blending the previous
  // and current returned values avoids a false zero-reset snap at midnight;
  // exact per-day quantities remain in inspectors and the Toolbox.
  const pendingArrivalsLandedEndpoints = adjacentDailyVectorEndpoints(
    ledger, previousLedger, 'pending_arrivals_landed', serviceCount,
  )
  const sameDayDeliveryLandedEndpoints = adjacentDailyVectorEndpoints(
    ledger, previousLedger, 'same_day_delivery_landed', serviceCount,
  )
  const pendingNextDayEndpoints = adjacentDailyVectorEndpoints(
    ledger, previousLedger, 'pending_next_day', serviceCount,
  )
  const repairDispatchEndpoints = adjacentDailyVectorEndpoints(
    ledger, previousLedger, 'repair_dispatch', serviceCount,
  )
  const repairSupplyEndpoints = adjacentDailyVectorEndpoints(
    ledger, previousLedger, 'repair_supply', serviceCount,
  )
  const spoilageEndpoints = adjacentDailyVectorEndpoints(
    ledger, previousLedger, 'spoilage', serviceCount,
  )
  const capacityOverflowEndpoints = adjacentDailyVectorEndpoints(
    ledger, previousLedger, 'capacity_overflow', serviceCount,
  )
  const pendingHeldEndpoints = adjacentDailyVectorEndpoints(
    ledger, previousLedger, 'pending_arrivals_held', serviceCount,
  )
  const sameDayHeldEndpoints = adjacentDailyVectorEndpoints(
    ledger, previousLedger, 'same_day_delivery_held', serviceCount,
  )

  const depotStock = interpolateVector(depotStockEndpoints, easedProgress)
  const depotDamageFactor = interpolateVector(depotDamageFactorEndpoints, conditionProgress)
  const depotDamagePenalty = interpolateVector(depotDamagePenaltyEndpoints, conditionProgress)
  const depotDamageDaysRemaining = interpolateVector(depotDamageDaysRemainingEndpoints, conditionProgress)
  const throughputFactor = interpolateVector(throughputFactorEndpoints, conditionProgress)
  const roadCapacity = lerp(
    roadCapacityEndpoints.start,
    roadCapacityEndpoints.end,
    conditionProgress,
  )
  const pendingArrivalsLanded = interpolateVector(pendingArrivalsLandedEndpoints, easedProgress)
  const sameDayDeliveryLanded = interpolateVector(sameDayDeliveryLandedEndpoints, easedProgress)
  const landedUnits = pendingArrivalsLanded.map(
    (pending, index) => pending + (sameDayDeliveryLanded[index] ?? 0),
  )
  const pendingNextDay = interpolateVector(pendingNextDayEndpoints, easedProgress)
  const repairDispatch = interpolateVector(repairDispatchEndpoints, easedProgress)
  const repairSupply = interpolateVector(repairSupplyEndpoints, easedProgress)
  const spoilage = interpolateVector(spoilageEndpoints, easedProgress)
  const capacityOverflow = interpolateVector(capacityOverflowEndpoints, easedProgress)
  const pendingHeld = interpolateVector(pendingHeldEndpoints, easedProgress)
  const sameDayHeld = interpolateVector(sameDayHeldEndpoints, easedProgress)
  const constrainedUnits = capacityOverflow.map((overflow, index) => Math.max(
    overflow,
    (pendingHeld[index] ?? 0) + (sameDayHeld[index] ?? 0),
  ))

  const sample: PresentationLogisticsSample = {
    depotCapacity: sameLengthVector(ledger.depot_capacity, serviceCount),
    depotStock,
    depotStockEndpoints,
    depotDamageFactor,
    depotDamageFactorEndpoints,
    depotDamagePenalty,
    depotDamagePenaltyEndpoints,
    depotDamageDaysRemaining,
    depotDamageDaysRemainingEndpoints,
    throughputFactor,
    throughputFactorEndpoints,
    roadCapacity,
    roadCapacityEndpoints,
    pendingArrivalsLanded,
    sameDayDeliveryLanded,
    landedUnits,
    pendingNextDay,
    pendingNextDayEndpoints,
    repairDispatch,
    repairSupply,
    spoilage,
    capacityOverflow,
    constrainedUnits,
  }

  // This is intentionally a presentation clone. Exact daily allocation and
  // manifest selectors must continue to receive `recordedDay`, not this value.
  const visualLedger: LogisticsLedger = {
    ...ledger,
    depot_stock_end: depotStock,
    depot_damage_factor: depotDamageFactor,
    depot_damage_penalty: depotDamagePenalty,
    depot_damage_days_remaining: depotDamageDaysRemaining,
    road_capacity: roadCapacity,
    throughput_factor: throughputFactor,
    pending_arrivals_landed: pendingArrivalsLanded,
    same_day_delivery_landed: sameDayDeliveryLanded,
    pending_arrivals_held: pendingHeld,
    same_day_delivery_held: sameDayHeld,
    pending_next_day: pendingNextDay,
    repair_dispatch: repairDispatch,
    repair_supply: repairSupply,
    spoilage,
    capacity_overflow: capacityOverflow,
  }
  return { sample, visualLedger }
}

function weightedWellbeing(services: readonly number[], priorities: readonly number[]): number {
  const weights = services.map((_, index) => Math.max(0, finiteOr(priorities[index] ?? 1, 1)))
  const denominator = Math.max(0.001, weights.reduce((sum, weight) => sum + weight, 0))
  return services.reduce(
    (sum, service, index) => sum + service * (weights[index] ?? 1),
    0,
  ) / denominator
}

/**
 * Samples one immutable engine response at presentation time. It never looks
 * ahead to a future day, so a future shock cannot leak into the current day.
 * A shock starts exactly at its day boundary without a numeric jump. The early
 * impact window carries the view from services_before to services_after_shock;
 * assessment holds that measured floor, then response eases toward the returned
 * services_end state.
 */
export function sampleRunPresentation(
  result: CompareResponse,
  cursor: PresentationCursor,
): PresentationSample {
  const trajectory = result.candidate.trajectory
  if (!trajectory.length) throw new RangeError('Cannot sample an empty candidate trajectory')

  const requestedIndex = Math.floor(finiteOr(cursor.dayIndex, 0))
  const dayIndex = clamp(requestedIndex, 0, trajectory.length - 1)
  const progress = normalizedProgress(cursor.progress)
  const easedProgress = presentationEase(progress)
  const day = trajectory[dayIndex]
  const previous = trajectory[dayIndex - 1]
  const shockAtBoundary = Boolean(day.shock.type)
  const serviceCount = result.services.length
  const shockImpactProgress = shockAtBoundary
    ? presentationEase(progress / SHOCK_IMPACT_WINDOW_FRACTION)
    : 0
  const recoveryProgress = shockAtBoundary
    ? presentationEase(
        (progress - SHOCK_RESPONSE_START_FRACTION)
          / (1 - SHOCK_RESPONSE_START_FRACTION),
      )
    : easedProgress
  const incidentSegment = shockAtBoundary
    ? progress < SHOCK_IMPACT_WINDOW_FRACTION - PRESENTATION_BOUNDARY_EPSILON
      ? 'impact'
      : progress < SHOCK_RESPONSE_START_FRACTION - PRESENTATION_BOUNDARY_EPSILON
        ? 'assessment'
        : 'recovery'
    : 'clear'
  const serviceEndpoints = vectorEndpoints(
    day.services_before,
    day.services_end,
    serviceCount,
  )
  const servicesAfterShock = sameLengthVector(day.services_after_shock, serviceCount)
  const services = shockAtBoundary
    ? progress < SHOCK_IMPACT_WINDOW_FRACTION - PRESENTATION_BOUNDARY_EPSILON
      ? interpolateVector(
          { start: serviceEndpoints.start, end: servicesAfterShock },
          shockImpactProgress,
        )
      : interpolateVector(
          { start: servicesAfterShock, end: serviceEndpoints.end },
          recoveryProgress,
        )
    : interpolateVector(serviceEndpoints, recoveryProgress)
  const availableBudgetEndpoints = {
    start: finiteOr(previous?.available_budget ?? day.available_budget, day.available_budget),
    end: finiteOr(day.available_budget, previous?.available_budget ?? 0),
  }
  const availableBudget = lerp(
    availableBudgetEndpoints.start,
    availableBudgetEndpoints.end,
    shockAtBoundary ? shockImpactProgress : easedProgress,
  )
  const wellbeing = weightedWellbeing(services, result.scenario.priorities)
  const logisticsResult = sampleLogistics(
    day,
    previous,
    serviceCount,
    easedProgress,
    shockAtBoundary ? shockImpactProgress : easedProgress,
  )
  const visualDay: DayResult = {
    ...day,
    available_budget: availableBudget,
    services_end: services,
    resilience: wellbeing,
    logistics: logisticsResult?.visualLedger ?? day.logistics,
  }

  return {
    cursor: { dayIndex, progress },
    dayIndex,
    dayNumber: day.day,
    dayCount: trajectory.length,
    progress,
    easedProgress,
    shockAtBoundary,
    incidentSegment,
    shockImpactProgress,
    recoveryProgress,
    recordedDay: day,
    services,
    serviceEndpoints,
    servicesAfterShock,
    availableBudget,
    availableBudgetEndpoints,
    wellbeing,
    logistics: logisticsResult?.sample ?? null,
    visualDay,
  }
}
