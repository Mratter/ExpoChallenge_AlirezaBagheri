import type {
  CompareResponse,
  DayResult,
  PlannerResult,
  Scenario,
  Service,
} from './types'

export type PlannerKey = 'candidate' | 'baseline'

export const MATERIAL_GROUP = { start: 0, end: 5 } as const
export const MATERIAL_UTILIZATION = { start: 5, end: 6 } as const
export const CREW_GROUP = { start: 6, end: 11 } as const
export const CREW_UTILIZATION = { start: 11, end: 12 } as const
export const STOCK_RELEASE_GROUP = { start: 12, end: 17 } as const
export const PREPAREDNESS_GROUP = { start: 17, end: 22 } as const

const CRITICAL_SERVICE_FLOOR = 0.30

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value))
}

export function tailStartDay(scenario: Pick<Scenario, 'horizon_days' | 'assessment_tail_days'>): number {
  return scenario.horizon_days - scenario.assessment_tail_days + 1
}

export function canScheduleForcedShock(scenario: Scenario, day: number): boolean {
  return Number.isInteger(day) && day >= 1 && day < tailStartDay(scenario)
}

export function scenarioIssue(scenario: Scenario, seed: number): string | null {
  if (!Number.isInteger(seed) || seed < 0 || seed > 4_294_967_295) return 'Seed must be a uint32 integer.'
  if (!scenario.name.trim() || scenario.name.length > 64) return 'Scenario name must be 1–64 characters.'
  if (scenario.horizon_days !== 30 || scenario.assessment_tail_days !== 3) {
    return 'V3 scenarios use exactly 30 days with a three-day assessment tail.'
  }
  if (scenario.daily_budget < 50 || scenario.daily_budget > 500) return 'Daily material budget must be 50–500.'
  if (scenario.daily_crew_pool < 50 || scenario.daily_crew_pool > 300) return 'Daily crew pool must be 50–300.'
  if (scenario.shock_probability < 0 || scenario.shock_probability > 0.35) return 'Shock probability must be 0–35%.'
  if (scenario.severity_min < 0.05 || scenario.severity_min > 0.25) return 'Minimum severity must be 5–25%.'
  if (scenario.severity_max < 0.10 || scenario.severity_max > 0.40) return 'Maximum severity must be 10–40%.'
  if (scenario.severity_min >= scenario.severity_max) return 'Minimum severity must be lower than maximum severity.'
  if (scenario.initial_services.some((value) => value < 0.05 || value > 0.95)) return 'Each initial service must be 5–95%.'
  if (scenario.priorities.some((value) => value < 0.5 || value > 2)) return 'Each service priority must be 0.5–2.0.'
  if (scenario.recovery_targets.some((value) => value < 0.45 || value > 0.75)) return 'Each recovery target must be 45–75%.'
  const forced = [...(scenario.forced_shock ? [scenario.forced_shock] : []), ...scenario.forced_shocks]
  if (forced.some((shock) => !canScheduleForcedShock(scenario, shock.day))) {
    return `Forced shocks must occur before the assessment tail (days ${tailStartDay(scenario)}–30).`
  }
  if (forced.some((shock) => shock.severity < 0.05 || shock.severity > 0.40)) {
    return 'Forced-shock severity must be 5–40%.'
  }
  return null
}

function plannerFor(result: CompareResponse, planner: PlannerKey): PlannerResult {
  return result[planner]
}

function priorCriticalStreak(result: CompareResponse, planner: PlannerKey, dayIndex: number): number[] {
  let streak: number[] = result.scenario.initial_services.map((value) => value < CRITICAL_SERVICE_FLOOR ? 1 : 0)
  const trajectory = plannerFor(result, planner).trajectory
  for (let index = 0; index < dayIndex; index += 1) {
    streak = trajectory[index].services_end.map((value, serviceIndex) => (
      value < CRITICAL_SERVICE_FLOOR ? streak[serviceIndex] + 1 : 0
    ))
  }
  return streak
}

function daysSinceLastShock(result: CompareResponse, dayIndex: number): number {
  let days: number = result.scenario.horizon_days
  for (let index = 0; index <= dayIndex; index += 1) {
    days = result.shock_schedule[index].type
      ? 0
      : Math.min(result.scenario.horizon_days, days + 1)
  }
  return days
}

function initialWeightedResilience(result: CompareResponse): number {
  const total = result.scenario.priorities.reduce((sum, value) => sum + value, 0)
  return result.scenario.initial_services.reduce(
    (sum, value, index) => sum + value * result.scenario.priorities[index] / total,
    0,
  )
}

/** Reconstructs the exact public observation sent to the selected planner on a day. */
export function observationVectorForDay(
  result: CompareResponse,
  planner: PlannerKey,
  dayIndex: number,
): number[] {
  const trajectory = plannerFor(result, planner).trajectory
  const day = trajectory[dayIndex]
  if (!day) throw new RangeError(`Day index ${dayIndex} is outside the returned trajectory.`)
  const capacity = day.logistics.depot_capacity
  const pending = day.logistics.pending_arrivals
  const previousResilience = dayIndex === 0
    ? initialWeightedResilience(result)
    : trajectory[dayIndex - 1].resilience
  const vector = [
    ...day.services_after_shock,
    ...result.scenario.priorities.map((value) => value / 2),
    ...day.support,
    ...day.shock.impact,
    ...day.logistics.depot_stock_ready.map((value, index) => value / capacity[index]),
    ...pending.map((value, index) => value / Math.max(value + capacity[index], 1e-9)),
    ...day.throughput,
    ...day.logistics.depot_damage_penalty,
    ...day.logistics.depot_damage_days_remaining.map((value) => value / 10),
    ...result.scenario.recovery_targets,
    ...priorCriticalStreak(result, planner, dayIndex).map((value) => value / result.scenario.horizon_days),
    ...day.preparedness_after_hazard,
    day.available_budget / 500,
    day.available_crew / 300,
    (result.scenario.horizon_days - dayIndex) / result.scenario.horizon_days,
    day.shock.assessment_tail ? 1 : 0,
    day.shock.severity,
    day.logistics.road_capacity,
    daysSinceLastShock(result, dayIndex) / result.scenario.horizon_days,
    previousResilience,
    ...day.public_next_day_risk,
  ].map(clamp01)
  if (vector.length !== 73) throw new Error(`Observation reconstruction produced ${vector.length}, expected 73.`)
  return vector
}

export function observationEntriesForDay(
  result: CompareResponse,
  planner: PlannerKey,
  dayIndex: number,
): Array<{ name: string; value: number }> {
  const values = observationVectorForDay(result, planner, dayIndex)
  return result.observation_order.map((name, index) => ({ name, value: values[index] }))
}

export function actionEntriesForDay(
  result: CompareResponse,
  planner: PlannerKey,
  dayIndex: number,
): Array<{ name: string; value: number }> {
  const action = plannerFor(result, planner).trajectory[dayIndex]?.raw_action
  if (!action || action.length !== result.action_order.length) {
    throw new Error('Returned action vector does not match its declared order.')
  }
  return result.action_order.map((name, index) => ({ name, value: action[index] }))
}

export function actionGroupSummaries(day: DayResult): Array<{ label: string; value: number }> {
  const mean = (start: number, end: number) => (
    day.raw_action.slice(start, end).reduce((sum, value) => sum + value, 0) / (end - start)
  )
  return [
    { label: 'Material shares', value: mean(MATERIAL_GROUP.start, MATERIAL_GROUP.end) },
    { label: 'Material use', value: day.raw_action[MATERIAL_UTILIZATION.start] },
    { label: 'Crew shares', value: mean(CREW_GROUP.start, CREW_GROUP.end) },
    { label: 'Crew use', value: day.raw_action[CREW_UTILIZATION.start] },
    { label: 'Stock release', value: mean(STOCK_RELEASE_GROUP.start, STOCK_RELEASE_GROUP.end) },
    { label: 'Preparedness', value: mean(PREPAREDNESS_GROUP.start, PREPAREDNESS_GROUP.end) },
  ]
}

export function serviceLabel(service: Service): string {
  return service === 'public_services'
    ? 'Public services'
    : `${service[0].toUpperCase()}${service.slice(1)}`
}

export function outcomeReasonLabel(reason: string): string {
  const labels: Record<string, string> = {
    zero_hard_violations: 'Hard constraints',
    conservation_verified: 'Material conservation',
    assessment_tail_targets_met: 'Three-day target hold',
    resilience_auc_met: 'Resilience AUC floor',
    critical_service_day_cap_met: 'Critical-service cap',
    terminal_pending_within_capacity: 'Terminal depot capacity',
  }
  return labels[reason] ?? reason.replaceAll('_', ' ')
}
