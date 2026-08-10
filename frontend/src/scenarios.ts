import {
  defaultCompareRequest,
  defaultScenario as backendDefaultScenario,
  type Scenario,
} from './types'

function vectorsMatch(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index])
}

function shocksMatch(
  left: Scenario['forced_shock'],
  right: Scenario['forced_shock'],
): boolean {
  if (left === null || right === null) return left === right
  return left.day === right.day && left.type === right.type && left.severity === right.severity
}

/** Compare the scenario contract by value; API key order is not meaningful. */
export function scenariosMatch(left: Scenario, right: Scenario): boolean {
  return (
    left.name === right.name
    && left.horizon_days === right.horizon_days
    && left.daily_budget === right.daily_budget
    && left.daily_crew_pool === right.daily_crew_pool
    && left.shock_probability === right.shock_probability
    && left.severity_min === right.severity_min
    && left.severity_max === right.severity_max
    && left.assessment_tail_days === right.assessment_tail_days
    && vectorsMatch(left.initial_services, right.initial_services)
    && vectorsMatch(left.priorities, right.priorities)
    && vectorsMatch(left.recovery_targets, right.recovery_targets)
    && shocksMatch(left.forced_shock, right.forced_shock)
    && left.forced_shocks.length === right.forced_shocks.length
    && left.forced_shocks.every((shock, index) => shocksMatch(shock, right.forced_shocks[index] ?? null))
  )
}

export const defaultScenario: Scenario = {
  ...backendDefaultScenario,
  name: 'Central district recovery exercise',
}

export const defaultSeed = defaultCompareRequest.seed
