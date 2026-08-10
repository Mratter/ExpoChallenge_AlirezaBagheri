import { describe, expect, it } from 'vitest'
import { defaultScenario, scenariosMatch } from './scenarios'
import type { CompareResponse, DayResult, Vector5, Vector22 } from './types'
import { actionEntriesForDay, canScheduleForcedShock, observationVectorForDay, scenarioIssue, tailStartDay } from './viewModel'

const five = (value: number): Vector5 => [value, value, value, value, value]

function observationFixture(): CompareResponse {
  const day = {
    services_after_shock: five(0.4),
    support: five(0.6),
    shock: { impact: five(0.2), assessment_tail: false, severity: 0.1, type: null },
    logistics: {
      depot_capacity: five(100),
      depot_stock_ready: five(50),
      pending_arrivals: five(20),
      depot_damage_penalty: five(0.1),
      depot_damage_days_remaining: five(2),
      road_capacity: 0.8,
    },
    throughput: five(0.7),
    preparedness_after_hazard: five(0.3),
    available_budget: 250,
    available_crew: 150,
    public_next_day_risk: five(0.05),
    resilience: 0.45,
    services_end: five(0.45),
    raw_action: Array.from({ length: 22 }, (_, index) => index / 100) as Vector22,
  } as unknown as DayResult
  return {
    scenario: { ...defaultScenario, priorities: five(1), initial_services: five(0.4) },
    candidate: { trajectory: [day] },
    baseline: { trajectory: [day] },
    shock_schedule: [{ type: null }],
    observation_order: Array.from({ length: 73 }, (_, index) => `observation_${index}`),
    action_order: Array.from({ length: 22 }, (_, index) => `action_${index}`),
  } as unknown as CompareResponse
}

describe('scenario protocol guards', () => {
  it('reserves days 28–30 as the assessment tail', () => {
    expect(tailStartDay(defaultScenario)).toBe(28)
    expect(canScheduleForcedShock(defaultScenario, 27)).toBe(true)
    expect(canScheduleForcedShock(defaultScenario, 28)).toBe(false)
    expect(canScheduleForcedShock(defaultScenario, 30)).toBe(false)
  })

  it('rejects any forced shock attached inside the tail', () => {
    const scenario = {
      ...defaultScenario,
      forced_shocks: [{ day: 28, type: 'weather' as const, severity: 0.2 }],
    }
    expect(scenarioIssue(scenario, 42)).toMatch(/Forced shocks must occur before the assessment tail/)
  })

  it('accepts the complete default 30-day public scenario', () => {
    expect(scenarioIssue(defaultScenario, 424242)).toBeNull()
  })

  it('treats a backend-reordered but value-identical scenario as unchanged', () => {
    const backendOrder: typeof defaultScenario = {
      assessment_tail_days: defaultScenario.assessment_tail_days,
      daily_budget: defaultScenario.daily_budget,
      daily_crew_pool: defaultScenario.daily_crew_pool,
      forced_shock: defaultScenario.forced_shock,
      forced_shocks: defaultScenario.forced_shocks,
      horizon_days: defaultScenario.horizon_days,
      initial_services: defaultScenario.initial_services,
      name: defaultScenario.name,
      priorities: defaultScenario.priorities,
      recovery_targets: defaultScenario.recovery_targets,
      severity_max: defaultScenario.severity_max,
      severity_min: defaultScenario.severity_min,
      shock_probability: defaultScenario.shock_probability,
    }
    expect(JSON.stringify(backendOrder)).not.toBe(JSON.stringify(defaultScenario))
    expect(scenariosMatch(defaultScenario, backendOrder)).toBe(true)
    expect(scenariosMatch(defaultScenario, { ...backendOrder, daily_budget: 181 })).toBe(false)
  })

  it('reconstructs the complete ordered public input and action receipts', () => {
    const fixture = observationFixture()
    const observation = observationVectorForDay(fixture, 'candidate', 0)
    expect(observation).toHaveLength(73)
    expect(observation.slice(0, 5)).toEqual(five(0.4))
    expect(observation.slice(5, 10)).toEqual(five(0.5))
    expect(observation[60]).toBe(0.5)
    expect(observation[61]).toBe(0.5)
    expect(observation[62]).toBe(1)
    expect(observation.slice(68)).toEqual(five(0.05))
    const actions = actionEntriesForDay(fixture, 'candidate', 0)
    expect(actions).toHaveLength(22)
    expect(actions.at(-1)).toEqual({ name: 'action_21', value: 0.21 })
  })
})
