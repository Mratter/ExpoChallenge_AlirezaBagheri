import { describe, expect, it } from 'vitest'
import {
  tutorialLessonFor,
  tutorialPhaseForDay,
} from '../src/game/TutorialGuide'
import type { CompareResponse, DayResult, Shock } from '../src/types'

const WEATHER: Shock = {
  day: 2,
  type: 'weather',
  severity: 0.24,
  impact: [0.75, 0.55, 0.5, 0.4, 0.6],
  budget_factor: 0.25,
  forced: true,
}

function day(dayNumber: number, shock: Shock | null = null): DayResult {
  const previous = dayNumber === 2 ? 0.4 : 0.34 + dayNumber * 0.01
  const afterShock = dayNumber === 2 ? 0.328 : previous
  return {
    day: dayNumber,
    shock: shock ?? { day: dayNumber, type: null, severity: 0, impact: [0, 0, 0, 0, 0], budget_factor: 0, forced: false },
    available_budget: dayNumber === 2 ? 169.2 : 180,
    services_before: [previous, 0.4, 0.4, 0.4, 0.4],
    services_after_shock: [afterShock, 0.4, 0.4, 0.4, 0.4],
    raw_action: [0, 0, 0, 0, 0], raw_proposal: [40, 35, 35, 35, 35],
    lower_bounds: [0, 0, 0, 0, 0], upper_bounds: [90, 90, 90, 90, 90],
    allocation: [40, 35, 35, 35, dayNumber === 2 ? 24.2 : 35],
    projection: { distance: 0, sum: dayNumber === 2 ? 169.2 : 180, constraint_violations: 0, violation_breakdown: { sum_violations: 0, budget_violations: 0, lower_violations: 0, upper_violations: 0, total: 0 }, bindings: [] },
    planner_evidence: null, support: [0.7, 0.7, 0.7, 0.7, 0.7],
    gain: [0.01, 0.01, 0.01, 0.01, 0.01], strain: [0, 0, 0, 0, 0],
    services_end: [afterShock + 0.01, 0.41, 0.41, 0.41, 0.41], resilience: 0.4, reward: 0.4,
    logistics: {
      depot_capacity: [400, 400, 400, 400, 400], depot_stock_before: [0, 0, 0, 0, 0],
      pending_arrivals: [0, 0, 0, 0, 0], pending_arrivals_landed: [0, 0, 0, 0, 0], pending_arrivals_held: [0, 0, 0, 0, 0],
      depot_stock_after_pending: [0, 0, 0, 0, 0], depot_damage_penalty: [0, 0, 0, 0, 0], depot_damage_days_remaining: [0, 0, 0, 0, 0], depot_damage_factor: [1, 1, 1, 1, 1],
      road_capacity: 1, throughput_factor: [1, 1, 1, 1, 1], mutual_aid_transfers: [], mutual_aid_net: [0, 0, 0, 0, 0], depot_stock_ready: [0, 0, 0, 0, 0], pending_next_day: [0, 0, 0, 0, 0],
      same_day_delivery_scheduled: [0, 0, 0, 0, 0], same_day_delivery_landed: [0, 0, 0, 0, 0], same_day_delivery_held: [0, 0, 0, 0, 0], delayed_delivery_scheduled: [0, 0, 0, 0, 0], repair_reserve: [0, 0, 0, 0, 0], repair_request: [0, 0, 0, 0, 0], repair_dispatch: [0, 0, 0, 0, 0], repair_supply: [31.5, 20, 20, 20, 20], spoilage: [0, 0, 0, 0, 0], depot_stock_end: [0, 0, 0, 0, 0], capacity_overflow: [0, 0, 0, 0, 0], conservation_residual: [0, 0, 0, 0, 0],
    },
  }
}

function tutorialResult(): CompareResponse {
  const trajectory = Array.from({ length: 8 }, (_, index) => day(index + 1, index === 1 ? WEATHER : null))
  return {
    scenario: { name: 'Relay City tutorial', horizon_days: 8, daily_budget: 180, initial_services: [0.4, 0.4, 0.4, 0.4, 0.4], priorities: [1, 1, 1, 1, 1], shock_probability: 0, severity_min: 0.1, severity_max: 0.28, forced_shock: null, forced_shocks: [{ day: 2, type: 'weather', severity: 0.24 }] },
    services: ['transport', 'housing', 'food', 'healthcare', 'public_services'],
    shock_schedule: trajectory.map((entry) => entry.shock), candidate: { trajectory }, baseline: { trajectory },
  } as CompareResponse
}

describe('returned-data tutorial guide', () => {
  it('advances through the five instructional phases without claiming planner foresight', () => {
    expect(tutorialPhaseForDay(0, 'CLEAR')).toBe('TELEGRAPH')
    expect(tutorialPhaseForDay(1, 'IMPACT')).toBe('IMPACT')
    expect(tutorialPhaseForDay(1, 'ASSESSMENT')).toBe('ASSESSMENT')
    expect(tutorialPhaseForDay(2, 'IMPACT')).toBe('RESPONSE')
    expect(tutorialPhaseForDay(3, 'CLEAR')).toBe('RECOVERY')
  })

  it('cites only returned schedule, service, allocation, and logistics values', () => {
    const result = tutorialResult()
    expect(tutorialLessonFor(result, 0, 'TELEGRAPH').evidence).toBe(
      'Returned day 2: Weather at raw 0.24; strongest typed footprint Transport at 0.75.',
    )
    expect(tutorialLessonFor(result, 1, 'IMPACT').evidence).toContain('40.0% to 32.8% before recovery allocation')
    expect(tutorialLessonFor(result, 1, 'RESPONSE').evidence).toContain('40.0 units; 169.2 of 169.2 available units are assigned')
    expect(tutorialLessonFor(result, 3, 'CLEAR').source).toBe('candidate.trajectory[1].services_before / candidate.trajectory[3].services_end / logistics.repair_supply')
  })

  it('ends the incident-specific arc at its pre-event target and then teaches citywide change', () => {
    const result = tutorialResult()
    const ongoing = tutorialLessonFor(result, 3, 'RECOVERY')
    const restored = tutorialLessonFor(result, 4, 'CLEAR')

    expect(ongoing.explanation).toContain('still the same incident recovery arc')
    expect(ongoing.evidence).toContain('against its 40.0% pre-event target')
    expect(restored.phase).toBe('RECOVERY')
    expect(restored.heading).toBe('The incident target is restored; widen the view.')
    expect(restored.explanation).toContain('crossed its pre-event target on returned day 5')
    expect(restored.explanation).toContain('does not extend that incident recovery arc')
    expect(restored.evidence).toContain('largest current-day city improvement')
    expect(restored.explanation).not.toContain('still the same incident recovery arc')
  })
})
