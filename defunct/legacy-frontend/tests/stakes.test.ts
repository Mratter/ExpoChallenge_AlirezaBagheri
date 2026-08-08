import { describe, expect, it } from 'vitest'
import {
  CRITICAL_FLOOR,
  deriveCityOutcome,
  deriveRunDebrief,
  describeConventionalPlanner,
} from '../src/game/stakes'
import type { CompareResponse, DayResult, Service, Shock } from '../src/types'

const SERVICES: Service[] = [
  'transport',
  'housing',
  'food',
  'healthcare',
  'public_services',
]

const NO_SHOCK: Shock = {
  day: 1,
  type: null,
  severity: 0,
  impact: [0, 0, 0, 0, 0],
  budget_factor: 0,
  forced: false,
}

function makeDay(day: number, servicesEnd: number[], overrides: Partial<DayResult> = {}): DayResult {
  const resilience = servicesEnd.reduce((sum, level) => sum + level, 0) / servicesEnd.length
  return {
    day,
    shock: { ...NO_SHOCK, day },
    available_budget: 180,
    services_before: [...servicesEnd],
    services_after_shock: [...servicesEnd],
    raw_action: [0, 0, 0, 0, 0],
    raw_proposal: [36, 36, 36, 36, 36],
    lower_bounds: [0, 0, 0, 0, 0],
    upper_bounds: [90, 90, 90, 90, 90],
    allocation: [36, 36, 36, 36, 36],
    projection: {
      distance: 0,
      sum: 180,
      constraint_violations: 0,
      violation_breakdown: {
        sum_violations: 0,
        budget_violations: 0,
        lower_violations: 0,
        upper_violations: 0,
        total: 0,
      },
      bindings: [],
    },
    planner_evidence: null,
    support: [0.5, 0.5, 0.5, 0.5, 0.5],
    gain: [0, 0, 0, 0, 0],
    strain: [0, 0, 0, 0, 0],
    services_end: [...servicesEnd],
    resilience,
    reward: resilience,
    ...overrides,
  }
}

function levels(overrides: Partial<Record<Service, number>> = {}): number[] {
  return SERVICES.map((service) => overrides[service] ?? 0.5)
}

function makePlanner(trajectory: DayResult[]) {
  const resilience = trajectory.map((day) => day.resilience)
  return {
    planner: 'fixture',
    rauc: resilience.reduce((sum, value) => sum + value, 0) / resilience.length,
    final_resilience: resilience.at(-1) ?? 0,
    minimum_resilience: Math.min(...resilience),
    post_shock_recovery_shortfall_auc: 0,
    days_to_pre_shock_recovery_after_largest_loss: 0,
    critical_service_days: 0,
    total_projection_distance: 0,
    constraint_violations: 0,
    trajectory,
  }
}

describe('city condition stakes', () => {
  it('uses a strict floor: equality is safe and the first value below it stumbles', () => {
    const outcome = deriveCityOutcome([
      makeDay(1, levels({ housing: CRITICAL_FLOOR })),
      makeDay(2, levels({ housing: CRITICAL_FLOOR - 0.001 })),
    ])

    expect(outcome.conditions[0].stumble).toBe(false)
    expect(outcome.conditions[0].belowFloor).toEqual([])
    expect(outcome.conditions[1].stumble).toBe(true)
    expect(outcome.conditions[1].belowFloor).toEqual(['housing'])
    expect(outcome.firstStumbleDay).toBe(2)
  })

  it('turns a district dark on its third consecutive low day and restores it immediately on recovery', () => {
    const outcome = deriveCityOutcome([
      makeDay(1, levels({ transport: 0.1 })),
      makeDay(2, levels({ transport: 0.11 })),
      makeDay(3, levels({ transport: 0.08 })),
      makeDay(4, levels({ transport: CRITICAL_FLOOR })),
    ])

    expect(outcome.conditions.map((day) => day.darkServices)).toEqual([
      [],
      [],
      ['transport'],
      [],
    ])
    expect(outcome.darkPeriods).toEqual([{
      service: 'transport',
      startedDay: 3,
      endedDay: 3,
      recoveredDay: 4,
    }])
    expect(outcome.recoveryEvents).toEqual([{ service: 'transport', day: 4 }])
    expect(outcome.recoveryCount).toBe(1)
  })

  it('resets the dark streak after even one safe day', () => {
    const outcome = deriveCityOutcome([
      makeDay(1, levels({ housing: 0.1 })),
      makeDay(2, levels({ housing: 0.1 })),
      makeDay(3, levels({ housing: 0.2 })),
      makeDay(4, levels({ housing: 0.1 })),
      makeDay(5, levels({ housing: 0.1 })),
    ])

    expect(outcome.conditions.every((day) => day.darkServices.length === 0)).toBe(true)
    expect(outcome.conditions.at(-1)?.underFloorStreaks.housing).toBe(2)
    expect(outcome.recoveryCount).toBe(1)
  })

  it('does not fall an essential service one day early, then falls on the fourth consecutive day', () => {
    const threeDays = [1, 2, 3].map((day) => makeDay(day, levels({ healthcare: 0.08 })))
    const safe = deriveCityOutcome(threeDays)
    const fallen = deriveCityOutcome([
      ...threeDays,
      makeDay(4, levels({ healthcare: 0.08 })),
    ])

    expect(safe.survived).toBe(true)
    expect(safe.fall).toBeNull()
    expect(fallen.survived).toBe(false)
    expect(fallen.fall).toEqual({
      day: 4,
      causes: [{ kind: 'essential', services: ['healthcare'], consecutiveDays: 4 }],
    })
  })

  it('falls on the second consecutive cascade day even when the affected pair changes', () => {
    const outcome = deriveCityOutcome([
      makeDay(1, levels({ transport: 0.08, housing: 0.09 })),
      makeDay(2, levels({ food: 0.08, healthcare: 0.09 })),
    ])

    expect(outcome.conditions[0].cascadeStreak).toBe(1)
    expect(outcome.fall).toEqual({
      day: 2,
      causes: [{
        kind: 'cascade',
        services: ['food', 'healthcare'],
        consecutiveDays: 2,
      }],
    })
  })

  it('resets a cascade after a non-cascade day', () => {
    const outcome = deriveCityOutcome([
      makeDay(1, levels({ transport: 0.08, housing: 0.09 })),
      makeDay(2, levels({ transport: 0.08 })),
      makeDay(3, levels({ food: 0.08, healthcare: 0.09 })),
    ])

    expect(outcome.survived).toBe(true)
    expect(outcome.conditions.map((day) => day.cascadeStreak)).toEqual([1, 0, 1])
  })

  it('records both valid causes when essential and cascade rules trigger together', () => {
    const outcome = deriveCityOutcome([
      makeDay(1, levels({ food: 0.08 })),
      makeDay(2, levels({ food: 0.08 })),
      makeDay(3, levels({ food: 0.08, transport: 0.08 })),
      makeDay(4, levels({ food: 0.08, transport: 0.08 })),
    ])

    expect(outcome.fall).toEqual({
      day: 4,
      causes: [
        { kind: 'essential', services: ['food'], consecutiveDays: 4 },
        { kind: 'cascade', services: ['transport', 'food'], consecutiveDays: 2 },
      ],
    })
  })

  it('ends outcome accounting at the fall day and ignores later precomputed recovery', () => {
    const outcome = deriveCityOutcome([
      makeDay(1, levels({ food: 0.08, housing: 0.08 }), { resilience: 0.2 }),
      makeDay(2, levels({ food: 0.07, housing: 0.07 }), { resilience: 0.1 }),
      makeDay(3, levels(), { resilience: 0.9 }),
    ])

    expect(outcome.terminalDay).toBe(2)
    expect(outcome.conditions).toHaveLength(2)
    expect(outcome.finalWellbeing).toBe(0.1)
    expect(outcome.resilienceAuc).toBeCloseTo(0.15)
    expect(outcome.recoveryCount).toBe(0)
    expect(outcome.worstMoment?.day).toBe(2)
  })

  it('chooses the earliest minimum as the worst moment and derives survivor metrics from observed days', () => {
    const outcome = deriveCityOutcome([
      makeDay(1, levels({ housing: 0.2 }), { resilience: 0.42 }),
      makeDay(2, levels({ housing: 0.1 }), { resilience: 0.31 }),
      makeDay(3, levels({ housing: 0.2 }), { resilience: 0.31 }),
    ])

    expect(outcome.survived).toBe(true)
    expect(outcome.terminalDay).toBe(3)
    expect(outcome.finalWellbeing).toBe(0.31)
    expect(outcome.resilienceAuc).toBeCloseTo((0.42 + 0.31 + 0.31) / 3)
    expect(outcome.worstMoment).toMatchObject({
      day: 2,
      wellbeing: 0.31,
      weakestService: 'housing',
      weakestLevel: 0.1,
      belowFloor: ['housing'],
    })
  })

  it('never lets budget, allocations, projector evidence, or reward change condition rules', () => {
    const serviceTape = [
      makeDay(1, levels({ public_services: 0.1 })),
      makeDay(2, levels({ public_services: 0.1 })),
      makeDay(3, levels({ public_services: 0.1 })),
    ]
    const differentEconomics = serviceTape.map((day) => ({
      ...day,
      available_budget: 0,
      allocation: [0, 0, 0, 0, 0],
      raw_proposal: [999, 999, 999, 999, 999],
      reward: -999,
      projection: {
        ...day.projection,
        distance: 999,
        sum: 0,
        constraint_violations: 999,
      },
    }))

    const selectCondition = (trajectory: DayResult[]) => {
      const outcome = deriveCityOutcome(trajectory)
      return {
        conditions: outcome.conditions,
        fall: outcome.fall,
        darkPeriods: outcome.darkPeriods,
        recoveryCount: outcome.recoveryCount,
      }
    }

    expect(selectCondition(differentEconomics)).toEqual(selectCondition(serviceTape))
  })
})

describe('run debrief derivations', () => {
  it('evaluates candidate and baseline independently and tallies only actual scheduled disasters', () => {
    const candidateTrajectory = [
      makeDay(1, levels(), { resilience: 0.6 }),
      makeDay(2, levels(), { resilience: 0.7 }),
      makeDay(3, levels(), { resilience: 0.8 }),
      makeDay(4, levels(), { resilience: 0.9 }),
    ]
    const baselineTrajectory = [1, 2, 3, 4].map((day) => (
      makeDay(day, levels({ healthcare: 0.08 }), { resilience: 0.08 })
    ))
    const shockSchedule: Shock[] = [
      { ...NO_SHOCK, day: 1, type: 'aftershock', severity: 0.18, forced: true },
      { ...NO_SHOCK, day: 2, type: 'weather', severity: 0.2, forced: false },
      { ...NO_SHOCK, day: 3, type: 'utility', severity: 0.3, forced: true },
      { ...NO_SHOCK, day: 4, type: 'supply', severity: 0.1, forced: false },
    ]
    const result = {
      services: SERVICES,
      shock_schedule: shockSchedule,
      scenario: {
        name: 'Debrief fixture',
        horizon_days: 4,
        daily_budget: 180,
        initial_services: levels(),
        priorities: [1, 1, 1, 1, 1],
        shock_probability: 0.2,
        severity_min: 0.05,
        severity_max: 0.4,
        forced_shock: { day: 1, type: 'aftershock', severity: 0.18 },
        forced_shocks: [
          { day: 3, type: 'supply', severity: 0.1 },
          { day: 3, type: 'utility', severity: 0.3 },
        ],
      },
      candidate: makePlanner(candidateTrajectory),
      baseline: makePlanner(baselineTrajectory),
    } as CompareResponse

    const debrief = deriveRunDebrief(result)

    expect(debrief.candidate.survived).toBe(true)
    expect(debrief.candidate.finalWellbeing).toBe(0.9)
    expect(debrief.baseline.survived).toBe(false)
    expect(debrief.baseline.fall?.day).toBe(4)
    expect(debrief.disasters).toEqual({ ambient: 2, authored: 0, player: 0, storedUnknown: 2, total: 4 })
    expect(debrief.schedule.find((entry) => entry.day === 3 && entry.type === 'supply')).toMatchObject({
      source: 'Stored forced event — origin unavailable',
      status: 'overridden',
    })

    const withLiveProvenance = deriveRunDebrief(result, {
      authoredShocks: [{ day: 3, type: 'utility', severity: 0.3 }],
      authoredLabel: 'Authored preset · Fault-line city',
      playerShocks: [{ day: 3, type: 'utility', severity: 0.3 }],
    })
    expect(withLiveProvenance.disasters.player).toBe(1)
    expect(withLiveProvenance.disasters.storedUnknown).toBe(1)
    expect(withLiveProvenance.schedule.filter((entry) => entry.day === 3)).toEqual(expect.arrayContaining([
      expect.objectContaining({ source: 'Player', status: 'reached' }),
      expect.objectContaining({ source: 'Authored preset · Fault-line city', status: 'overridden' }),
    ]))
    expect(debrief.conventionalCounterfactual).toBe(
      'A conventional rule-based planner falls on day 4 in this same run at 8% weighted wellbeing, with Healthcare dark from day 3.',
    )
  })

  it('describes a recovered dark period without implying it stayed dark', () => {
    const outcome = deriveCityOutcome([
      makeDay(1, levels({ housing: 0.1 }), { resilience: 0.4 }),
      makeDay(2, levels({ housing: 0.1 }), { resilience: 0.4 }),
      makeDay(3, levels({ housing: 0.1 }), { resilience: 0.4 }),
      makeDay(4, levels({ housing: 0.4 }), { resilience: 0.58 }),
    ])

    expect(describeConventionalPlanner(outcome)).toBe(
      'A conventional rule-based planner ends this same run at 58% weighted wellbeing after Housing went dark on day 3 and recovered on day 4.',
    )
  })

  it('does not count scheduled shocks after the candidate run has already fallen', () => {
    const fallenTrajectory = [1, 2].map((day) => (
      makeDay(day, levels({ transport: 0.08, housing: 0.08 }), { resilience: 0.1 })
    ))
    const survivorTrajectory = [1, 2, 3].map((day) => makeDay(day, levels(), { resilience: 0.5 }))
    const result = {
      services: SERVICES,
      shock_schedule: [
        { ...NO_SHOCK, day: 1, type: 'weather', severity: 0.2, forced: false },
        { ...NO_SHOCK, day: 2 },
        { ...NO_SHOCK, day: 3, type: 'utility', severity: 0.3, forced: true },
      ],
      scenario: {
        name: 'Terminal tally', horizon_days: 3, daily_budget: 180,
        initial_services: levels(), priorities: [1, 1, 1, 1, 1],
        shock_probability: 0.2, severity_min: 0.05, severity_max: 0.4,
        forced_shock: null,
        forced_shocks: [{ day: 3, type: 'utility', severity: 0.3 }],
      },
      candidate: makePlanner(fallenTrajectory),
      baseline: makePlanner(survivorTrajectory),
    } as CompareResponse

    const debrief = deriveRunDebrief(result)
    expect(debrief.disasters).toEqual({ ambient: 1, authored: 0, player: 0, storedUnknown: 0, total: 1 })
    expect(debrief.schedule.find((entry) => entry.day === 3)).toMatchObject({
      status: 'not-reached',
      source: 'Stored forced event — origin unavailable',
    })
  })

  it('does not invent a dark district when the baseline stays above the floor', () => {
    const outcome = deriveCityOutcome([
      makeDay(1, levels(), { resilience: 0.55 }),
      makeDay(2, levels(), { resilience: 0.58 }),
    ])

    expect(describeConventionalPlanner(outcome)).toBe(
      'A conventional rule-based planner ends this same run at 58% weighted wellbeing without any district going dark.',
    )
  })
})
