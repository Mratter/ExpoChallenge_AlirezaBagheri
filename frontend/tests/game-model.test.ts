import { describe, expect, it } from 'vitest'
import {
  appendForcedShock,
  closestDistrict,
  damageStateFor,
  isBuildingRebuilding,
  relayNarration,
  shockImpactFor,
  type DamageState,
} from '../src/game/model'
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

function makeDay(overrides: Partial<DayResult> = {}): DayResult {
  const day = overrides.day ?? 1
  return {
    day,
    shock: { ...NO_SHOCK, day },
    available_budget: 180,
    services_before: [0.55, 0.55, 0.55, 0.55, 0.55],
    services_after_shock: [0.55, 0.55, 0.55, 0.55, 0.55],
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
    support: [0.7, 0.7, 0.7, 0.7, 0.7],
    gain: [0, 0, 0, 0, 0],
    strain: [0, 0, 0, 0, 0],
    services_end: [0.55, 0.55, 0.55, 0.55, 0.55],
    resilience: 0.55,
    reward: 0.55,
    ...overrides,
  }
}

function makeResult(trajectory: DayResult[]): CompareResponse {
  return {
    services: SERVICES,
    candidate: { trajectory },
  } as CompareResponse
}

describe('damageStateFor', () => {
  it('maps a service level to a stable per-building distribution containing every damage state', () => {
    const expected: DamageState[] = [
      'intact',
      'slight',
      'slight',
      'moderate',
      'moderate',
      'moderate',
      'rubble',
    ]

    const firstPass = expected.map((_, buildingIndex) => damageStateFor(0.6, buildingIndex))
    const secondPass = expected.map((_, buildingIndex) => damageStateFor(0.6, buildingIndex))

    expect(firstPass).toEqual(expected)
    expect(secondPass).toEqual(firstPass)
    expect(new Set(firstPass)).toEqual(new Set<DamageState>(['intact', 'slight', 'moderate', 'rubble']))
  })

  it('shifts the deterministic distribution toward rubble as the real service level falls', () => {
    const distribution = (serviceLevel: number) =>
      Array.from({ length: 7 }, (_, buildingIndex) => damageStateFor(serviceLevel, buildingIndex))

    expect(distribution(0.9)).toEqual([
      'intact', 'intact', 'intact', 'intact', 'slight', 'slight', 'slight',
    ])
    expect(distribution(0.2)).toEqual([
      'moderate', 'rubble', 'rubble', 'rubble', 'rubble', 'rubble', 'rubble',
    ])
    expect(distribution(0)).toEqual([
      'rubble', 'rubble', 'rubble', 'rubble', 'rubble', 'rubble', 'rubble',
    ])
  })
})

describe('forced disaster game layer', () => {
  it('appends kicks without mutating either the authored scenario or its existing history', () => {
    const scenario = {
      name: 'Kick history',
      horizon_days: 14,
      daily_budget: 180,
      initial_services: [0.4, 0.4, 0.4, 0.4, 0.4],
      priorities: [1, 1, 1, 1, 1],
      shock_probability: 0.2,
      severity_min: 0.1,
      severity_max: 0.28,
      forced_shock: null,
      forced_shocks: [{ day: 3, type: 'weather' as const, severity: 0.17 }],
    }

    const appended = appendForcedShock(scenario, { day: 7, type: 'utility', severity: 0.31 })

    expect(appended.forced_shocks).toEqual([
      { day: 3, type: 'weather', severity: 0.17 },
      { day: 7, type: 'utility', severity: 0.31 },
    ])
    expect(scenario.forced_shocks).toEqual([{ day: 3, type: 'weather', severity: 0.17 }])
    expect(appended).not.toBe(scenario)
  })

  it('uses the authored engine footprint for all five disaster types', () => {
    expect(shockImpactFor('aftershock', 'housing')).toBe(1)
    expect(shockImpactFor('supply', 'food')).toBe(1)
    expect(shockImpactFor('epidemic', 'healthcare')).toBe(1)
    expect(shockImpactFor('utility', 'public_services')).toBe(1)
    expect(shockImpactFor('weather', 'transport')).toBe(0.75)
    expect(shockImpactFor('weather', 'healthcare')).toBe(0.4)
  })

  it('maps a plate point to the same nearest district on every pass', () => {
    const points = [
      [-7.2, -5.3, 'housing'],
      [6.9, -5.4, 'healthcare'],
      [-6.8, 5.1, 'food'],
      [7.4, 5.2, 'transport'],
      [0.1, 7.3, 'public_services'],
    ] as const

    for (const [x, z, service] of points) {
      expect(closestDistrict(x, z).service).toBe(service)
      expect(closestDistrict(x, z).service).toBe(service)
    }
  })
})

describe('relayNarration', () => {
  it('deterministically reports the measured severity and strongest shock impact', () => {
    const shockedDay = makeDay({
      day: 4,
      shock: {
        day: 4,
        type: 'weather',
        severity: 0.314,
        impact: [0.03, 0.08, 0.27, 0.12, 0.05],
        budget_factor: 0.25,
        forced: true,
      },
    })
    const result = makeResult([shockedDay])

    expect(relayNarration(result, 0)).toBe('SHOCK DETECTED — FOOD. SEVERITY 0.31.')
    expect(relayNarration(result, 0)).toBe(relayNarration(result, 0))
  })

  it('reports rebuilding only for the service with the largest observed recovery', () => {
    const previous = makeDay({
      day: 1,
      services_end: [0.51, 0.43, 0.52, 0.48, 0.5],
    })
    const recovered = makeDay({
      day: 2,
      allocation: [24, 47.6, 31, 43, 34.4],
      services_end: [0.512, 0.452, 0.519, 0.49, 0.506],
    })
    const result = makeResult([previous, recovered])

    expect(relayNarration(result, 1)).toBe('REBUILDING HOUSING DISTRICT — 48 UNITS ROUTED.')
    expect(relayNarration(result, 1)).toBe(relayNarration(result, 1))
  })

  it('derives rerouting narration from the largest actual allocation increase', () => {
    const previous = makeDay({
      day: 5,
      allocation: [28, 42, 39, 36, 35],
      services_end: [0.55, 0.55, 0.55, 0.55, 0.55],
    })
    const rerouted = makeDay({
      day: 6,
      allocation: [40.6, 36, 38, 34, 31.4],
      services_end: [0.55, 0.55, 0.55, 0.55, 0.55],
    })

    expect(relayNarration(makeResult([previous, rerouted]), 1)).toBe(
      'REROUTING 13 ADDITIONAL UNITS TO TRANSPORT.',
    )
  })
})

describe('isBuildingRebuilding', () => {
  it('selects a deterministic subset of damaged buildings from positive trajectory recovery', () => {
    const previous = makeDay({
      day: 1,
      services_end: [0.55, 0.45, 0.55, 0.55, 0.55],
    })
    const recovered = makeDay({
      day: 2,
      allocation: [30, 60, 30, 30, 30],
      gain: [0, 0, 0, 0, 0],
      services_end: [0.55, 0.5, 0.55, 0.55, 0.55],
    })

    const selected = Array.from({ length: 7 }, (_, buildingIndex) => buildingIndex)
      .filter((buildingIndex) => isBuildingRebuilding(recovered, previous, 'housing', buildingIndex))

    expect(selected).toEqual([3, 4])
    expect(
      Array.from({ length: 7 }, (_, buildingIndex) => buildingIndex)
        .filter((buildingIndex) => isBuildingRebuilding(recovered, previous, 'housing', buildingIndex)),
    ).toEqual(selected)
  })

  it('does not reconstruct when the observed trajectory failed to recover, even if gain is positive', () => {
    const previous = makeDay({
      day: 1,
      services_end: [0.55, 0.45, 0.55, 0.55, 0.55],
    })
    const declined = makeDay({
      day: 2,
      allocation: [30, 60, 30, 30, 30],
      gain: [0, 0.12, 0, 0, 0],
      services_end: [0.55, 0.44, 0.55, 0.55, 0.55],
    })

    expect(isBuildingRebuilding(declined, previous, 'housing', 3)).toBe(false)
  })

  it('uses the current day gain only when there is no previous trajectory day', () => {
    const firstDay = makeDay({
      day: 2,
      allocation: [30, 60, 30, 30, 30],
      gain: [0, 0.04, 0, 0, 0],
      services_end: [0.55, 0.5, 0.55, 0.55, 0.55],
    })

    expect(isBuildingRebuilding(firstDay, undefined, 'housing', 3)).toBe(true)
    expect(
      isBuildingRebuilding({ ...firstDay, gain: [0, 0.002, 0, 0, 0] }, undefined, 'housing', 3),
    ).toBe(false)
  })
})
