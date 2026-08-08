import { describe, expect, it } from 'vitest'
import {
  appendForcedShock,
  closestDistrict,
  DISTRICTS,
  damageOrderForBuilding,
  damageSeverity,
  damageStateFor,
  damageStatesForDistrict,
  isBuildingRebuilding,
  rebuildingCohortForDay,
  relayNarration,
  shockImpactFor,
  type DamageState,
} from '../src/game/model'
import { CITY_BUILDING_OFFSETS, CITY_DISTRICTS } from '../src/game/worldLayout'
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
  it('gives all 36 buildings unique low-discrepancy damage ranks', () => {
    const ranks = CITY_BUILDING_OFFSETS.map((_, buildingIndex) => (
      damageOrderForBuilding(buildingIndex)
    ))

    expect(ranks).toHaveLength(36)
    expect(new Set(ranks).size).toBe(36)
    expect([...ranks].sort((left, right) => left - right)).toEqual(
      Array.from({ length: 36 }, (_, index) => index),
    )
  })

  it('maps a service level to a stable dense distribution containing every damage state', () => {
    const firstPass = damageStatesForDistrict(0.6)
    const secondPass = damageStatesForDistrict(0.6)

    expect(firstPass).toHaveLength(CITY_BUILDING_OFFSETS.length)
    expect(secondPass).toEqual(firstPass)
    expect(new Set(firstPass)).toEqual(new Set<DamageState>(['intact', 'slight', 'moderate', 'rubble']))
  })

  it('stages a real multi-day recovery across small building subsets', () => {
    const levels = [0.36, 0.40, 0.44, 0.48, 0.52]
    const distributions = levels.map(damageStatesForDistrict)
    const totalDamage = distributions.map((states) => (
      states.reduce((total, state) => total + damageSeverity(state), 0)
    ))

    for (let dayIndex = 1; dayIndex < distributions.length; dayIndex += 1) {
      const changed = distributions[dayIndex].filter(
        (state, buildingIndex) => state !== distributions[dayIndex - 1][buildingIndex],
      )
      expect(changed.length).toBeGreaterThan(0)
      expect(changed.length).toBeLessThanOrEqual(12)
      expect(totalDamage[dayIndex]).toBeLessThan(totalDamage[dayIndex - 1])
      distributions[dayIndex].forEach((state, buildingIndex) => {
        expect(damageSeverity(state)).toBeLessThanOrEqual(
          damageSeverity(distributions[dayIndex - 1][buildingIndex]),
        )
      })
    }

    expect(damageStatesForDistrict(0)).toEqual(Array<DamageState>(36).fill('rubble'))
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
    expect(DISTRICTS.map(({ service, center }) => ({ service, center }))).toEqual(
      CITY_DISTRICTS.map(({ service, center }) => ({ service, center: [...center] })),
    )
    for (const district of CITY_DISTRICTS) {
      const [x, , z] = district.center
      expect(closestDistrict(x, z).service).toBe(district.service)
      expect(closestDistrict(x + 0.2, z - 0.2).service).toBe(district.service)
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

    expect(relayNarration(result, 0)).toBe('SEVERE WEATHER — RAW 0.31. FULL 180.0-UNIT ARRIVAL RECEIVED. STAGING 36.0 UNITS AT FOOD POINT OF DISTRIBUTION. RECOVERY RANGE 0–1 DAYS.')
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

    expect(relayNarration(result, 1)).toBe('RECOVERY WAVE — HOUSING GAIN 0.022. 47.6 UNITS MOVING THROUGH THE POINT OF DISTRIBUTION.')
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
      'HOUSING SEQUENCED LATER — TRANSPORT DEFICIT SIGNAL 1.0× WEIGHTED. 12.6 ADDITIONAL UNITS STAGED.',
    )
  })

  it('announces reopening milestones only when the recorded threshold is crossed', () => {
    const previous = makeDay({
      day: 8,
      services_end: [0.54, 0.62, 0.57, 0.61, 0.59],
    })
    const reopened = makeDay({
      day: 9,
      services_end: [0.56, 0.63, 0.59, 0.62, 0.61],
    })
    expect(relayNarration(makeResult([previous, reopened]), 1)).toBe(
      'REOPENING MILESTONE — MARKET DISTRIBUTION RESTORED. FOOD STATE 0.59.',
    )
  })
})

describe('isBuildingRebuilding', () => {
  it('keeps crews on deterministic sites across a real multi-day recovery', () => {
    const days = [0.40, 0.43, 0.46, 0.49, 0.52].map((housing, dayIndex) => makeDay({
      day: dayIndex + 1,
      allocation: [30, 60, 30, 30, 30],
      services_end: [0.55, housing, 0.55, 0.55, 0.55],
    }))

    const cohorts = days.slice(1).map((day, index) => (
      rebuildingCohortForDay(day, days[index], 'housing')
    ))

    cohorts.forEach((cohort, cohortIndex) => {
      expect(cohort.length).toBeGreaterThan(0)
      expect(cohort.length).toBeLessThanOrEqual(6)
      expect(new Set(cohort).size).toBe(cohort.length)
      expect(cohort.every((buildingIndex) => (
        damageStateFor(days[cohortIndex + 1].services_end[1], buildingIndex) !== 'intact'
      ))).toBe(true)
    })
    for (let index = 1; index < cohorts.length; index += 1) {
      const previousMembers = new Set(cohorts[index - 1])
      expect(cohorts[index].some((buildingIndex) => previousMembers.has(buildingIndex))).toBe(true)
      const currentMembers = new Set(cohorts[index])
      for (const formerSite of cohorts[index - 1]) {
        if (currentMembers.has(formerSite)) continue
        expect(damageStateFor(days[index + 1].services_end[1], formerSite)).toBe('intact')
      }
    }

    expect(rebuildingCohortForDay(days[1], days[0], 'housing')).toEqual(cohorts[0])
    expect(
      Array.from({ length: 36 }, (_, buildingIndex) => buildingIndex)
        .filter((buildingIndex) => isBuildingRebuilding(days[1], days[0], 'housing', buildingIndex)),
    ).toEqual([...cohorts[0]].sort((left, right) => left - right))
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

    expect(rebuildingCohortForDay(declined, previous, 'housing')).toEqual([])
    expect(isBuildingRebuilding(declined, previous, 'housing', 3)).toBe(false)
  })

  it('uses the current day gain only when there is no previous trajectory day', () => {
    const firstDay = makeDay({
      day: 2,
      allocation: [30, 60, 30, 30, 30],
      gain: [0, 0.04, 0, 0, 0],
      services_end: [0.55, 0.5, 0.55, 0.55, 0.55],
    })

    const cohort = rebuildingCohortForDay(firstDay, undefined, 'housing')
    expect(cohort.length).toBeGreaterThan(0)
    expect(cohort.length).toBeLessThanOrEqual(6)
    expect(cohort.every((buildingIndex) => (
      isBuildingRebuilding(firstDay, undefined, 'housing', buildingIndex)
    ))).toBe(true)
    expect(rebuildingCohortForDay(
      { ...firstDay, gain: [0, 0.002, 0, 0, 0] },
      undefined,
      'housing',
    )).toEqual([])
  })
})
