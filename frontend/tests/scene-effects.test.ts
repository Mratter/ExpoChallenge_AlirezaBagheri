import { describe, expect, it } from 'vitest'
import { convoyPlansForDay, repairPlansForDay, sceneMotionStep } from '../src/game/SceneEffects'
import type { DayResult } from '../src/types'

function trajectoryDay(overrides: Partial<DayResult> = {}): DayResult {
  return {
    day: 4,
    available_budget: 180,
    allocation: [10, 20, 30, 40, 80],
    gain: [0, 0, 0, 0, 0],
    services_after_shock: [0.4, 0.4, 0.4, 0.4, 0.4],
    services_end: [0.4, 0.4, 0.4, 0.4, 0.4],
    ...overrides,
  } as DayResult
}

describe('trajectory-derived city traffic', () => {
  it('binds convoy count and speed monotonically to the exact allocation share', () => {
    const plans = convoyPlansForDay(trajectoryDay())
    const transport = plans.find((plan) => plan.service === 'transport')
    const civic = plans.find((plan) => plan.service === 'public_services')

    expect(transport?.allocation).toBe(10)
    expect(civic?.allocation).toBe(80)
    expect(transport?.vehicleCount).toBe(1)
    expect(civic?.vehicleCount).toBe(5)
    expect(civic!.speed).toBeGreaterThan(transport!.speed)
  })

  it('omits repair activity unless the realized candidate trajectory improved', () => {
    const previous = trajectoryDay({
      day: 3,
      services_end: [0.5, 0.42, 0.46, 0.48, 0.44],
    })
    const current = trajectoryDay({
      day: 4,
      allocation: [30, 60, 30, 30, 30],
      services_end: [0.5, 0.451, 0.459, 0.48, 0.466],
    })

    const plans = repairPlansForDay(current, previous)

    expect(plans.map((plan) => plan.service)).toEqual(['housing', 'public_services'])
    expect(plans.find((plan) => plan.service === 'housing')?.realizedGain).toBeCloseTo(0.031)
    expect(plans.every((plan) => plan.realizedGain > 0.002)).toBe(true)
    expect(plans.every((plan) => Number.isInteger(plan.buildingIndex))).toBe(true)
  })

  it('derives first-day repair motion from measured post-shock-to-end recovery', () => {
    const plans = repairPlansForDay(trajectoryDay({
      day: 1,
      services_after_shock: [0.3, 0.35, 0.4, 0.45, 0.5],
      services_end: [0.31, 0.35, 0.4, 0.45, 0.5],
      gain: [0.01, 0, 0, 0, 0],
    }), undefined)

    expect(plans.map((plan) => plan.service)).toEqual(['transport'])
  })

  it('keeps a trajectory-derived dark district still', () => {
    const day = trajectoryDay({
      allocation: [20, 70, 30, 30, 30],
      services_after_shock: [0.4, 0.3, 0.4, 0.4, 0.4],
      services_end: [0.41, 0.34, 0.41, 0.41, 0.41],
      gain: [0.01, 0.04, 0.01, 0.01, 0.01],
    })

    expect(convoyPlansForDay(day, undefined, ['housing']).some((plan) => plan.service === 'housing')).toBe(false)
    expect(repairPlansForDay(day, undefined, undefined, ['housing']).some((plan) => plan.service === 'housing')).toBe(false)
  })

  it('freezes every continuous city animation step under reduced motion', () => {
    expect(sceneMotionStep(0.016, true)).toBe(0)
    expect(sceneMotionStep(0.016, false)).toBe(0.016)
    expect(sceneMotionStep(0.5, false)).toBe(0.075)
  })
})
