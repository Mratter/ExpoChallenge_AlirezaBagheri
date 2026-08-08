import { describe, expect, it } from 'vitest'
import {
  convoyPlansForDay,
  hasCurrentDayRepairWork,
  repairActivityMotionTime,
  repairPlansForDay,
  repairPresentationCursor,
} from '../src/game/SceneEffects'
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
    expect(transport?.vehicleCount).toBe(2)
    expect(civic?.vehicleCount).toBe(9)
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

    expect(new Set(plans.map((plan) => plan.service))).toEqual(new Set(['housing', 'public_services']))
    expect(plans.find((plan) => plan.service === 'housing')?.realizedGain).toBeCloseTo(0.031)
    expect(plans.every((plan) => plan.realizedGain > 0.002)).toBe(true)
    expect(plans.every((plan) => Number.isInteger(plan.buildingIndex))).toBe(true)
    expect(new Set(plans.map((plan) => `${plan.service}-${plan.buildingIndex}`)).size).toBe(plans.length)
    expect(plans.reduce((total, plan) => total + plan.vehicleCount, 0)).toBeLessThanOrEqual(2)
    for (const service of new Set(plans.map((plan) => plan.service))) {
      expect(plans.filter((plan) => plan.service === service)
        .reduce((total, plan) => total + plan.vehicleCount, 0)).toBe(1)
    }
  })

  it('derives first-day repair motion from measured post-shock-to-end recovery', () => {
    const plans = repairPlansForDay(trajectoryDay({
      day: 1,
      services_after_shock: [0.3, 0.35, 0.4, 0.45, 0.5],
      services_end: [0.31, 0.35, 0.4, 0.45, 0.5],
      gain: [0.01, 0, 0, 0, 0],
    }), undefined)

    expect(new Set(plans.map((plan) => plan.service))).toEqual(new Set(['transport']))
  })

  it('requires positive recorded repair supply before rendering schema-v3 repair work', () => {
    const previous = trajectoryDay({
      day: 3,
      services_end: [0.4, 0.4, 0.4, 0.4, 0.4],
    })
    const withoutSupply = trajectoryDay({
      day: 4,
      services_end: [0.43, 0.4, 0.4, 0.4, 0.4],
      logistics: { repair_supply: [0, 12, 12, 12, 12] } as DayResult['logistics'],
    })
    const withSupply = trajectoryDay({
      ...withoutSupply,
      logistics: { repair_supply: [6.5, 12, 12, 12, 12] } as DayResult['logistics'],
    })

    expect(hasCurrentDayRepairWork(withoutSupply, previous, 'transport')).toBe(false)
    expect(repairPlansForDay(withoutSupply, previous).some((plan) => plan.service === 'transport')).toBe(false)
    expect(hasCurrentDayRepairWork(withSupply, previous, 'transport')).toBe(true)
    expect(repairPlansForDay(withSupply, previous).some((plan) => plan.service === 'transport')).toBe(true)
  })

  it('retains the released service-gain presentation for legacy days without a depot ledger', () => {
    const previous = trajectoryDay({ day: 3, services_end: [0.4, 0.4, 0.4, 0.4, 0.4] })
    const current = trajectoryDay({ day: 4, services_end: [0.42, 0.4, 0.4, 0.4, 0.4] })

    expect(hasCurrentDayRepairWork(current, previous, 'transport')).toBe(true)
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

  it('freezes the prior returned repair site through impact and assessment', () => {
    expect(repairPresentationCursor(4, 0, false)).toEqual({ dayIndex: 3, progress: 1 })
    expect(repairPresentationCursor(4, 0.18, false)).toEqual({ dayIndex: 3, progress: 1 })
    expect(repairPresentationCursor(4, 0.359, false)).toEqual({ dayIndex: 3, progress: 1 })
    expect(repairPresentationCursor(4, 0, true)).toEqual({ dayIndex: 4, progress: 0 })
    expect(repairPresentationCursor(4, 0.42, true)).toEqual({ dayIndex: 4, progress: 0.42 })
    expect(repairPresentationCursor(0, 0.18, false)).toBeNull()
    expect(repairActivityMotionTime(4, 999, false)).toBe(repairActivityMotionTime(4, 123, false))
    expect(repairActivityMotionTime(4, 9.5, true)).toBe(9.5)
  })

})
