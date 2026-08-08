import { describe, expect, it } from 'vitest'
import {
  DISASTER_EFFECT_KINDS,
  DISASTER_PRESENTATION_SECONDS_PER_DAY,
  DISASTER_STRIKE_SETTLE_END_FRACTION,
  disasterEffectPlan,
  disasterMagnitude,
  disasterStrikeTimeSeconds,
  disasterStrikeSettleVisibility,
  disasterTelegraphTimeSeconds,
  disasterUnitNoise,
  earthquakeCameraOffset,
  earthquakeForeshockOffset,
  type DisasterEffectEvent,
} from '../src/game/DisasterEffects'
import type { Service, ShockType } from '../src/types'

const RESPONSE_ORDER: Service[] = ['food', 'public_services', 'transport', 'healthcare', 'housing']

function event(overrides: Partial<DisasterEffectEvent> = {}): DisasterEffectEvent {
  return {
    id: 'result:day-7',
    type: 'weather',
    severity: 0.28,
    day: 7,
    point: [-4, 0.24, 5],
    service: 'food',
    // response order: food, civic, transport, healthcare, housing
    impact: [0.5, 0.6, 0.75, 0.4, 0.55],
    ...overrides,
  }
}

describe('deterministic typed disaster effects', () => {
  it('maps the returned footprint by the returned service order', () => {
    const plan = disasterEffectPlan(event(), RESPONSE_ORDER)
    const byService = Object.fromEntries(plan.districts.map((district) => [district.service, district]))

    expect(byService.transport.coefficient).toBe(0.75)
    expect(byService.housing.coefficient).toBe(0.55)
    expect(byService.food.coefficient).toBe(0.5)
    expect(byService.healthcare.coefficient).toBe(0.4)
    expect(byService.public_services.coefficient).toBe(0.6)
    expect(byService.transport.magnitude).toBeCloseTo(0.28 * 0.75)
    expect(plan.strongestService).toBe('transport')
    expect(plan.targetService).toBe('food')
  })

  it('does not silently fall back to canonical offsets for a missing response service', () => {
    const services: Service[] = ['housing', 'food', 'healthcare', 'public_services']
    const plan = disasterEffectPlan(event({ impact: [0.55, 0.5, 0.4, 0.6] }), services)

    expect(plan.districts.find((district) => district.service === 'transport')).toMatchObject({
      coefficient: 0,
      magnitude: 0,
      normalized: 0,
    })
  })

  it('gives every frozen raw key a distinct restrained arrival plan', () => {
    const keys = Object.keys(DISASTER_EFFECT_KINDS) as ShockType[]
    expect(keys).toEqual(['aftershock', 'supply', 'epidemic', 'utility', 'weather'])
    expect(new Set(keys.map((type) => disasterEffectPlan(event({ type }), RESPONSE_ORDER).kind)).size).toBe(5)
  })

  it('scales footprint magnitude monotonically from returned severity and coefficient', () => {
    expect(disasterMagnitude(0.2, 0.75)).toBeCloseTo(0.15)
    expect(disasterMagnitude(0.3, 0.75)).toBeGreaterThan(disasterMagnitude(0.2, 0.75))
    expect(disasterMagnitude(0.3, 1)).toBeGreaterThan(disasterMagnitude(0.3, 0.4))
    expect(disasterMagnitude(9, 9)).toBe(0.4)
  })

  it('uses stable procedural noise without runtime randomness', () => {
    const first = Array.from({ length: 8 }, (_, index) => disasterUnitNoise(417, index, 3))
    const repeat = Array.from({ length: 8 }, (_, index) => disasterUnitNoise(417, index, 3))
    const changed = Array.from({ length: 8 }, (_, index) => disasterUnitNoise(418, index, 3))

    expect(repeat).toEqual(first)
    expect(changed).not.toEqual(first)
    expect(first.every((value) => value >= 0 && value <= 1)).toBe(true)
  })

  it('shakes the camera only for Earthquake presentation and never under reduced motion', () => {
    const earthquake = earthquakeCameraOffset('aftershock', 0.36, 0.17, 701, false)
    expect(Math.hypot(...earthquake)).toBeGreaterThan(0)
    expect(earthquakeCameraOffset('aftershock', 0.36, 2, 701, false)).toEqual([0, 0])
    expect(earthquakeCameraOffset('aftershock', 0.36, 0.17, 701, true)).toEqual([0, 0])

    ;(['supply', 'epidemic', 'utility', 'weather'] as ShockType[]).forEach((type) => {
      expect(earthquakeCameraOffset(type, 0.4, 0.17, 701, false)).toEqual([0, 0])
    })
  })

  it('telegraphs Earthquake with brief deterministic micro-tremors', () => {
    expect(Math.hypot(...earthquakeForeshockOffset(0.3, 0.05, 701, false))).toBeGreaterThan(0)
    expect(earthquakeForeshockOffset(0.3, 0.5, 701, false)).toEqual([0, 0])
    expect(earthquakeForeshockOffset(0.3, 2.65, 701, true)).toEqual([0, 0])
    expect(earthquakeForeshockOffset(0.3, 0.05, 701, false)).toEqual(
      earthquakeForeshockOffset(0.3, 0.05, 701, false),
    )
  })

  it('samples telegraph motion only from the shared current-day cursor', () => {
    expect(disasterTelegraphTimeSeconds(7, 5)).toBe(0)
    expect(disasterTelegraphTimeSeconds(7, 5.25)).toBeCloseTo(1.75)
    expect(disasterTelegraphTimeSeconds(7, 5.25)).toBe(disasterTelegraphTimeSeconds(7, 5.25))
    expect(disasterTelegraphTimeSeconds(7, 5.75)).toBeCloseTo(5.25)
    expect(disasterTelegraphTimeSeconds(7, 6)).toBe(7)
    expect(disasterTelegraphTimeSeconds(7, 5.25)).toBeCloseTo(1.75)
  })

  it('anchors strike motion to the returned event-day boundary across seeks and pauses', () => {
    const boundary = 6
    const pausedSample = disasterStrikeTimeSeconds(7, boundary + 0.2)
    expect(disasterStrikeTimeSeconds(7, boundary - 0.01)).toBe(0)
    expect(disasterStrikeTimeSeconds(7, boundary)).toBe(0)
    expect(pausedSample).toBeCloseTo(1.4)
    expect(disasterStrikeTimeSeconds(7, boundary + 0.2)).toBe(pausedSample)
    expect(disasterStrikeTimeSeconds(7, boundary + 0.05)).toBeCloseTo(0.35)
    expect(disasterStrikeTimeSeconds(7, boundary + 0.2)).toBe(pausedSample)
  })

  it('holds typed overlays through assessment and dissolves them deterministically in early response', () => {
    const responseStart = DISASTER_PRESENTATION_SECONDS_PER_DAY * 0.36
    const middle = DISASTER_PRESENTATION_SECONDS_PER_DAY
      * ((0.36 + DISASTER_STRIKE_SETTLE_END_FRACTION) / 2)
    const settleEnd = DISASTER_PRESENTATION_SECONDS_PER_DAY
      * DISASTER_STRIKE_SETTLE_END_FRACTION

    expect(disasterStrikeSettleVisibility(responseStart - 0.001)).toBe(1)
    expect(disasterStrikeSettleVisibility(responseStart)).toBe(1)
    expect(disasterStrikeSettleVisibility(middle)).toBeCloseTo(0.5)
    expect(disasterStrikeSettleVisibility(settleEnd)).toBe(0)
    expect(disasterStrikeSettleVisibility(settleEnd + 4)).toBe(0)
    expect(disasterStrikeSettleVisibility(middle)).toBe(
      disasterStrikeSettleVisibility(middle),
    )
  })
})
