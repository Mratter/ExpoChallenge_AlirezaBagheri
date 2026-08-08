import { describe, expect, it } from 'vitest'
import {
  advanceAdaptiveQuality,
  applyQualityCeiling,
  createAdaptiveQualityState,
  medianFrameRate,
  renderQualityProfile,
  usesDetailedBuilding,
  viewportQualityPolicy,
} from '../src/game/renderQuality'

describe('adaptive render quality', () => {
  it('starts desktop high and caps compact portrait devices at balanced', () => {
    expect(viewportQualityPolicy({ width: 1440, height: 900, devicePixelRatio: 1 })).toEqual({
      initialTier: 'high',
      ceiling: 'high',
    })
    expect(viewportQualityPolicy({ width: 390, height: 844, devicePixelRatio: 3 })).toEqual({
      initialTier: 'balanced',
      ceiling: 'balanced',
    })
  })

  it('uses hysteresis before degrading or recovering', () => {
    let state = createAdaptiveQualityState({ initialTier: 'high', ceiling: 'high' })
    state = advanceAdaptiveQuality(state, 48)
    expect(state.tier).toBe('high')
    state = advanceAdaptiveQuality(state, 48)
    expect(state.tier).toBe('balanced')

    for (let window = 0; window < 3; window += 1) state = advanceAdaptiveQuality(state, 72)
    expect(state.tier).toBe('balanced')
    state = advanceAdaptiveQuality(state, 72)
    expect(state.tier).toBe('high')
  })

  it('enforces a new viewport ceiling without dropping essential systems', () => {
    const high = createAdaptiveQualityState({ initialTier: 'high', ceiling: 'high' })
    const compact = applyQualityCeiling(high, 'balanced')
    expect(compact.tier).toBe('balanced')
    const profile = renderQualityProfile(compact.tier, {
      width: 390,
      height: 844,
      devicePixelRatio: 3,
    })
    expect(profile).toMatchObject({
      dpr: 1.1,
      shadowMapSize: 1024,
      detailTier: 'medium',
      essentialDepots: true,
      essentialIncidents: true,
    })
  })

  it('degrades ornament detail while retaining one landmark per district', () => {
    expect(usesDetailedBuilding(0, 'low')).toBe(true)
    expect(usesDetailedBuilding(17, 'low')).toBe(false)
    expect(usesDetailedBuilding(17, 'medium')).toBe(true)
    expect(usesDetailedBuilding(35, 'medium')).toBe(false)
    expect(usesDetailedBuilding(35, 'full')).toBe(true)
  })

  it('reports median frame rate and rejects unusable samples', () => {
    expect(medianFrameRate([1 / 60, 1 / 58, 1 / 62])).toBeCloseTo(60, 5)
    expect(medianFrameRate([Number.NaN, 0, 2])).toBeNull()
  })
})
