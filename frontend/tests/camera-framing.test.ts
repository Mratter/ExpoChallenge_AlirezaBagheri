import { describe, expect, it } from 'vitest'
import {
  cityCameraContainedDistance,
  cityCameraFraming,
  cityCameraRequiredDistance,
  cityCameraVerticalFov,
} from '../src/game/cameraFraming'

const orbit = {
  minPolarAngle: 0.42,
  maxPolarAngle: 1.25,
}

describe('cityCameraFraming', () => {
  it('keeps the portrait orbit farther out than the landscape orbit', () => {
    const landscape = cityCameraFraming({
      width: 1440,
      height: 900,
      verticalFovDegrees: cityCameraVerticalFov(1440, 900),
      ...orbit,
    })
    const portrait = cityCameraFraming({
      width: 390,
      height: 844,
      verticalFovDegrees: cityCameraVerticalFov(390, 844),
      ...orbit,
    })

    expect(portrait.minDistance).toBeGreaterThan(landscape.minDistance)
    expect(landscape.minDistance).toBeGreaterThan(35)
    expect(landscape.minDistance).toBeLessThan(45)
    expect(portrait.minDistance).toBeGreaterThan(40)
    expect(portrait.minDistance).toBeLessThan(55)
  })

  it('widens portrait coverage while keeping landscape perspective restrained', () => {
    expect(cityCameraVerticalFov(1440, 900)).toBe(40)
    expect(cityCameraVerticalFov(390, 844)).toBeGreaterThan(74)
    expect(cityCameraVerticalFov(768, 1024)).toBeGreaterThan(40)
  })

  it('returns usable zoom and clipping ranges for every viewport', () => {
    for (const [width, height] of [[1440, 900], [390, 844], [0, 0]]) {
      const framing = cityCameraFraming({
        width,
        height,
        verticalFovDegrees: cityCameraVerticalFov(width, height),
        ...orbit,
      })
      expect(Number.isFinite(framing.minDistance)).toBe(true)
      expect(framing.maxDistance).toBeGreaterThan(framing.minDistance)
      expect(framing.far).toBeGreaterThan(framing.maxDistance)
    }
  })

  it('enforces plate-corner containment at every sampled orbit pose', () => {
    for (const [width, height] of [[1440, 900], [390, 844]]) {
      const verticalFovDegrees = cityCameraVerticalFov(width, height)
      const framing = cityCameraFraming({ width, height, verticalFovDegrees, ...orbit })
      for (let polarIndex = 0; polarIndex <= 12; polarIndex += 1) {
        const polarAngle = orbit.minPolarAngle
          + (orbit.maxPolarAngle - orbit.minPolarAngle) * polarIndex / 12
        for (let azimuthIndex = 0; azimuthIndex < 32; azimuthIndex += 1) {
          const azimuthAngle = azimuthIndex / 32 * Math.PI * 2
          const input = { width, height, verticalFovDegrees, polarAngle, azimuthAngle }
          const required = cityCameraRequiredDistance(input)
          const contained = cityCameraContainedDistance(framing.minDistance, input)
          expect(contained).toBeGreaterThanOrEqual(required)
          expect(framing.maxDistance).toBeGreaterThan(required)
        }
      }
    }
  })
})
