import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import {
  cityCameraContainedDistance,
  cityCameraFraming,
  cityCameraRequiredDistance,
  cityCameraVerticalFov,
} from '../src/game/cameraFraming'
import {
  CITY_CAMERA_LAYOUT,
  CITY_CAMERA_TARGET_Y,
  CITY_PLATE_CAMERA_BOUNDS,
} from '../src/game/worldLayout'

const orbit = {
  minPolarAngle: 0.42,
  maxPolarAngle: 1.25,
}

function projectedPlateBounds(width: number, height: number) {
  const fov = cityCameraVerticalFov(width, height)
  const framing = cityCameraFraming({
    width,
    height,
    verticalFovDegrees: fov,
    ...orbit,
  })
  const target = new THREE.Vector3(0, CITY_CAMERA_TARGET_Y, 0)
  const direction = new THREE.Vector3(...CITY_CAMERA_LAYOUT.defaultPosition).sub(target)
  const spherical = new THREE.Spherical().setFromVector3(direction)
  if (width < height) {
    spherical.phi = CITY_CAMERA_LAYOUT.portraitDefaultPolarAngle
    spherical.theta = CITY_CAMERA_LAYOUT.portraitDefaultAzimuthAngle
    direction.setFromSpherical(spherical)
  }
  const containmentInput = {
    width,
    height,
    verticalFovDegrees: fov,
    polarAngle: spherical.phi,
    azimuthAngle: spherical.theta,
  }
  const distance = cityCameraContainedDistance(framing.minDistance, containmentInput) * 1.035
  const camera = new THREE.PerspectiveCamera(fov, width / height, 0.1, framing.far)
  camera.position.copy(target).add(direction.normalize().multiplyScalar(distance))
  camera.lookAt(target)
  camera.updateProjectionMatrix()
  camera.updateMatrixWorld()

  const points = [-CITY_PLATE_CAMERA_BOUNDS.halfWidth, CITY_PLATE_CAMERA_BOUNDS.halfWidth].flatMap((x) =>
    [CITY_PLATE_CAMERA_BOUNDS.minY, CITY_PLATE_CAMERA_BOUNDS.maxY].flatMap((y) =>
      [-CITY_PLATE_CAMERA_BOUNDS.halfDepth, CITY_PLATE_CAMERA_BOUNDS.halfDepth].map((z) => (
        new THREE.Vector3(x, y, z).project(camera)
      )),
    ),
  )
  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  }
}

describe('cityCameraFraming', () => {
  it('gives portrait an intentionally closer aligned composition than landscape', () => {
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

    expect(portrait.minDistance).toBeLessThan(landscape.minDistance)
    expect(landscape.minDistance).toBeGreaterThan(80)
    expect(landscape.minDistance).toBeLessThan(100)
    expect(portrait.minDistance).toBeGreaterThan(70)
    expect(portrait.minDistance).toBeLessThan(85)
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

  it('keeps the enlarged plate fully visible and visually dominant at both target viewports', () => {
    const landscape = projectedPlateBounds(1440, 900)
    const portrait = projectedPlateBounds(390, 844)

    for (const bounds of [landscape, portrait]) {
      expect(bounds.minX).toBeGreaterThanOrEqual(-0.96)
      expect(bounds.maxX).toBeLessThanOrEqual(0.96)
      expect(bounds.minY).toBeGreaterThanOrEqual(-0.96)
      expect(bounds.maxY).toBeLessThanOrEqual(0.96)
    }

    expect(landscape.maxX - landscape.minX).toBeGreaterThan(1.3)
    expect(landscape.maxY - landscape.minY).toBeGreaterThan(1.2)
    expect(portrait.maxX - portrait.minX).toBeGreaterThan(1.7)
    expect(portrait.maxY - portrait.minY).toBeGreaterThan(0.52)
  })
})
