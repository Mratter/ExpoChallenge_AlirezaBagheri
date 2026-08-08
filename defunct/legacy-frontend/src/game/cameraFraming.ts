import {
  CITY_CAMERA_LAYOUT,
  CITY_CAMERA_TARGET_Y,
  CITY_PLATE_CAMERA_BOUNDS,
} from './worldLayout'

export { CITY_CAMERA_TARGET_Y } from './worldLayout'

export type CityCameraFraming = {
  minDistance: number
  maxDistance: number
  far: number
}

/**
 * Keeps the plate legible in landscape while widening portrait coverage instead
 * of pushing the camera so far away that the city becomes a thumbnail.
 */
export function cityCameraVerticalFov(width: number, height: number): number {
  const aspect = Math.max(width, 1) / Math.max(height, 1)
  if (aspect >= 1) return 40
  return Math.min(75, 40 + (1 - aspect) * 65)
}

type ViewportFramingInput = {
  width: number
  height: number
  verticalFovDegrees: number
  minPolarAngle: number
  maxPolarAngle: number
}

type OrbitContainmentInput = {
  width: number
  height: number
  verticalFovDegrees: number
  polarAngle: number
  azimuthAngle: number
}

const PLATE_CORNERS = [-CITY_PLATE_CAMERA_BOUNDS.halfWidth, CITY_PLATE_CAMERA_BOUNDS.halfWidth].flatMap((x) =>
  [CITY_PLATE_CAMERA_BOUNDS.minY, CITY_PLATE_CAMERA_BOUNDS.maxY].flatMap((y) =>
    [-CITY_PLATE_CAMERA_BOUNDS.halfDepth, CITY_PLATE_CAMERA_BOUNDS.halfDepth].map((z) => [x, y - CITY_CAMERA_TARGET_Y, z] as const),
  ),
)

function requiredDistance(
  verticalFov: number,
  aspect: number,
  polar: number,
  azimuth: number,
): number {
  const tanVertical = Math.tan(verticalFov / 2)
  const tanHorizontal = tanVertical * aspect
  const sinPolar = Math.sin(polar)
  const cosPolar = Math.cos(polar)
  const sinAzimuth = Math.sin(azimuth)
  const cosAzimuth = Math.cos(azimuth)

  // Camera basis for an orbit position around [0, CAMERA_TARGET_Y, 0].
  const forward = [-sinPolar * sinAzimuth, -cosPolar, -sinPolar * cosAzimuth] as const
  const right = [cosAzimuth, 0, -sinAzimuth] as const
  const up = [-cosPolar * sinAzimuth, sinPolar, -cosPolar * cosAzimuth] as const

  return PLATE_CORNERS.reduce((distance, point) => {
    const depthOffset = point[0] * forward[0] + point[1] * forward[1] + point[2] * forward[2]
    const horizontal = Math.abs(point[0] * right[0] + point[2] * right[2])
    const vertical = Math.abs(point[0] * up[0] + point[1] * up[1] + point[2] * up[2])
    return Math.max(
      distance,
      horizontal / tanHorizontal - depthOffset,
      vertical / tanVertical - depthOffset,
    )
  }, 0)
}

/** Minimum camera distance that contains every plate corner at one orbit pose. */
export function cityCameraRequiredDistance({
  width,
  height,
  verticalFovDegrees,
  polarAngle,
  azimuthAngle,
}: OrbitContainmentInput): number {
  const aspect = Math.max(width, 1) / Math.max(height, 1)
  const verticalFov = verticalFovDegrees * Math.PI / 180
  return requiredDistance(verticalFov, aspect, polarAngle, azimuthAngle) * 1.045
    + CITY_CAMERA_LAYOUT.containmentPadding
}

/** Applies the pose-specific containment floor without sacrificing closer safe views. */
export function cityCameraContainedDistance(
  requestedDistance: number,
  input: OrbitContainmentInput,
): number {
  return Math.max(requestedDistance, cityCameraRequiredDistance(input))
}

/**
 * Samples the complete orbit deterministically. Each viewport starts at an intentional
 * composition distance while CityCamera applies the exact pose-specific containment
 * floor during orbit and zoom.
 */
export function cityCameraFraming({
  width,
  height,
  verticalFovDegrees,
  minPolarAngle,
  maxPolarAngle,
}: ViewportFramingInput): CityCameraFraming {
  const aspect = Math.max(width, 1) / Math.max(height, 1)
  let required = 0
  const polarSamples = 32
  const azimuthSamples = 96

  for (let polarIndex = 0; polarIndex <= polarSamples; polarIndex += 1) {
    const progress = polarIndex / polarSamples
    const polar = minPolarAngle + (maxPolarAngle - minPolarAngle) * progress
    for (let azimuthIndex = 0; azimuthIndex < azimuthSamples; azimuthIndex += 1) {
      const azimuth = (azimuthIndex / azimuthSamples) * Math.PI * 2
      required = Math.max(required, cityCameraRequiredDistance({
        width,
        height,
        verticalFovDegrees,
        polarAngle: polar,
        azimuthAngle: azimuth,
      }))
    }
  }

  const fullyContainedDistance = required
  const compositionDistance = aspect >= 1
    ? CITY_CAMERA_LAYOUT.landscapeCompositionDistance
    : CITY_CAMERA_LAYOUT.portraitCompositionDistance
  const minDistance = Math.min(fullyContainedDistance, compositionDistance)
  const maxDistance = Math.max(
    minDistance + CITY_CAMERA_LAYOUT.minimumZoomSpan,
    minDistance * 1.42,
    fullyContainedDistance + CITY_CAMERA_LAYOUT.orbitClearance,
  )
  return {
    minDistance,
    maxDistance,
    far: maxDistance + CITY_CAMERA_LAYOUT.farClearance,
  }
}
