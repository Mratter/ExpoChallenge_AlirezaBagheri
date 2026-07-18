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

export const CITY_CAMERA_TARGET_Y = 0.35
const PLATE_HALF_WIDTH = 12.6
const PLATE_HALF_DEPTH = 11.6
const PLATE_MIN_Y = -0.62
const PLATE_MAX_Y = 0.16

const PLATE_CORNERS = [-PLATE_HALF_WIDTH, PLATE_HALF_WIDTH].flatMap((x) =>
  [PLATE_MIN_Y, PLATE_MAX_Y].flatMap((y) =>
    [-PLATE_HALF_DEPTH, PLATE_HALF_DEPTH].map((z) => [x, y - CITY_CAMERA_TARGET_Y, z] as const),
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
  return requiredDistance(verticalFov, aspect, polarAngle, azimuthAngle) * 1.045 + 0.35
}

/** Applies the pose-specific containment floor without sacrificing closer safe views. */
export function cityCameraContainedDistance(
  requestedDistance: number,
  input: OrbitContainmentInput,
): number {
  return Math.max(requestedDistance, cityCameraRequiredDistance(input))
}

/**
 * Samples the complete orbit deterministically. Portrait uses the fully-contained
 * distance; landscape keeps a closer preferred composition while CityCamera applies
 * the exact pose-specific containment floor during orbit and zoom.
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
  const compositionDistance = 40 + Math.max(0, 1 - aspect) * 28
  const minDistance = Math.min(fullyContainedDistance, compositionDistance)
  const maxDistance = Math.max(
    minDistance + 14,
    minDistance * 1.42,
    fullyContainedDistance + 8,
  )
  return {
    minDistance,
    maxDistance,
    far: maxDistance + 42,
  }
}
