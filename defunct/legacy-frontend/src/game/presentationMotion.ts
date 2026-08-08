import { CONTINUOUS_MOTION_BASE_RATE, GAME_DAY_DURATION_MS } from './pacing'

/**
 * Ambient world motion is a view of the presentation cursor, not a second
 * clock. One presented day advances this many animation seconds at the
 * intentionally restrained city-motion rate.
 */
export const PRESENTATION_MOTION_SECONDS_PER_DAY = (
  GAME_DAY_DURATION_MS / 1_000
) * CONTINUOUS_MOTION_BASE_RATE

/** The accepted brief crew handoff, expressed on the shared seven-second day. */
export const REPAIR_SHIFT_CHANGE_MS = 650
export const REPAIR_SHIFT_CHANGE_DAY_FRACTION = REPAIR_SHIFT_CHANGE_MS / GAME_DAY_DURATION_MS

/**
 * Converts the shared absolute presentation day into a deterministic motion
 * time. Pausing freezes it, speed changes only its slope, and seeking to the
 * same cursor always produces the same phase.
 */
export function presentationMotionTime(absoluteDay: number): number {
  if (!Number.isFinite(absoluteDay)) return 0
  return Math.max(0, absoluteDay) * PRESENTATION_MOTION_SECONDS_PER_DAY
}

/**
 * Pure work-site clock. It removes the first 650ms-equivalent slice from every
 * presented day, so cranes and lifts hold their exact boundary pose for a
 * restrained shift handoff and resume without a phase jump.
 */
export function repairPresentationMotionTime(absoluteDay: number): number {
  const boundedAbsoluteDay = Math.max(0, Number.isFinite(absoluteDay) ? absoluteDay : 0)
  const dayIndex = Math.floor(boundedAbsoluteDay)
  const dayProgress = boundedAbsoluteDay - dayIndex
  const activeDayTime = (
    dayIndex * (1 - REPAIR_SHIFT_CHANGE_DAY_FRACTION)
    + Math.max(0, dayProgress - REPAIR_SHIFT_CHANGE_DAY_FRACTION)
  )
  return activeDayTime * PRESENTATION_MOTION_SECONDS_PER_DAY
}

/** Smooth visibility handoff from yesterday's parked site to today's work plan. */
export function repairShiftChangeBlend(absoluteDay: number): number {
  const boundedAbsoluteDay = Math.max(0, Number.isFinite(absoluteDay) ? absoluteDay : 0)
  const dayProgress = boundedAbsoluteDay - Math.floor(boundedAbsoluteDay)
  const linear = Math.min(1, dayProgress / REPAIR_SHIFT_CHANGE_DAY_FRACTION)
  return linear * linear * linear * (linear * (linear * 6 - 15) + 10)
}

/** Stable wrapped phase for looping view-layer motion. */
export function presentationMotionPhase(
  motionTime: number,
  cyclesPerSecond: number,
  offset = 0,
): number {
  const raw = motionTime * cyclesPerSecond + offset
  return ((raw % 1) + 1) % 1
}
