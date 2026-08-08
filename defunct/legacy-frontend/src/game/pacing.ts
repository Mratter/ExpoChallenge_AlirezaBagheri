export const GAME_DAY_DURATION_MS = 7_000
export const GAME_HORIZON_DAYS = 20

export const PLAYBACK_SPEEDS = [0.5, 1, 2] as const
export type PlaybackSpeed = (typeof PLAYBACK_SPEEDS)[number]

export const AIMING_DAY_DURATION_MULTIPLIER = 4
export const CONTINUOUS_MOTION_SLOWDOWN = 5
export const CONTINUOUS_MOTION_BASE_RATE = 1 / CONTINUOUS_MOTION_SLOWDOWN

/** Real-time duration of one presented simulation day at the selected playback rate. */
export function gameDayDurationMs(
  speed: PlaybackSpeed,
  aiming = false,
): number {
  return (GAME_DAY_DURATION_MS / speed)
    * (aiming ? AIMING_DAY_DURATION_MULTIPLIER : 1)
}
