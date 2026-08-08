import { describe, expect, it } from 'vitest'
import {
  AIMING_DAY_DURATION_MULTIPLIER,
  CONTINUOUS_MOTION_BASE_RATE,
  CONTINUOUS_MOTION_SLOWDOWN,
  GAME_DAY_DURATION_MS,
  GAME_HORIZON_DAYS,
  PLAYBACK_SPEEDS,
  gameDayDurationMs,
} from '../src/game/pacing'

describe('game presentation pacing', () => {
  it('uses the deliberate three-rate playback contract', () => {
    expect(PLAYBACK_SPEEDS).toEqual([0.5, 1, 2])
    expect(gameDayDurationMs(0.5)).toBe(14_000)
    expect(gameDayDurationMs(1)).toBe(GAME_DAY_DURATION_MS)
    expect(gameDayDurationMs(2)).toBe(3_500)
  })

  it('slows an aimed day without changing the selected playback rate', () => {
    expect(AIMING_DAY_DURATION_MULTIPLIER).toBe(4)
    expect(gameDayDurationMs(1, true)).toBe(28_000)
    expect(gameDayDurationMs(2, true)).toBe(14_000)
  })

  it('makes the standard uninterrupted horizon several minutes long', () => {
    expect(GAME_HORIZON_DAYS).toBe(20)
    expect(GAME_HORIZON_DAYS * gameDayDurationMs(1)).toBe(140_000)
  })

  it('pins the fivefold continuous-motion scale used by the shared cursor', () => {
    expect(CONTINUOUS_MOTION_SLOWDOWN).toBe(5)
    expect(CONTINUOUS_MOTION_BASE_RATE).toBe(0.2)
  })
})
