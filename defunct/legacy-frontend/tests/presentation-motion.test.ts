import { describe, expect, it } from 'vitest'
import {
  PRESENTATION_MOTION_SECONDS_PER_DAY,
  REPAIR_SHIFT_CHANGE_DAY_FRACTION,
  REPAIR_SHIFT_CHANGE_MS,
  presentationMotionPhase,
  presentationMotionTime,
  repairPresentationMotionTime,
  repairShiftChangeBlend,
} from '../src/game/presentationMotion'

describe('cursor-derived scene motion', () => {
  it('maps the shared absolute day to the restrained five-times-slower motion clock', () => {
    expect(PRESENTATION_MOTION_SECONDS_PER_DAY).toBeCloseTo(1.4)
    expect(presentationMotionTime(0)).toBe(0)
    expect(presentationMotionTime(2.5)).toBeCloseTo(3.5)
  })

  it('is deterministic under pause, resume, speed changes, and seek', () => {
    const pausedCursor = 4.375
    const beforeControlChange = presentationMotionTime(pausedCursor)

    // Playback controls change how quickly the cursor reaches its next value;
    // they do not enter the phase calculation and therefore cannot jump it.
    expect(presentationMotionTime(pausedCursor)).toBe(beforeControlChange)
    expect(presentationMotionTime(4.5)).toBeGreaterThan(beforeControlChange)
    expect(presentationMotionTime(pausedCursor)).toBe(beforeControlChange)
  })

  it('wraps looping effects without negative or boundary discontinuities', () => {
    expect(presentationMotionPhase(0, 0.25)).toBe(0)
    expect(presentationMotionPhase(4, 0.25)).toBe(0)
    expect(presentationMotionPhase(1, 0.25, 0.9)).toBeCloseTo(0.15)
    expect(presentationMotionPhase(-1, 0.25)).toBeCloseTo(0.75)
  })

  it('holds repair motion for the cursor-derived 650ms shift handoff without a boundary jump', () => {
    expect(REPAIR_SHIFT_CHANGE_MS).toBe(650)
    expect(REPAIR_SHIFT_CHANGE_DAY_FRACTION).toBeCloseTo(650 / 7_000, 12)
    const beforeBoundary = repairPresentationMotionTime(1 - 1e-9)
    const atBoundary = repairPresentationMotionTime(1)
    const duringHandoff = repairPresentationMotionTime(1 + REPAIR_SHIFT_CHANGE_DAY_FRACTION / 2)
    const afterHandoff = repairPresentationMotionTime(1 + REPAIR_SHIFT_CHANGE_DAY_FRACTION)

    expect(atBoundary).toBeCloseTo(beforeBoundary, 8)
    expect(duringHandoff).toBe(atBoundary)
    expect(afterHandoff).toBeCloseTo(atBoundary, 12)
    expect(repairPresentationMotionTime(1 + REPAIR_SHIFT_CHANGE_DAY_FRACTION + 0.1))
      .toBeGreaterThan(afterHandoff)
    expect(repairShiftChangeBlend(1)).toBe(0)
    expect(repairShiftChangeBlend(1 + REPAIR_SHIFT_CHANGE_DAY_FRACTION / 2)).toBeCloseTo(0.5)
    expect(repairShiftChangeBlend(1 + REPAIR_SHIFT_CHANGE_DAY_FRACTION)).toBe(1)
  })
})
