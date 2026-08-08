import { describe, expect, it, vi } from 'vitest'
import {
  createPresentationClock,
  MAX_PRESENTATION_FRAME_DELTA_MS,
  type PresentationFrameCallback,
  type PresentationFrameScheduler,
} from '../src/game/presentationClock'

class ManualFrameScheduler implements PresentationFrameScheduler {
  private nextHandle = 1
  private callbacks = new Map<number, PresentationFrameCallback>()

  request(callback: PresentationFrameCallback): number {
    const handle = this.nextHandle
    this.nextHandle += 1
    this.callbacks.set(handle, callback)
    return handle
  }

  cancel(handle: number): void {
    this.callbacks.delete(handle)
  }

  frame(timestampMs: number): void {
    const callbacks = [...this.callbacks.values()]
    this.callbacks.clear()
    callbacks.forEach((callback) => callback(timestampMs))
  }

  get pending(): number {
    return this.callbacks.size
  }
}

describe('deterministic requestAnimationFrame presentation clock', () => {
  it('publishes a cached useSyncExternalStore snapshot and advances on one frame loop', () => {
    let nowMs = 0
    const scheduler = new ManualFrameScheduler()
    const clock = createPresentationClock({
      dayCount: 3,
      now: () => nowMs,
      scheduler,
      maxFrameDeltaMs: 1_000_000,
    })
    const listener = vi.fn()
    const unsubscribe = clock.subscribe(listener)
    const initial = clock.getSnapshot()

    expect(clock.getSnapshot()).toBe(initial)
    expect(scheduler.pending).toBe(1)
    nowMs = 3_500
    scheduler.frame(nowMs)
    expect(clock.getSnapshot()).not.toBe(initial)
    expect(clock.getSnapshot()).toMatchObject({ dayIndex: 0, progress: 0.5, terminal: false })
    expect(clock.getSnapshot()).toBe(clock.getSnapshot())
    expect(listener).toHaveBeenCalledTimes(1)
    expect(scheduler.pending).toBe(1)

    unsubscribe()
    clock.destroy()
    expect(scheduler.pending).toBe(0)
  })

  it('honors the seven-second base day, speed, and fourfold aiming slowdown without resetting progress', () => {
    let nowMs = 0
    const scheduler = new ManualFrameScheduler()
    const clock = createPresentationClock({ dayCount: 4, now: () => nowMs, scheduler, maxFrameDeltaMs: 1_000_000 })

    nowMs = 1_750
    scheduler.frame(nowMs)
    expect(clock.getSnapshot().progress).toBe(0.25)

    // A control change between rendered frames first accounts for elapsed time
    // at the old rate, then preserves that exact fractional cursor.
    nowMs += 1_750
    clock.setSpeed(2)
    expect(clock.getSnapshot().progress).toBe(0.5)
    nowMs += 875
    scheduler.frame(nowMs)
    expect(clock.getSnapshot()).toMatchObject({ dayIndex: 0, progress: 0.75, speed: 2 })

    clock.setAiming(true)
    const beforeAim = clock.getSnapshot().progress
    expect(beforeAim).toBe(0.75)
    nowMs += 1_750
    scheduler.frame(nowMs)
    expect(clock.getSnapshot().progress).toBeCloseTo(0.875, 12)
    expect(clock.getSnapshot().aiming).toBe(true)
    clock.destroy()
  })

  it('freezes exactly while paused or blocked and resumes from the same fractional cursor', () => {
    let nowMs = 0
    const scheduler = new ManualFrameScheduler()
    const clock = createPresentationClock({ dayCount: 3, now: () => nowMs, scheduler, maxFrameDeltaMs: 1_000_000 })

    nowMs = 2_100
    scheduler.frame(nowMs)
    expect(clock.getSnapshot().progress).toBeCloseTo(0.3)
    clock.setPaused(true)
    const pausedAt = clock.getSnapshot().progress
    expect(scheduler.pending).toBe(0)
    nowMs += 20_000
    scheduler.frame(nowMs)
    expect(clock.getSnapshot().progress).toBe(pausedAt)

    clock.setPaused(false)
    expect(scheduler.pending).toBe(1)
    nowMs += 700
    scheduler.frame(nowMs)
    expect(clock.getSnapshot().progress).toBeCloseTo(0.4)

    clock.setBlocked(true)
    const blockedAt = clock.getSnapshot().progress
    expect(scheduler.pending).toBe(0)
    nowMs += 20_000
    clock.setSpeed(2)
    expect(clock.getSnapshot().progress).toBe(blockedAt)
    clock.setBlocked(false)
    nowMs += 350
    scheduler.frame(nowMs)
    expect(clock.getSnapshot().progress).toBeCloseTo(0.5)
    clock.destroy()
  })

  it('fires every crossed day boundary and terminal callback exactly once', () => {
    let nowMs = 0
    const scheduler = new ManualFrameScheduler()
    const onDayChange = vi.fn()
    const onTerminal = vi.fn()
    const clock = createPresentationClock({
      dayCount: 3,
      now: () => nowMs,
      scheduler,
      onDayChange,
      onTerminal,
      maxFrameDeltaMs: 1_000_000,
    })

    nowMs = 21_000
    scheduler.frame(nowMs)
    expect(clock.getSnapshot()).toMatchObject({
      dayIndex: 2,
      dayCount: 3,
      progress: 1,
      absoluteDay: 3,
      terminal: true,
    })
    expect(onDayChange.mock.calls.map(([event]) => event)).toEqual([
      { completedDayIndex: 0, fromDayIndex: 0, toDayIndex: 1 },
      { completedDayIndex: 1, fromDayIndex: 1, toDayIndex: 2 },
    ])
    expect(onTerminal).toHaveBeenCalledOnce()
    expect(onTerminal).toHaveBeenCalledWith({ dayIndex: 2, dayCount: 3 })
    expect(scheduler.pending).toBe(0)

    clock.setPaused(true)
    clock.setPaused(false)
    clock.setSpeed(2)
    scheduler.frame(nowMs + 50_000)
    expect(onDayChange).toHaveBeenCalledTimes(2)
    expect(onTerminal).toHaveBeenCalledTimes(1)
    clock.destroy()
  })

  it('seeks and replaces a run without callbacks while preserving a canonical fractional cursor', () => {
    let nowMs = 0
    const scheduler = new ManualFrameScheduler()
    const onDayChange = vi.fn()
    const onTerminal = vi.fn()
    const clock = createPresentationClock({
      dayCount: 5,
      initialPaused: true,
      now: () => nowMs,
      scheduler,
      onDayChange,
      onTerminal,
      maxFrameDeltaMs: 1_000_000,
    })

    clock.seek({ dayIndex: 1, progress: 0.4 })
    expect(clock.getSnapshot()).toMatchObject({ dayIndex: 1, absoluteDay: 1.4 })
    expect(clock.getSnapshot().progress).toBeCloseTo(0.4, 12)
    clock.replaceRun(8)
    expect(clock.getSnapshot()).toMatchObject({ dayIndex: 1, absoluteDay: 1.4, dayCount: 8 })
    expect(clock.getSnapshot().progress).toBeCloseTo(0.4, 12)

    clock.seek({ dayIndex: 1, progress: 1 })
    expect(clock.getSnapshot()).toMatchObject({ dayIndex: 2, progress: 0, absoluteDay: 2 })
    clock.replaceRun(1)
    expect(clock.getSnapshot()).toMatchObject({ dayIndex: 0, progress: 1, absoluteDay: 1, terminal: true })
    clock.replaceRun(5)
    expect(clock.getSnapshot()).toMatchObject({ dayIndex: 1, progress: 0, absoluteDay: 1, terminal: false })
    expect(onDayChange).not.toHaveBeenCalled()
    expect(onTerminal).not.toHaveBeenCalled()
    clock.destroy()
  })

  it('supports callback replacement and one new terminal event after seeking backward', () => {
    let nowMs = 0
    const scheduler = new ManualFrameScheduler()
    const oldTerminal = vi.fn()
    const nextTerminal = vi.fn()
    const clock = createPresentationClock({
      dayCount: 1,
      now: () => nowMs,
      scheduler,
      onTerminal: oldTerminal,
      maxFrameDeltaMs: 1_000_000,
    })
    clock.setCallbacks({ onTerminal: nextTerminal })
    nowMs = 7_000
    scheduler.frame(nowMs)
    expect(oldTerminal).not.toHaveBeenCalled()
    expect(nextTerminal).toHaveBeenCalledOnce()

    clock.seek({ dayIndex: 0, progress: 0.5 })
    nowMs += 3_500
    scheduler.frame(nowMs)
    expect(nextTerminal).toHaveBeenCalledTimes(2)
    clock.destroy()
  })

  it('discards suspended-tab wall time instead of skipping returned days', () => {
    let nowMs = 0
    const scheduler = new ManualFrameScheduler()
    const onDayChange = vi.fn()
    const clock = createPresentationClock({
      dayCount: 20,
      now: () => nowMs,
      scheduler,
      onDayChange,
    })

    nowMs = 90_000
    scheduler.frame(nowMs)
    expect(clock.getSnapshot().dayIndex).toBe(0)
    expect(clock.getSnapshot().progress).toBeCloseTo(MAX_PRESENTATION_FRAME_DELTA_MS / 7_000, 12)
    expect(onDayChange).not.toHaveBeenCalled()

    nowMs += 16
    scheduler.frame(nowMs)
    expect(clock.getSnapshot().progress).toBeCloseTo((MAX_PRESENTATION_FRAME_DELTA_MS + 16) / 7_000, 12)
    clock.destroy()
  })
})
