import {
  AIMING_DAY_DURATION_MULTIPLIER,
  GAME_DAY_DURATION_MS,
  type PlaybackSpeed,
} from './pacing'
import type { PresentationCursor } from './presentation'

export type PresentationFrameCallback = (timestampMs: number) => void

export type PresentationFrameScheduler = Readonly<{
  request: (callback: PresentationFrameCallback) => number
  cancel: (handle: number) => void
}>

export type PresentationDayChange = Readonly<{
  completedDayIndex: number
  fromDayIndex: number
  toDayIndex: number
}>

export type PresentationTerminalEvent = Readonly<{
  dayIndex: number
  dayCount: number
}>

export type PresentationClockCallbacks = Readonly<{
  onDayChange?: (event: PresentationDayChange) => void
  onTerminal?: (event: PresentationTerminalEvent) => void
}>

export type PresentationClockSnapshot = Readonly<{
  dayIndex: number
  dayCount: number
  progress: number
  absoluteDay: number
  paused: boolean
  speed: PlaybackSpeed
  aiming: boolean
  blocked: boolean
  terminal: boolean
}>

export type PresentationClockOptions = PresentationClockCallbacks & Readonly<{
  dayCount: number
  initialCursor?: PresentationCursor
  initialPaused?: boolean
  initialSpeed?: PlaybackSpeed
  baseDayDurationMs?: number
  aimingDurationMultiplier?: number
  /** Test seam; production uses the shared suspension-safe frame cap. */
  maxFrameDeltaMs?: number
  now?: () => number
  scheduler?: PresentationFrameScheduler
}>

/**
 * The view does not catch up hidden-tab wall time. This matches the capped 3D
 * effect step and prevents one resumed rAF from skipping incident phases or
 * several returned days.
 */
export const MAX_PRESENTATION_FRAME_DELTA_MS = 75

export type PresentationClockStore = Readonly<{
  subscribe: (listener: () => void) => () => void
  /** Cached identity until the next actual state change; safe for useSyncExternalStore. */
  getSnapshot: () => PresentationClockSnapshot
  setPaused: (paused: boolean) => void
  setSpeed: (speed: PlaybackSpeed) => void
  setAiming: (aiming: boolean) => void
  setBlocked: (blocked: boolean) => void
  seek: (cursor: PresentationCursor) => void
  /** Replaces the run length while preserving and clamping the fractional cursor. */
  replaceRun: (dayCount: number) => void
  setCallbacks: (callbacks: PresentationClockCallbacks) => void
  destroy: () => void
}>

function validDayCount(dayCount: number): number {
  if (!Number.isFinite(dayCount)) return 1
  return Math.max(1, Math.floor(dayCount))
}

function validPositive(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value))
}

function cursorForAbsoluteDay(
  absoluteDay: number,
  dayCount: number,
): Pick<PresentationClockSnapshot, 'dayIndex' | 'progress' | 'absoluteDay' | 'terminal'> {
  const bounded = clamp(Number.isFinite(absoluteDay) ? absoluteDay : 0, 0, dayCount)
  if (bounded >= dayCount) {
    return {
      dayIndex: dayCount - 1,
      progress: 1,
      absoluteDay: dayCount,
      terminal: true,
    }
  }
  const dayIndex = Math.floor(bounded)
  const progress = bounded - dayIndex
  return { dayIndex, progress, absoluteDay: bounded, terminal: false }
}

function normalizedCursor(cursor: PresentationCursor, dayCount: number) {
  const dayIndex = Number.isFinite(cursor.dayIndex) ? Math.floor(cursor.dayIndex) : 0
  const progress = Number.isFinite(cursor.progress) ? cursor.progress : 0
  return cursorForAbsoluteDay(dayIndex + progress, dayCount)
}

function browserScheduler(): PresentationFrameScheduler {
  return {
    request: (callback) => globalThis.requestAnimationFrame(callback),
    cancel: (handle) => globalThis.cancelAnimationFrame(handle),
  }
}

function sameSnapshot(
  left: PresentationClockSnapshot,
  right: PresentationClockSnapshot,
): boolean {
  return left.dayIndex === right.dayIndex
    && left.dayCount === right.dayCount
    && left.progress === right.progress
    && left.absoluteDay === right.absoluteDay
    && left.paused === right.paused
    && left.speed === right.speed
    && left.aiming === right.aiming
    && left.blocked === right.blocked
    && left.terminal === right.terminal
}

export function createPresentationClock(
  options: PresentationClockOptions,
): PresentationClockStore {
  const scheduler = options.scheduler ?? browserScheduler()
  const now = options.now ?? (() => performance.now())
  const baseDayDurationMs = validPositive(options.baseDayDurationMs ?? GAME_DAY_DURATION_MS, GAME_DAY_DURATION_MS)
  const aimingDurationMultiplier = validPositive(
    options.aimingDurationMultiplier ?? AIMING_DAY_DURATION_MULTIPLIER,
    AIMING_DAY_DURATION_MULTIPLIER,
  )
  const maxFrameDeltaMs = validPositive(
    options.maxFrameDeltaMs ?? MAX_PRESENTATION_FRAME_DELTA_MS,
    MAX_PRESENTATION_FRAME_DELTA_MS,
  )
  const initialDayCount = validDayCount(options.dayCount)
  const initialCursor = normalizedCursor(options.initialCursor ?? { dayIndex: 0, progress: 0 }, initialDayCount)
  let snapshot: PresentationClockSnapshot = Object.freeze({
    ...initialCursor,
    dayCount: initialDayCount,
    paused: options.initialPaused ?? false,
    speed: options.initialSpeed ?? 1,
    aiming: false,
    blocked: false,
  })
  let callbacks: PresentationClockCallbacks = {
    onDayChange: options.onDayChange,
    onTerminal: options.onTerminal,
  }
  const listeners = new Set<() => void>()
  let lastTimestamp = now()
  let frameHandle: number | null = null
  let destroyed = false

  const isAdvancing = () => !snapshot.paused && !snapshot.blocked && !snapshot.terminal

  const commit = (next: PresentationClockSnapshot) => {
    if (sameSnapshot(snapshot, next)) return false
    snapshot = Object.freeze(next)
    for (const listener of [...listeners]) listener()
    return true
  }

  const cancelFrame = () => {
    if (frameHandle === null) return
    scheduler.cancel(frameHandle)
    frameHandle = null
  }

  let scheduleFrame: () => void

  const advanceTo = (timestampMs: number) => {
    const safeTimestamp = Number.isFinite(timestampMs)
      ? Math.max(lastTimestamp, timestampMs)
      : lastTimestamp
    const elapsedMs = Math.min(safeTimestamp - lastTimestamp, maxFrameDeltaMs)
    lastTimestamp = safeTimestamp
    if (!isAdvancing() || elapsedMs <= 0) return

    const rate = snapshot.speed / (snapshot.aiming ? aimingDurationMultiplier : 1)
    let remainingProgress = elapsedMs * rate / baseDayDurationMs
    if (remainingProgress <= 0) return

    let dayIndex = snapshot.dayIndex
    let progress = snapshot.progress
    let terminal = snapshot.terminal
    const boundaryEvents: PresentationDayChange[] = []
    let terminalEvent: PresentationTerminalEvent | null = null

    while (remainingProgress > 0 && !terminal) {
      const toBoundary = Math.max(0, 1 - progress)
      if (remainingProgress + Number.EPSILON < toBoundary) {
        progress += remainingProgress
        remainingProgress = 0
        break
      }

      remainingProgress = Math.max(0, remainingProgress - toBoundary)
      if (dayIndex < snapshot.dayCount - 1) {
        const fromDayIndex = dayIndex
        dayIndex += 1
        progress = 0
        boundaryEvents.push({
          completedDayIndex: fromDayIndex,
          fromDayIndex,
          toDayIndex: dayIndex,
        })
      } else {
        progress = 1
        terminal = true
        remainingProgress = 0
        terminalEvent = { dayIndex, dayCount: snapshot.dayCount }
      }
    }

    commit({
      ...snapshot,
      dayIndex,
      progress,
      absoluteDay: terminal ? snapshot.dayCount : dayIndex + progress,
      terminal,
    })
    for (const event of boundaryEvents) callbacks.onDayChange?.(event)
    if (terminalEvent) callbacks.onTerminal?.(terminalEvent)
  }

  const onFrame: PresentationFrameCallback = (timestampMs) => {
    frameHandle = null
    if (destroyed) return
    advanceTo(timestampMs)
    scheduleFrame()
  }

  scheduleFrame = () => {
    if (destroyed || frameHandle !== null || !isAdvancing()) return
    frameHandle = scheduler.request(onFrame)
  }

  const reschedule = () => {
    cancelFrame()
    lastTimestamp = now()
    scheduleFrame()
  }

  const syncBeforeControlChange = () => {
    if (isAdvancing()) advanceTo(now())
  }

  const setControl = <Key extends 'paused' | 'speed' | 'aiming' | 'blocked'>(
    key: Key,
    value: PresentationClockSnapshot[Key],
  ) => {
    if (destroyed) return
    syncBeforeControlChange()
    commit({ ...snapshot, [key]: value })
    reschedule()
  }

  scheduleFrame()

  return {
    subscribe(listener) {
      if (destroyed) return () => undefined
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    getSnapshot: () => snapshot,
    setPaused: (paused) => setControl('paused', paused),
    setSpeed: (speed) => setControl('speed', speed),
    setAiming: (aiming) => setControl('aiming', aiming),
    setBlocked: (blocked) => setControl('blocked', blocked),
    seek(cursor) {
      if (destroyed) return
      const normalized = normalizedCursor(cursor, snapshot.dayCount)
      commit({ ...snapshot, ...normalized })
      reschedule()
    },
    replaceRun(dayCount) {
      if (destroyed) return
      const nextDayCount = validDayCount(dayCount)
      const preserved = cursorForAbsoluteDay(snapshot.absoluteDay, nextDayCount)
      commit({ ...snapshot, ...preserved, dayCount: nextDayCount })
      reschedule()
    },
    setCallbacks(nextCallbacks) {
      callbacks = { ...nextCallbacks }
    },
    destroy() {
      if (destroyed) return
      destroyed = true
      cancelFrame()
      listeners.clear()
    },
  }
}
