import type { Service } from '../types'

const MASTER_LEVEL = 0.26
const REDUCED_SENSORY_LEVEL = 0.12
const MIN_GAIN = 0.0001

export type CityAudioSnapshot = {
  day: number
  shockType: string | null
  shockSeverity: number
  narration: string
  darkServices: readonly Service[]
  /** Keep the dark-state bed current while suppressing high-frequency scrub cues. */
  cuesEnabled: boolean
}

export type CityAudioCue =
  | { kind: 'impact'; key: string; severity: number }
  | { kind: 'relay'; key: string }
  | { kind: 'district-dark'; key: string; services: readonly Service[] }

function relayKey(snapshot: CityAudioSnapshot): string {
  return `${snapshot.day}:${snapshot.narration}`
}

function impactKey(snapshot: CityAudioSnapshot): string | null {
  if (!snapshot.shockType) return null
  return `${snapshot.day}:${snapshot.shockType}:${snapshot.shockSeverity.toFixed(4)}`
}

function darkKey(snapshot: CityAudioSnapshot | null): string {
  return snapshot ? [...snapshot.darkServices].sort().join(',') : ''
}

/**
 * Turns displayed, engine-backed run state into the only three allowed sound cues.
 * Keys deliberately ignore result ids so a forced-shock re-run cannot replay the
 * unchanged current day before its newly appended shock boundary arrives.
 */
export function deriveCityAudioCues(
  previous: CityAudioSnapshot | null,
  next: CityAudioSnapshot | null,
): CityAudioCue[] {
  const cues: CityAudioCue[] = []
  if (next?.cuesEnabled) {
    const nextImpactKey = impactKey(next)
    if (nextImpactKey && nextImpactKey !== (previous ? impactKey(previous) : null)) {
      cues.push({ kind: 'impact', key: nextImpactKey, severity: next.shockSeverity })
    }
    if (relayKey(next) !== (previous ? relayKey(previous) : null)) {
      cues.push({ kind: 'relay', key: relayKey(next) })
    }
  }

  if (darkKey(previous) !== darkKey(next)) {
    cues.push({ kind: 'district-dark', key: darkKey(next), services: next?.darkServices ?? [] })
  }
  return cues
}

function stringHash(value: string): number {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

function deterministicNoise(context: AudioContext, duration: number, seedText: string): AudioBuffer {
  const length = Math.max(1, Math.ceil(context.sampleRate * duration))
  const buffer = context.createBuffer(1, length, context.sampleRate)
  const data = buffer.getChannelData(0)
  let state = stringHash(seedText) || 1
  for (let index = 0; index < length; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    data[index] = (state / 4294967295) * 2 - 1
  }
  return buffer
}

function audioContextConstructor(): (new () => AudioContext) | null {
  if (typeof window === 'undefined') return null
  const windowWithWebkit = window as Window & { webkitAudioContext?: new () => AudioContext }
  if (typeof window.AudioContext === 'function') return window.AudioContext
  return windowWithWebkit.webkitAudioContext ?? null
}

type DroneNodes = {
  oscillator: OscillatorNode
  gain: GainNode
}

/** Lifecycle-safe, asset-free sound renderer for the recovery city. */
export class ProceduralCityAudio {
  private context: AudioContext | null = null
  private master: GainNode | null = null
  private snapshot: CityAudioSnapshot | null = null
  private renderedSnapshot: CityAudioSnapshot | null = null
  private drone: DroneNodes | null = null
  private muted = false
  private disposed = false

  update(snapshot: CityAudioSnapshot | null): void {
    this.snapshot = snapshot
    if (!this.context || !this.master || this.disposed) return
    const cues = deriveCityAudioCues(this.renderedSnapshot, snapshot)
    this.renderedSnapshot = snapshot
    this.render(cues)
  }

  unlock(): void {
    if (this.disposed) return
    if (this.context) {
      if (this.context.state === 'suspended') void this.context.resume().catch(() => undefined)
      return
    }
    const Context = audioContextConstructor()
    if (!Context) return
    try {
      const context = new Context()
      const master = context.createGain()
      master.gain.setValueAtTime(0, context.currentTime)
      master.connect(context.destination)
      this.context = context
      this.master = master
      this.applyMasterLevel()
      const cues = deriveCityAudioCues(null, this.snapshot)
      this.renderedSnapshot = this.snapshot
      this.render(cues)
      if (context.state === 'suspended') void context.resume().catch(() => undefined)
    } catch {
      this.context = null
      this.master = null
    }
  }

  setMuted(muted: boolean): void {
    this.muted = muted
    this.applyMasterLevel()
    if (muted) {
      this.stopDrone()
    } else if (this.snapshot?.darkServices.length) {
      this.startDrone(this.snapshot.darkServices)
    }
  }

  dispose(): void {
    this.disposed = true
    this.stopDrone()
    const context = this.context
    if (context && context.state !== 'closed') void context.close().catch(() => undefined)
    this.context = null
    this.master = null
    this.snapshot = null
    this.renderedSnapshot = null
  }

  private masterLevel(): number {
    if (this.muted) return 0
    const reducedSensoryMotion = typeof window !== 'undefined'
      && typeof window.matchMedia === 'function'
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    return reducedSensoryMotion ? REDUCED_SENSORY_LEVEL : MASTER_LEVEL
  }

  private applyMasterLevel(): void {
    if (!this.context || !this.master) return
    const now = this.context.currentTime
    this.master.gain.cancelScheduledValues(now)
    this.master.gain.setTargetAtTime(this.masterLevel(), now, 0.035)
  }

  private render(cues: CityAudioCue[]): void {
    const hasImpact = cues.some((cue) => cue.kind === 'impact')
    cues.forEach((cue) => {
      if (cue.kind === 'district-dark') {
        if (!cue.services.length || this.muted) this.stopDrone()
        else this.startDrone(cue.services)
        return
      }
      if (this.muted) return
      if (cue.kind === 'impact') this.playImpact(cue.key, cue.severity)
      else this.playRelay(cue.key, hasImpact ? 0.34 : 0)
    })
  }

  private playImpact(key: string, severity: number): void {
    const context = this.context
    const master = this.master
    if (!context || !master) return
    const strength = Math.max(0.125, Math.min(1, severity / 0.4))
    const now = context.currentTime + 0.012
    const duration = 0.62 + strength * 0.72

    const source = context.createBufferSource()
    source.buffer = deterministicNoise(context, duration, key)
    const filter = context.createBiquadFilter()
    filter.type = 'lowpass'
    filter.frequency.setValueAtTime(76 + strength * 42, now)
    filter.Q.setValueAtTime(0.72, now)
    const envelope = context.createGain()
    envelope.gain.setValueAtTime(MIN_GAIN, now)
    envelope.gain.exponentialRampToValueAtTime(0.055 + strength * 0.055, now + 0.045)
    envelope.gain.exponentialRampToValueAtTime(MIN_GAIN, now + duration)
    source.connect(filter).connect(envelope).connect(master)
    source.start(now)
    source.stop(now + duration + 0.02)

    const body = context.createOscillator()
    body.type = 'sine'
    body.frequency.setValueAtTime(36 + strength * 8, now)
    const bodyGain = context.createGain()
    bodyGain.gain.setValueAtTime(MIN_GAIN, now)
    bodyGain.gain.exponentialRampToValueAtTime(0.018 + strength * 0.018, now + 0.035)
    bodyGain.gain.exponentialRampToValueAtTime(MIN_GAIN, now + duration * 0.82)
    body.connect(bodyGain).connect(master)
    body.start(now)
    body.stop(now + duration)
  }

  private playRelay(key: string, delay: number): void {
    const context = this.context
    const master = this.master
    if (!context || !master) return
    const seed = stringHash(key)
    const frequency = 174 + (seed % 17)
    const pulseCount = 2 + (seed % 2)
    const start = context.currentTime + 0.018 + delay
    for (let index = 0; index < pulseCount; index += 1) {
      const onset = start + index * 0.105
      const oscillator = context.createOscillator()
      oscillator.type = 'sine'
      oscillator.frequency.setValueAtTime(frequency, onset)
      const filter = context.createBiquadFilter()
      filter.type = 'lowpass'
      filter.frequency.setValueAtTime(520, onset)
      const gain = context.createGain()
      gain.gain.setValueAtTime(MIN_GAIN, onset)
      gain.gain.exponentialRampToValueAtTime(0.028, onset + 0.012)
      gain.gain.exponentialRampToValueAtTime(MIN_GAIN, onset + 0.07)
      oscillator.connect(filter).connect(gain).connect(master)
      oscillator.start(onset)
      oscillator.stop(onset + 0.082)
    }
  }

  private startDrone(services: readonly Service[]): void {
    const context = this.context
    const master = this.master
    if (!context || !master || this.muted) return
    this.stopDrone()
    const now = context.currentTime
    const key = [...services].sort().join(',')
    const oscillator = context.createOscillator()
    oscillator.type = 'sine'
    oscillator.frequency.setValueAtTime(43 + (stringHash(key) % 6), now)
    const filter = context.createBiquadFilter()
    filter.type = 'lowpass'
    filter.frequency.setValueAtTime(105, now)
    const gain = context.createGain()
    gain.gain.setValueAtTime(MIN_GAIN, now)
    gain.gain.exponentialRampToValueAtTime(0.014 + Math.min(services.length, 5) * 0.003, now + 0.8)
    oscillator.connect(filter).connect(gain).connect(master)
    oscillator.start(now)
    this.drone = { oscillator, gain }
  }

  private stopDrone(): void {
    const context = this.context
    const drone = this.drone
    if (!context || !drone) return
    const now = context.currentTime
    drone.gain.gain.cancelScheduledValues(now)
    drone.gain.gain.setTargetAtTime(MIN_GAIN, now, 0.16)
    try {
      drone.oscillator.stop(now + 0.75)
    } catch {
      // The oscillator may already have been stopped during a rapid state change.
    }
    this.drone = null
  }
}
