import type { Service } from '../types'

const MASTER_LEVEL = 0.26
const REDUCED_SENSORY_LEVEL = 0.12
const MIN_GAIN = 0.0001

export type CityAudioTransient = {
  /** Stable run-derived identity. Reusing an id while active never repeats a cue. */
  id: string | number
  active: boolean
  /** Normalized presentation strength; values outside 0..1 are clamped. */
  strength?: number
}

/**
 * The original fields remain required so existing callers keep their compile-time
 * contract. New realism fields are optional and silent by default until CityGame can
 * supply the corresponding run-derived state.
 */
export type CityAudioSnapshot = {
  day: number
  shockType: string | null
  shockSeverity: number
  narration: string
  darkServices: readonly Service[]
  /** Keep state beds current while suppressing high-frequency scrub cues. */
  cuesEnabled: boolean
  /** Weighted city activity in 0..1, used only for the traffic bed. */
  trafficActivity?: number
  /** Active repair-site intensity in 0..1, used only for construction taps. */
  constructionActivity?: number
  /** Typed weather intensity in 0..1; weather shocks provide a safe fallback. */
  weatherActivity?: number
  /** Siren is legal only while this verified emergency-response wave is active. */
  emergencyWave?: CityAudioTransient | null
  /** Forklift beeps are legal only while this verified dock dwell is active. */
  dockDwell?: CityAudioTransient | null
  /** A false-to-true transition produces one quiet terminal silence beat. */
  fallen?: boolean
  /** Explicit accessibility signal in addition to the system media preference. */
  reducedSensory?: boolean
}

export type CityAudioCue =
  | { kind: 'impact'; key: string; type: string; severity: number }
  | { kind: 'relay'; key: string }
  | { kind: 'district-dark'; key: string; services: readonly Service[] }
  | { kind: 'emergency-siren'; key: string; strength: number }
  | { kind: 'dock-beep'; key: string; strength: number }
  | { kind: 'fall-silence'; key: string }

export type CityAudioMix = {
  traffic: number
  construction: number
  rain: number
}

export type CityImpactSoundProfile = 'earthquake-rumble' | 'weather-rain' | 'restrained-impact'

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, Number.isFinite(value) ? value : minimum))
}

export function cityAudioMasterLevel(muted: boolean, reducedSensory: boolean): number {
  if (muted) return 0
  return reducedSensory ? REDUCED_SENSORY_LEVEL : MASTER_LEVEL
}

/** Pure state-to-mix mapping used by both tests and the WebAudio renderer. */
export function cityAudioMix(snapshot: CityAudioSnapshot | null): CityAudioMix {
  if (!snapshot || snapshot.fallen) return { traffic: 0, construction: 0, rain: 0 }
  const weatherFallback = snapshot.shockType === 'weather'
    ? clamp(snapshot.shockSeverity / 0.4)
    : 0
  return {
    traffic: clamp(snapshot.trafficActivity ?? 0),
    construction: clamp(snapshot.constructionActivity ?? 0),
    rain: clamp(snapshot.weatherActivity ?? weatherFallback),
  }
}

export function cityImpactSoundProfile(type: string | null): CityImpactSoundProfile {
  if (type === 'aftershock') return 'earthquake-rumble'
  if (type === 'weather') return 'weather-rain'
  return 'restrained-impact'
}

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

function transientKey(prefix: string, transient: CityAudioTransient | null | undefined): string | null {
  return transient?.active ? `${prefix}:${transient.id}` : null
}

/**
 * Turns displayed, engine-backed run state into restrained one-shot cues. Keys ignore
 * result ids so a forced-shock re-run cannot replay the unchanged current day before
 * its newly appended shock boundary arrives.
 */
export function deriveCityAudioCues(
  previous: CityAudioSnapshot | null,
  next: CityAudioSnapshot | null,
): CityAudioCue[] {
  if (next?.fallen && !previous?.fallen) {
    return [{ kind: 'fall-silence', key: `fall:${next.day}` }]
  }
  if (previous?.fallen && !next) return []

  const cues: CityAudioCue[] = []
  if (next?.cuesEnabled && !next.fallen) {
    const nextImpactKey = impactKey(next)
    if (nextImpactKey && nextImpactKey !== (previous ? impactKey(previous) : null)) {
      cues.push({
        kind: 'impact',
        key: nextImpactKey,
        type: next.shockType!,
        severity: next.shockSeverity,
      })
    }
    if (relayKey(next) !== (previous ? relayKey(previous) : null)) {
      cues.push({ kind: 'relay', key: relayKey(next) })
    }

    const nextEmergencyKey = transientKey('emergency', next.emergencyWave)
    if (nextEmergencyKey && nextEmergencyKey !== transientKey('emergency', previous?.emergencyWave)) {
      cues.push({
        kind: 'emergency-siren',
        key: nextEmergencyKey,
        strength: clamp(next.emergencyWave?.strength ?? next.shockSeverity / 0.4),
      })
    }

    const nextDockKey = transientKey('dock', next.dockDwell)
    if (nextDockKey && nextDockKey !== transientKey('dock', previous?.dockDwell)) {
      cues.push({
        kind: 'dock-beep',
        key: nextDockKey,
        strength: clamp(next.dockDwell?.strength ?? 0.5),
      })
    }
  }

  if (!next?.fallen && darkKey(previous) !== darkKey(next)) {
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

function nextNoiseState(state: number): number {
  return (Math.imul(state, 1664525) + 1013904223) >>> 0
}

function deterministicNoise(context: AudioContext, duration: number, seedText: string): AudioBuffer {
  const length = Math.max(1, Math.ceil(context.sampleRate * duration))
  const buffer = context.createBuffer(1, length, context.sampleRate)
  const data = buffer.getChannelData(0)
  let state = stringHash(seedText) || 1
  for (let index = 0; index < length; index += 1) {
    state = nextNoiseState(state)
    data[index] = (state / 4294967295) * 2 - 1
  }
  return buffer
}

function deterministicTapPattern(
  context: AudioContext,
  duration: number,
  seedText: string,
  tapCount: number,
): AudioBuffer {
  const length = Math.max(1, Math.ceil(context.sampleRate * duration))
  const buffer = context.createBuffer(1, length, context.sampleRate)
  const data = buffer.getChannelData(0)
  const tapLength = Math.max(1, Math.round(context.sampleRate * 0.018))
  let state = stringHash(seedText) || 1
  for (let tap = 0; tap < tapCount; tap += 1) {
    state = nextNoiseState(state)
    const segment = length / Math.max(tapCount, 1)
    const start = Math.min(
      length - 1,
      Math.floor(tap * segment + (state / 4294967295) * segment * 0.72),
    )
    const amplitude = 0.35 + ((state >>> 8) / 16777215) * 0.45
    for (let sample = 0; sample < tapLength && start + sample < length; sample += 1) {
      state = nextNoiseState(state)
      const envelope = Math.pow(1 - sample / tapLength, 3)
      data[start + sample] += ((state / 4294967295) * 2 - 1) * envelope * amplitude
    }
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

type AmbienceBed = {
  source: AudioBufferSourceNode
  gain: GainNode
}

type AmbienceBeds = {
  traffic: AmbienceBed
  construction: AmbienceBed
  rain: AmbienceBed
}

/** Lifecycle-safe, asset-free sound renderer for the recovery city. */
export class ProceduralCityAudio {
  private context: AudioContext | null = null
  private master: GainNode | null = null
  private snapshot: CityAudioSnapshot | null = null
  private renderedSnapshot: CityAudioSnapshot | null = null
  private drone: DroneNodes | null = null
  private ambience: AmbienceBeds | null = null
  private muted = false
  private reducedSensory = false
  private disposed = false

  update(snapshot: CityAudioSnapshot | null): void {
    const sensoryChanged = Boolean(this.snapshot?.reducedSensory) !== Boolean(snapshot?.reducedSensory)
    this.snapshot = snapshot
    if (!this.context || !this.master || this.disposed) return
    if (sensoryChanged) this.applyMasterLevel()
    const cues = deriveCityAudioCues(this.renderedSnapshot, snapshot)
    this.renderedSnapshot = snapshot
    this.syncAmbience(cityAudioMix(snapshot))
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
      this.startAmbience()
      this.applyMasterLevel()
      this.syncAmbience(cityAudioMix(this.snapshot))
      const cues = deriveCityAudioCues(null, this.snapshot)
      this.renderedSnapshot = this.snapshot
      this.render(cues)
      if (context.state === 'suspended') void context.resume().catch(() => undefined)
    } catch {
      this.context = null
      this.master = null
      this.ambience = null
    }
  }

  setMuted(muted: boolean): void {
    this.muted = muted
    this.applyMasterLevel()
    if (muted) {
      this.stopDrone()
    } else if (this.snapshot?.darkServices.length && !this.snapshot.fallen) {
      this.startDrone(this.snapshot.darkServices)
    }
    this.syncAmbience(cityAudioMix(this.snapshot))
  }

  setReducedSensory(reduced: boolean): void {
    this.reducedSensory = reduced
    this.applyMasterLevel()
    this.syncAmbience(cityAudioMix(this.snapshot))
  }

  dispose(): void {
    this.disposed = true
    this.stopDrone()
    this.stopAmbience()
    const context = this.context
    if (context && context.state !== 'closed') void context.close().catch(() => undefined)
    this.context = null
    this.master = null
    this.snapshot = null
    this.renderedSnapshot = null
  }

  private sensoryReductionActive(): boolean {
    if (this.reducedSensory || this.snapshot?.reducedSensory) return true
    return typeof window !== 'undefined'
      && typeof window.matchMedia === 'function'
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches
  }

  private masterLevel(): number {
    return cityAudioMasterLevel(this.muted, this.sensoryReductionActive())
  }

  private sensoryFactor(): number {
    return this.sensoryReductionActive() ? 0.52 : 1
  }

  private applyMasterLevel(): void {
    if (!this.context || !this.master) return
    const now = this.context.currentTime
    this.master.gain.cancelScheduledValues(now)
    this.master.gain.setTargetAtTime(this.masterLevel(), now, 0.035)
  }

  private createAmbienceBed(
    kind: keyof AmbienceBeds,
    buffer: AudioBuffer,
    filterType: BiquadFilterType,
    frequency: number,
    q: number,
  ): AmbienceBed | null {
    const context = this.context
    const master = this.master
    if (!context || !master) return null
    const source = context.createBufferSource()
    source.buffer = buffer
    source.loop = true
    const filter = context.createBiquadFilter()
    filter.type = filterType
    filter.frequency.setValueAtTime(frequency, context.currentTime)
    filter.Q.setValueAtTime(q, context.currentTime)
    const gain = context.createGain()
    gain.gain.setValueAtTime(MIN_GAIN, context.currentTime)
    source.connect(filter).connect(gain).connect(master)
    source.start(context.currentTime + (stringHash(kind) % 17) / 1000)
    return { source, gain }
  }

  private startAmbience(): void {
    const context = this.context
    if (!context || this.ambience) return
    const traffic = this.createAmbienceBed(
      'traffic',
      deterministicNoise(context, 3.1, 'relay-city:traffic:v1'),
      'bandpass',
      118,
      0.55,
    )
    const construction = this.createAmbienceBed(
      'construction',
      deterministicTapPattern(context, 2.7, 'relay-city:construction:v1', 9),
      'highpass',
      480,
      0.72,
    )
    const rain = this.createAmbienceBed(
      'rain',
      deterministicNoise(context, 2.9, 'relay-city:rain:v1'),
      'highpass',
      920,
      0.48,
    )
    if (traffic && construction && rain) this.ambience = { traffic, construction, rain }
  }

  private syncAmbience(mix: CityAudioMix): void {
    const context = this.context
    const ambience = this.ambience
    if (!context || !ambience) return
    const sensory = this.sensoryFactor()
    const targets: Record<keyof AmbienceBeds, number> = {
      traffic: MIN_GAIN + mix.traffic * 0.017 * sensory,
      construction: MIN_GAIN + mix.construction * 0.011 * sensory,
      rain: MIN_GAIN + mix.rain * 0.023 * sensory,
    }
    ;(Object.keys(targets) as (keyof AmbienceBeds)[]).forEach((kind) => {
      const parameter = ambience[kind].gain.gain
      parameter.cancelScheduledValues(context.currentTime)
      parameter.setTargetAtTime(targets[kind], context.currentTime, 0.18)
    })
  }

  private stopAmbience(): void {
    const ambience = this.ambience
    if (!ambience) return
    for (const bed of Object.values(ambience)) {
      try {
        bed.source.stop()
      } catch {
        // A partially initialized context may already have stopped a source.
      }
    }
    this.ambience = null
  }

  private render(cues: CityAudioCue[]): void {
    const fall = cues.find((cue) => cue.kind === 'fall-silence')
    if (fall) {
      this.playFallSilence()
      return
    }
    const hasImpact = cues.some((cue) => cue.kind === 'impact')
    cues.forEach((cue) => {
      if (cue.kind === 'district-dark') {
        if (!cue.services.length || this.muted) this.stopDrone()
        else this.startDrone(cue.services)
        return
      }
      if (this.muted) return
      if (cue.kind === 'impact') this.playImpact(cue.key, cue.type, cue.severity)
      else if (cue.kind === 'relay') this.playRelay(cue.key, hasImpact ? 0.34 : 0)
      else if (cue.kind === 'emergency-siren') this.playEmergencySiren(cue.key, cue.strength)
      else if (cue.kind === 'dock-beep') this.playDockBeep(cue.key, cue.strength)
    })
  }

  private playImpact(key: string, type: string, severity: number): void {
    const profile = cityImpactSoundProfile(type)
    if (profile === 'earthquake-rumble') {
      this.playEarthquakeRumble(key, severity)
      this.playBrickClatter(key, severity)
    } else if (profile === 'weather-rain') {
      this.playDistantThunder(key, severity)
    } else {
      this.playRestrainedImpact(key, severity)
    }
  }

  private playEarthquakeRumble(key: string, severity: number): void {
    const context = this.context
    const master = this.master
    if (!context || !master) return
    const strength = clamp(severity / 0.4, 0.125, 1)
    const sensory = this.sensoryFactor()
    const now = context.currentTime + 0.012
    const duration = 0.72 + strength * 0.78
    const source = context.createBufferSource()
    source.buffer = deterministicNoise(context, duration, `${key}:rumble`)
    const filter = context.createBiquadFilter()
    filter.type = 'lowpass'
    filter.frequency.setValueAtTime(68 + strength * 38, now)
    filter.Q.setValueAtTime(0.72, now)
    const envelope = context.createGain()
    envelope.gain.setValueAtTime(MIN_GAIN, now)
    envelope.gain.exponentialRampToValueAtTime((0.05 + strength * 0.055) * sensory, now + 0.045)
    envelope.gain.exponentialRampToValueAtTime(MIN_GAIN, now + duration)
    source.connect(filter).connect(envelope).connect(master)
    source.start(now)
    source.stop(now + duration + 0.02)

    const body = context.createOscillator()
    body.type = 'sine'
    body.frequency.setValueAtTime(34 + strength * 8, now)
    const bodyGain = context.createGain()
    bodyGain.gain.setValueAtTime(MIN_GAIN, now)
    bodyGain.gain.exponentialRampToValueAtTime((0.016 + strength * 0.017) * sensory, now + 0.035)
    bodyGain.gain.exponentialRampToValueAtTime(MIN_GAIN, now + duration * 0.84)
    body.connect(bodyGain).connect(master)
    body.start(now)
    body.stop(now + duration)
  }

  private playBrickClatter(key: string, severity: number): void {
    const context = this.context
    const master = this.master
    if (!context || !master) return
    const strength = clamp(severity / 0.4, 0.125, 1)
    const sensory = this.sensoryFactor()
    const now = context.currentTime + 0.08
    const source = context.createBufferSource()
    source.buffer = deterministicTapPattern(
      context,
      0.82,
      `${key}:brick-clatter`,
      Math.max(3, Math.round(4 + strength * (this.sensoryReductionActive() ? 3 : 7))),
    )
    const filter = context.createBiquadFilter()
    filter.type = 'bandpass'
    filter.frequency.setValueAtTime(740 + strength * 360, now)
    filter.Q.setValueAtTime(0.8, now)
    const gain = context.createGain()
    gain.gain.setValueAtTime((0.018 + strength * 0.018) * sensory, now)
    gain.gain.exponentialRampToValueAtTime(MIN_GAIN, now + 0.86)
    source.connect(filter).connect(gain).connect(master)
    source.start(now)
    source.stop(now + 0.88)
  }

  private playDistantThunder(key: string, severity: number): void {
    const context = this.context
    const master = this.master
    if (!context || !master) return
    const strength = clamp(severity / 0.4, 0.125, 1)
    const sensory = this.sensoryFactor()
    const now = context.currentTime + 0.38 + (stringHash(key) % 11) / 100
    const duration = 1.1 + strength * 0.75
    const source = context.createBufferSource()
    source.buffer = deterministicNoise(context, duration, `${key}:thunder`)
    const filter = context.createBiquadFilter()
    filter.type = 'lowpass'
    filter.frequency.setValueAtTime(82 + strength * 36, now)
    filter.Q.setValueAtTime(0.58, now)
    const gain = context.createGain()
    gain.gain.setValueAtTime(MIN_GAIN, now)
    gain.gain.exponentialRampToValueAtTime((0.026 + strength * 0.03) * sensory, now + 0.13)
    gain.gain.exponentialRampToValueAtTime(MIN_GAIN, now + duration)
    source.connect(filter).connect(gain).connect(master)
    source.start(now)
    source.stop(now + duration + 0.02)
  }

  private playRestrainedImpact(key: string, severity: number): void {
    const context = this.context
    const master = this.master
    if (!context || !master) return
    const strength = clamp(severity / 0.4, 0.125, 1)
    const sensory = this.sensoryFactor()
    const now = context.currentTime + 0.012
    const duration = 0.32 + strength * 0.32
    const source = context.createBufferSource()
    source.buffer = deterministicNoise(context, duration, `${key}:impact`)
    const filter = context.createBiquadFilter()
    filter.type = 'lowpass'
    filter.frequency.setValueAtTime(92 + (stringHash(key) % 34), now)
    const gain = context.createGain()
    gain.gain.setValueAtTime(MIN_GAIN, now)
    gain.gain.exponentialRampToValueAtTime((0.024 + strength * 0.026) * sensory, now + 0.035)
    gain.gain.exponentialRampToValueAtTime(MIN_GAIN, now + duration)
    source.connect(filter).connect(gain).connect(master)
    source.start(now)
    source.stop(now + duration + 0.02)
  }

  private playEmergencySiren(key: string, strengthValue: number): void {
    const context = this.context
    const master = this.master
    if (!context || !master) return
    const strength = clamp(strengthValue)
    const sensory = this.sensoryFactor()
    const start = context.currentTime + 0.02
    const passCount = this.sensoryReductionActive() ? 1 : 2
    for (let pass = 0; pass < passCount; pass += 1) {
      const onset = start + pass * 0.48
      const duration = 0.42
      const oscillator = context.createOscillator()
      oscillator.type = 'sine'
      const base = 382 + (stringHash(key) % 23)
      oscillator.frequency.setValueAtTime(base, onset)
      oscillator.frequency.linearRampToValueAtTime(base + 92 + strength * 38, onset + duration * 0.46)
      oscillator.frequency.exponentialRampToValueAtTime(base * 0.92, onset + duration)
      const panner = context.createStereoPanner()
      panner.pan.setValueAtTime(-0.72, onset)
      panner.pan.linearRampToValueAtTime(0.72, onset + duration)
      const gain = context.createGain()
      gain.gain.setValueAtTime(MIN_GAIN, onset)
      gain.gain.exponentialRampToValueAtTime((0.008 + strength * 0.007) * sensory, onset + 0.06)
      gain.gain.exponentialRampToValueAtTime(MIN_GAIN, onset + duration)
      oscillator.connect(panner).connect(gain).connect(master)
      oscillator.start(onset)
      oscillator.stop(onset + duration + 0.02)
    }
  }

  private playDockBeep(key: string, strengthValue: number): void {
    const context = this.context
    const master = this.master
    if (!context || !master) return
    const strength = clamp(strengthValue)
    const sensory = this.sensoryFactor()
    const frequency = 720 + (stringHash(key) % 73)
    const pulseCount = this.sensoryReductionActive() ? 1 : 2
    for (let index = 0; index < pulseCount; index += 1) {
      const onset = context.currentTime + 0.02 + index * 0.14
      const oscillator = context.createOscillator()
      oscillator.type = 'sine'
      oscillator.frequency.setValueAtTime(frequency, onset)
      const gain = context.createGain()
      gain.gain.setValueAtTime(MIN_GAIN, onset)
      gain.gain.exponentialRampToValueAtTime((0.006 + strength * 0.004) * sensory, onset + 0.012)
      gain.gain.exponentialRampToValueAtTime(MIN_GAIN, onset + 0.065)
      oscillator.connect(gain).connect(master)
      oscillator.start(onset)
      oscillator.stop(onset + 0.075)
    }
  }

  private playFallSilence(): void {
    const context = this.context
    const master = this.master
    if (!context || !master) return
    this.stopDrone()
    const now = context.currentTime
    const returnLevel = this.masterLevel()
    master.gain.cancelScheduledValues(now)
    master.gain.setTargetAtTime(0, now, 0.045)
    master.gain.setValueAtTime(0, now + 0.72)
    master.gain.setTargetAtTime(returnLevel, now + 0.94, 0.16)
  }

  private playRelay(key: string, delay: number): void {
    const context = this.context
    const master = this.master
    if (!context || !master) return
    const seed = stringHash(key)
    const frequency = 174 + (seed % 17)
    const pulseCount = this.sensoryReductionActive() ? 1 : 2 + (seed % 2)
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
      gain.gain.exponentialRampToValueAtTime(0.028 * this.sensoryFactor(), onset + 0.012)
      gain.gain.exponentialRampToValueAtTime(MIN_GAIN, onset + 0.07)
      oscillator.connect(filter).connect(gain).connect(master)
      oscillator.start(onset)
      oscillator.stop(onset + 0.082)
    }
  }

  private startDrone(services: readonly Service[]): void {
    const context = this.context
    const master = this.master
    if (!context || !master || this.muted || this.snapshot?.fallen) return
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
    gain.gain.exponentialRampToValueAtTime(
      (0.014 + Math.min(services.length, 5) * 0.003) * this.sensoryFactor(),
      now + 0.8,
    )
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
