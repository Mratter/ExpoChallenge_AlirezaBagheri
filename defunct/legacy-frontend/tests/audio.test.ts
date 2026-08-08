import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  cityAudioMasterLevel,
  cityAudioMix,
  cityImpactSoundProfile,
  deriveCityAudioCues,
  type CityAudioSnapshot,
} from '../src/game/audio'
import {
  CITY_AUDIO_MUTED_STORAGE_KEY,
  persistCityAudioMuted,
  readStoredCityAudioMuted,
  useCityAudio,
} from '../src/game/useCityAudio'

function snapshot(overrides: Partial<CityAudioSnapshot> = {}): CityAudioSnapshot {
  return {
    day: 4,
    shockType: null,
    shockSeverity: 0,
    narration: 'DAY 4 — ROUTING 42 UNITS TO FOOD.',
    darkServices: [],
    cuesEnabled: true,
    ...overrides,
  }
}

describe('trajectory-derived city audio cues', () => {
  it('uses only the displayed shock, narration, and district-dark state for initial cues', () => {
    const current = snapshot({
      shockType: 'utility',
      shockSeverity: 0.31,
      narration: 'SHOCK DETECTED — CIVIC. SEVERITY 0.31.',
      darkServices: ['public_services'],
    })

    expect(deriveCityAudioCues(null, current)).toEqual([
      { kind: 'impact', key: '4:utility:0.3100', type: 'utility', severity: 0.31 },
      { kind: 'relay', key: '4:SHOCK DETECTED — CIVIC. SEVERITY 0.31.' },
      { kind: 'district-dark', key: 'public_services', services: ['public_services'] },
    ])
  })

  it('does not replay an unchanged day when a comparison result is replaced', () => {
    const current = snapshot({ shockType: 'weather', shockSeverity: 0.2 })
    expect(deriveCityAudioCues(current, { ...current })).toEqual([])
  })

  it('suppresses shock and RELAY cues during paused timeline scrubbing without replaying on resume', () => {
    const before = snapshot({ day: 3, narration: 'DAY 3 — ROUTING 38 UNITS TO HOUSING.' })
    const scrubbed = snapshot({
      day: 8,
      shockType: 'supply',
      shockSeverity: 0.25,
      narration: 'SHOCK DETECTED — FOOD. SEVERITY 0.25.',
      cuesEnabled: false,
    })
    expect(deriveCityAudioCues(before, scrubbed)).toEqual([])
    expect(deriveCityAudioCues(scrubbed, { ...scrubbed, cuesEnabled: true })).toEqual([])
  })

  it('starts and clears the quiet drone only when the real dark-service set changes', () => {
    const stable = snapshot()
    const dark = snapshot({ darkServices: ['food', 'housing'] })
    expect(deriveCityAudioCues(stable, dark)).toEqual([
      { kind: 'district-dark', key: 'food,housing', services: ['food', 'housing'] },
    ])
    expect(deriveCityAudioCues(dark, null)).toEqual([
      { kind: 'district-dark', key: '', services: [] },
    ])
  })

  it('sounds a siren only for a new verified emergency wave', () => {
    const stable = snapshot()
    const emergency = snapshot({
      emergencyWave: { id: 'day-4-healthcare-wave', active: true, strength: 0.72 },
    })

    expect(deriveCityAudioCues(stable, emergency)).toEqual([{
      kind: 'emergency-siren',
      key: 'emergency:day-4-healthcare-wave',
      strength: 0.72,
    }])
    expect(deriveCityAudioCues(emergency, { ...emergency })).toEqual([])
    expect(deriveCityAudioCues(stable, snapshot({
      emergencyWave: { id: 'inactive-wave', active: false, strength: 1 },
    }))).toEqual([])
  })

  it('beeps only when a new run-derived dock dwell begins', () => {
    const stable = snapshot()
    const dwelling = snapshot({ dockDwell: { id: 417, active: true, strength: 0.55 } })

    expect(deriveCityAudioCues(stable, dwelling)).toEqual([{
      kind: 'dock-beep',
      key: 'dock:417',
      strength: 0.55,
    }])
    expect(deriveCityAudioCues(dwelling, { ...dwelling, dockDwell: { ...dwelling.dockDwell! } })).toEqual([])
    expect(deriveCityAudioCues(stable, snapshot({
      dockDwell: { id: 418, active: false, strength: 0.9 },
    }))).toEqual([])
  })

  it('gives terminal fall silence priority over every simultaneous cue', () => {
    const before = snapshot({ darkServices: ['food'] })
    const fallen = snapshot({
      shockType: 'aftershock',
      shockSeverity: 0.4,
      narration: 'POST-FALL FACTUAL LINE.',
      darkServices: ['food', 'healthcare'],
      emergencyWave: { id: 'terminal', active: true },
      dockDwell: { id: 'terminal', active: true },
      fallen: true,
    })

    expect(deriveCityAudioCues(before, fallen)).toEqual([
      { kind: 'fall-silence', key: 'fall:4' },
    ])
    expect(deriveCityAudioCues(fallen, { ...fallen })).toEqual([])
    expect(deriveCityAudioCues(fallen, null)).toEqual([])
  })
})

describe('state-mixed procedural ambience', () => {
  it('maps only supplied run-derived activity levels into restrained beds', () => {
    expect(cityAudioMix(snapshot({
      trafficActivity: 0.64,
      constructionActivity: 0.37,
      weatherActivity: 0.22,
    }))).toEqual({ traffic: 0.64, construction: 0.37, rain: 0.22 })

    // Old snapshots remain valid and silent rather than inventing activity.
    expect(cityAudioMix(snapshot())).toEqual({ traffic: 0, construction: 0, rain: 0 })
  })

  it('uses the real typed weather severity as a backward-compatible rain fallback', () => {
    expect(cityAudioMix(snapshot({ shockType: 'weather', shockSeverity: 0.2 })).rain).toBe(0.5)
    expect(cityAudioMix(snapshot({ shockType: 'utility', shockSeverity: 0.4 })).rain).toBe(0)
    expect(cityAudioMix(snapshot({
      shockType: 'weather',
      shockSeverity: 0.4,
      weatherActivity: 0.18,
    })).rain).toBe(0.18)
  })

  it('clamps malformed presentation levels and silences every bed after a fall', () => {
    expect(cityAudioMix(snapshot({
      trafficActivity: 4,
      constructionActivity: -2,
      weatherActivity: Number.NaN,
    }))).toEqual({ traffic: 1, construction: 0, rain: 0 })
    expect(cityAudioMix(snapshot({
      trafficActivity: 1,
      constructionActivity: 1,
      weatherActivity: 1,
      fallen: true,
    }))).toEqual({ traffic: 0, construction: 0, rain: 0 })
  })

  it('routes the raw frozen shock keys to distinct typed sound profiles', () => {
    expect(cityImpactSoundProfile('aftershock')).toBe('earthquake-rumble')
    expect(cityImpactSoundProfile('weather')).toBe('weather-rain')
    for (const type of ['supply', 'epidemic', 'utility', null]) {
      expect(cityImpactSoundProfile(type)).toBe('restrained-impact')
    }
  })

  it('lowers the complete mix for sensory reduction and always honors mute', () => {
    expect(cityAudioMasterLevel(false, true)).toBeLessThan(cityAudioMasterLevel(false, false))
    expect(cityAudioMasterLevel(true, false)).toBe(0)
    expect(cityAudioMasterLevel(true, true)).toBe(0)
  })
})

function AudioPreferenceHarness() {
  const { muted, toggleMuted } = useCityAudio(null)
  return createElement(
    'button',
    { type: 'button', 'aria-pressed': muted, onClick: toggleMuted },
    muted ? 'Sound off' : 'Sound on',
  )
}

describe('persistent city mute preference', () => {
  beforeEach(() => window.localStorage.clear())
  afterEach(() => {
    cleanup()
    window.localStorage.clear()
  })

  it('restores the saved preference and persists each one-click change', () => {
    window.localStorage.setItem(CITY_AUDIO_MUTED_STORAGE_KEY, 'true')
    const view = render(createElement(AudioPreferenceHarness))

    expect(screen.getByRole('button', { name: 'Sound off' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'Sound off' }))
    expect(screen.getByRole('button', { name: 'Sound on' })).toHaveAttribute('aria-pressed', 'false')
    expect(window.localStorage.getItem(CITY_AUDIO_MUTED_STORAGE_KEY)).toBe('false')

    view.unmount()
    render(createElement(AudioPreferenceHarness))
    expect(screen.getByRole('button', { name: 'Sound on' })).toHaveAttribute('aria-pressed', 'false')
  })

  it('fails safely when browser storage is unavailable', () => {
    const unavailable = {
      getItem: () => { throw new Error('blocked') },
      setItem: () => { throw new Error('blocked') },
    }
    expect(readStoredCityAudioMuted(unavailable)).toBe(false)
    expect(() => persistCityAudioMuted(true, unavailable)).not.toThrow()
  })
})
