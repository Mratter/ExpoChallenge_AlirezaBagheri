import { describe, expect, it } from 'vitest'
import { deriveCityAudioCues, type CityAudioSnapshot } from '../src/game/audio'

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
      { kind: 'impact', key: '4:utility:0.3100', severity: 0.31 },
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
})
