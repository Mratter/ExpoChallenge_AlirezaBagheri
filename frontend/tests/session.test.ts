import { describe, expect, it } from 'vitest'
import { defaultScenario } from '../src/scenarios'
import {
  DIFFICULTY_DETAILS,
  STRESS_TEST_DISASTER_LIMIT,
  applyDifficultyPreset,
  canUseDisaster,
  createGameSession,
  recordDisaster,
  remainingDisasters,
} from '../src/game/session'

describe('game session presets', () => {
  it('maps Moderate to the engine defaults exactly', () => {
    expect(DIFFICULTY_DETAILS.moderate.preset).toEqual({
      shock_probability: 0.20,
      severity_min: 0.10,
      severity_max: 0.28,
      daily_budget: 180,
    })
  })

  it('maps Calm and Severe within the strict scenario schema', () => {
    expect(DIFFICULTY_DETAILS.calm.preset).toEqual({
      shock_probability: 0.10,
      severity_min: 0.05,
      severity_max: 0.16,
      daily_budget: 220,
    })
    expect(DIFFICULTY_DETAILS.severe.preset).toEqual({
      shock_probability: 0.34,
      severity_min: 0.18,
      severity_max: 0.40,
      daily_budget: 140,
    })
  })

  it('applies a preset only to normal start-screen sessions', () => {
    const authored = {
      ...defaultScenario,
      name: 'Authored in Toolbox',
      daily_budget: 307,
      shock_probability: 0.07,
      severity_min: 0.08,
      severity_max: 0.22,
    }

    const fromStart = applyDifficultyPreset(authored, 'severe', 'start-screen')
    const fromToolbox = applyDifficultyPreset(authored, 'severe', 'toolbox')

    expect(fromStart).not.toBe(authored)
    expect(fromStart).toMatchObject(DIFFICULTY_DETAILS.severe.preset)
    expect(fromStart.forced_shock).toBeNull()
    expect(fromStart.forced_shocks).toEqual([])
    expect(fromToolbox).toBe(authored)
    expect(fromToolbox.forced_shock).toEqual(defaultScenario.forced_shock)
    expect(authored).toMatchObject({
      daily_budget: 307,
      shock_probability: 0.07,
      severity_min: 0.08,
      severity_max: 0.22,
    })
  })
})

describe('finite Stress Test arsenal', () => {
  it('allows exactly six disasters before the arsenal is exhausted', () => {
    let session = createGameSession({ mode: 'stress', difficulty: 'moderate' })
    expect(session.arsenalLimit).toBe(STRESS_TEST_DISASTER_LIMIT)

    for (let index = 0; index < STRESS_TEST_DISASTER_LIMIT; index += 1) {
      expect(canUseDisaster(session)).toBe(true)
      session = recordDisaster(session)
    }

    expect(remainingDisasters(session)).toBe(0)
    expect(canUseDisaster(session)).toBe(false)
    expect(recordDisaster(session)).toBe(session)
  })

  it('keeps Sandbox unlimited', () => {
    let session = createGameSession({ mode: 'sandbox', difficulty: 'calm' })
    for (let index = 0; index < 20; index += 1) session = recordDisaster(session)

    expect(session.arsenalLimit).toBeNull()
    expect(remainingDisasters(session)).toBeNull()
    expect(canUseDisaster(session)).toBe(true)
    expect(session.disastersUsed).toBe(20)
  })
})
