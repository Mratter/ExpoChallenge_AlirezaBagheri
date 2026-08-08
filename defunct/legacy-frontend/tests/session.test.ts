import { describe, expect, it } from 'vitest'
import { defaultScenario } from '../src/scenarios'
import { GAME_HORIZON_DAYS } from '../src/game/pacing'
import {
  DIFFICULTY_DETAILS,
  SCENARIO_PRESETS,
  STRESS_TEST_DISASTER_LIMIT,
  TUTORIAL_SEED,
  TUTORIAL_SHOCK,
  applyAuthoredScenarioPreset,
  applyDifficultyPreset,
  canUseDisaster,
  createGameSession,
  createTutorialScenario,
  createTutorialSession,
  recordDisaster,
  remainingDisasters,
  storedForcedShocksBeforePlayer,
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
      horizon_days: 9,
      daily_budget: 307,
      shock_probability: 0.07,
      severity_min: 0.08,
      severity_max: 0.22,
    }

    const fromStart = applyDifficultyPreset(authored, 'severe', 'start-screen')
    const fromToolbox = applyDifficultyPreset(authored, 'severe', 'toolbox')

    expect(fromStart).not.toBe(authored)
    expect(fromStart).toMatchObject(DIFFICULTY_DETAILS.severe.preset)
    expect(fromStart.horizon_days).toBe(GAME_HORIZON_DAYS)
    expect(fromStart.forced_shock).toBeNull()
    expect(fromStart.forced_shocks).toEqual([])
    expect(fromToolbox).toBe(authored)
    expect(fromToolbox.horizon_days).toBe(9)
    expect(fromToolbox.forced_shock).toEqual(defaultScenario.forced_shock)
    expect(authored).toMatchObject({
      horizon_days: 9,
      daily_budget: 307,
      shock_probability: 0.07,
      severity_min: 0.08,
      severity_max: 0.22,
    })
  })

  it('keeps the 20-day game horizon independent of difficulty', () => {
    for (const difficulty of ['calm', 'moderate', 'severe'] as const) {
      expect(
        applyDifficultyPreset(defaultScenario, difficulty, 'start-screen').horizon_days,
      ).toBe(GAME_HORIZON_DAYS)
    }
  })

  it('composes each disclosed authored schedule after applying difficulty', () => {
    expect(applyAuthoredScenarioPreset(defaultScenario, 'calm', 'fault-line')).toMatchObject({
      name: 'Relay City — Fault-line city',
      horizon_days: 20,
      daily_budget: 220,
      shock_probability: 0.1,
      forced_shock: null,
      forced_shocks: SCENARIO_PRESETS['fault-line'].forcedShocks,
    })
    expect(SCENARIO_PRESETS.coastal.forcedShocks).toEqual([
      { day: 4, type: 'weather', severity: 0.24 },
      { day: 10, type: 'supply', severity: 0.18 },
      { day: 16, type: 'weather', severity: 0.34 },
    ])
    expect(SCENARIO_PRESETS['supply-corridor'].forcedShocks).toHaveLength(4)
  })

  it('builds an eight-day fixed tutorial with no ambient draws or player arsenal', () => {
    expect(TUTORIAL_SEED).toBe(17_008)
    expect(createTutorialScenario(defaultScenario)).toMatchObject({
      name: 'Relay City — guided weather incident',
      horizon_days: 8,
      daily_budget: 180,
      shock_probability: 0,
      forced_shock: null,
      forced_shocks: [TUTORIAL_SHOCK],
    })
    const session = createTutorialSession()
    expect(session.tutorial).toBe(true)
    expect(remainingDisasters(session)).toBe(0)
    expect(canUseDisaster(session)).toBe(false)
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

  it('records exact player-shock provenance without consuming authored events', () => {
    const session = createGameSession({ mode: 'stress', difficulty: 'moderate', preset: 'fault-line' })
    const shock = { day: 3, type: 'aftershock' as const, severity: 0.24 }
    const next = recordDisaster(session, shock)
    expect(next.disastersUsed).toBe(1)
    expect(next.playerShocks).toEqual([shock])
    expect(SCENARIO_PRESETS['fault-line'].forcedShocks).toHaveLength(3)
  })

  it('keeps restored forced events unknown while recognizing only the appended player suffix', () => {
    const playerShocks = [
      { day: 3, type: 'aftershock' as const, severity: 0.24 },
      { day: 7, type: 'weather' as const, severity: 0.31 },
    ]
    const scenario = {
      ...defaultScenario,
      forced_shock: { day: 2, type: 'utility' as const, severity: 0.2 },
      forced_shocks: [
        { day: 3, type: 'aftershock' as const, severity: 0.24 },
        ...playerShocks,
      ],
    }

    expect(storedForcedShocksBeforePlayer(scenario, playerShocks)).toEqual([
      { day: 2, type: 'utility', severity: 0.2 },
      { day: 3, type: 'aftershock', severity: 0.24 },
    ])
  })
})
