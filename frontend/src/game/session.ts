import type { Scenario } from '../types'

export const gameModes = ['sandbox', 'stress'] as const
export type GameMode = (typeof gameModes)[number]

export const difficulties = ['calm', 'moderate', 'severe'] as const
export type Difficulty = (typeof difficulties)[number]

export type SessionLaunchSource = 'start-screen' | 'toolbox'

export type SessionSelection = {
  mode: GameMode
  difficulty: Difficulty
}

export type GameSessionState = SessionSelection & {
  arsenalLimit: number | null
  disastersUsed: number
}

export const STRESS_TEST_DISASTER_LIMIT = 6

export const MODE_DETAILS: Readonly<Record<GameMode, {
  label: string
  summary: string
  arsenalLimit: number | null
}>> = {
  sandbox: {
    label: 'Sandbox',
    summary: 'Unlimited disasters and an unscored run.',
    arsenalLimit: null,
  },
  stress: {
    label: 'Stress Test',
    summary: 'Six disasters, a finite run, and a plain-language debrief.',
    arsenalLimit: STRESS_TEST_DISASTER_LIMIT,
  },
}

type ScenarioPreset = Pick<Scenario,
  'shock_probability' | 'severity_min' | 'severity_max' | 'daily_budget'
>

export const DIFFICULTY_DETAILS: Readonly<Record<Difficulty, {
  label: string
  summary: string
  preset: ScenarioPreset
}>> = {
  calm: {
    label: 'Calm',
    summary: 'Fewer ambient shocks, gentler conditions, and more daily capacity.',
    preset: {
      shock_probability: 0.10,
      severity_min: 0.05,
      severity_max: 0.16,
      daily_budget: 220,
    },
  },
  moderate: {
    label: 'Moderate',
    summary: 'The city at its standard operating conditions.',
    preset: {
      shock_probability: 0.20,
      severity_min: 0.10,
      severity_max: 0.28,
      daily_budget: 180,
    },
  },
  severe: {
    label: 'Severe',
    summary: 'Frequent hard shocks and less daily recovery capacity.',
    preset: {
      shock_probability: 0.34,
      severity_min: 0.18,
      severity_max: 0.40,
      daily_budget: 140,
    },
  },
}

/**
 * Applies a start-screen preset without mutating the authored scenario. Scenarios
 * launched from the Analyst Toolbox pass through untouched so their raw settings
 * remain the source of truth.
 */
export function applyDifficultyPreset(
  scenario: Scenario,
  difficulty: Difficulty,
  source: SessionLaunchSource,
): Scenario {
  if (source === 'toolbox') return scenario
  return {
    ...scenario,
    ...DIFFICULTY_DETAILS[difficulty].preset,
    forced_shock: null,
    forced_shocks: [],
  }
}

export function createGameSession(selection: SessionSelection): GameSessionState {
  return {
    ...selection,
    arsenalLimit: MODE_DETAILS[selection.mode].arsenalLimit,
    disastersUsed: 0,
  }
}

export function remainingDisasters(session: GameSessionState): number | null {
  if (session.arsenalLimit === null) return null
  return Math.max(0, session.arsenalLimit - session.disastersUsed)
}

export function canUseDisaster(session: GameSessionState): boolean {
  const remaining = remainingDisasters(session)
  return remaining === null || remaining > 0
}

export function recordDisaster(session: GameSessionState): GameSessionState {
  if (!canUseDisaster(session)) return session
  return { ...session, disastersUsed: session.disastersUsed + 1 }
}
