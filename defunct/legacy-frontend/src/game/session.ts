import type { ForcedShock, Scenario } from '../types'
import { GAME_HORIZON_DAYS } from './pacing'

export const gameModes = ['sandbox', 'stress'] as const
export type GameMode = (typeof gameModes)[number]

export const difficulties = ['calm', 'moderate', 'severe'] as const
export type Difficulty = (typeof difficulties)[number]

export const scenarioPresetIds = ['fault-line', 'coastal', 'supply-corridor'] as const
export type ScenarioPresetId = (typeof scenarioPresetIds)[number]

export type SessionLaunchSource = 'start-screen' | 'toolbox'

export type SessionSelection = {
  mode: GameMode
  difficulty: Difficulty
  preset?: ScenarioPresetId | null
}

export type GameSessionState = {
  mode: GameMode
  difficulty: Difficulty
  scenarioPreset: ScenarioPresetId | null
  tutorial: boolean
  arsenalLimit: number | null
  disastersUsed: number
  playerShocks: ForcedShock[]
}

export const STRESS_TEST_DISASTER_LIMIT = 6

export const MODE_DETAILS: Readonly<Record<GameMode, {
  label: string
  summary: string
  arsenalLimit: number | null
}>> = {
  sandbox: {
    label: 'Sandbox',
    summary: 'Unlimited disasters and an unscored Relay City run.',
    arsenalLimit: null,
  },
  stress: {
    label: 'Stress Test',
    summary: 'Six player disasters, a finite run, and a plain-language debrief.',
    arsenalLimit: STRESS_TEST_DISASTER_LIMIT,
  },
}

type DifficultyScenarioPreset = Pick<Scenario,
  'shock_probability' | 'severity_min' | 'severity_max' | 'daily_budget'
>

export const DIFFICULTY_DETAILS: Readonly<Record<Difficulty, {
  label: string
  summary: string
  parameters: string
  preset: DifficultyScenarioPreset
}>> = {
  calm: {
    label: 'Calm',
    summary: 'Fewer ambient shocks, gentler conditions, and more daily supply arrivals.',
    parameters: 'Changes only ambient shock probability to 0.10, the ambient severity range to 0.05–0.16, and daily supply arrivals to 220 units. Ambient type weights, authored-event severities, the player severity slider, 20-day horizon, and six-disaster Stress allowance do not change.',
    preset: {
      shock_probability: 0.10,
      severity_min: 0.05,
      severity_max: 0.16,
      daily_budget: 220,
    },
  },
  moderate: {
    label: 'Moderate',
    summary: 'Relay City at its standard operating conditions.',
    parameters: 'Changes only ambient shock probability to 0.20, the ambient severity range to 0.10–0.28, and daily supply arrivals to 180 units. Ambient type weights, authored-event severities, the player severity slider, 20-day horizon, and six-disaster Stress allowance do not change.',
    preset: {
      shock_probability: 0.20,
      severity_min: 0.10,
      severity_max: 0.28,
      daily_budget: 180,
    },
  },
  severe: {
    label: 'Severe',
    summary: 'Frequent hard shocks and fewer daily supply arrivals.',
    parameters: 'Changes only ambient shock probability to 0.34, the ambient severity range to 0.18–0.40, and daily supply arrivals to 140 units. Ambient type weights, authored-event severities, the player severity slider, 20-day horizon, and six-disaster Stress allowance do not change.',
    preset: {
      shock_probability: 0.34,
      severity_min: 0.18,
      severity_max: 0.40,
      daily_budget: 140,
    },
  },
}

export type AuthoredScenarioPreset = {
  id: ScenarioPresetId
  label: string
  summary: string
  disclosedMix: string
  forcedShocks: readonly ForcedShock[]
}

export const SCENARIO_PRESETS: Readonly<Record<ScenarioPresetId, AuthoredScenarioPreset>> = {
  'fault-line': {
    id: 'fault-line',
    label: 'Fault-line city',
    summary: 'Repeated ground motion with one infrastructure follow-on.',
    disclosedMix: 'Earthquake ×2 at raw 0.24 / 0.31 · Utility ×1 at raw 0.22',
    forcedShocks: [
      { day: 3, type: 'aftershock', severity: 0.24 },
      { day: 9, type: 'aftershock', severity: 0.31 },
      { day: 15, type: 'utility', severity: 0.22 },
    ],
  },
  coastal: {
    id: 'coastal',
    label: 'Coastal storm season',
    summary: 'Two weather fronts around one constrained-arrivals episode.',
    disclosedMix: 'Weather ×2 at raw 0.24 / 0.34 · Supply ×1 at raw 0.18',
    forcedShocks: [
      { day: 4, type: 'weather', severity: 0.24 },
      { day: 10, type: 'supply', severity: 0.18 },
      { day: 16, type: 'weather', severity: 0.34 },
    ],
  },
  'supply-corridor': {
    id: 'supply-corridor',
    label: 'Fragile supply corridor',
    summary: 'Repeated freight disruption with utility and healthcare pressure.',
    disclosedMix: 'Supply ×2 at raw 0.22 / 0.32 · Utility ×1 at raw 0.20 · Epidemic ×1 at raw 0.18',
    forcedShocks: [
      { day: 3, type: 'supply', severity: 0.22 },
      { day: 8, type: 'utility', severity: 0.20 },
      { day: 13, type: 'supply', severity: 0.32 },
      { day: 17, type: 'epidemic', severity: 0.18 },
    ],
  },
}

export const TUTORIAL_SEED = 17_008
export const TUTORIAL_SHOCK: Readonly<ForcedShock> = {
  day: 2,
  type: 'weather',
  severity: 0.24,
}

/**
 * Applies a start-screen difficulty without mutating the authored scenario. Scenarios
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
    horizon_days: GAME_HORIZON_DAYS,
    forced_shock: null,
    forced_shocks: [],
  }
}

export function applyAuthoredScenarioPreset(
  scenario: Scenario,
  difficulty: Difficulty,
  presetId: ScenarioPresetId,
): Scenario {
  const difficultyScenario = applyDifficultyPreset(scenario, difficulty, 'start-screen')
  const preset = SCENARIO_PRESETS[presetId]
  return {
    ...difficultyScenario,
    name: `Relay City — ${preset.label}`,
    forced_shocks: preset.forcedShocks.map((shock) => ({ ...shock })),
  }
}

export function createTutorialScenario(scenario: Scenario): Scenario {
  return {
    ...scenario,
    name: 'Relay City — guided weather incident',
    horizon_days: 8,
    daily_budget: 180,
    shock_probability: 0,
    severity_min: 0.10,
    severity_max: 0.28,
    forced_shock: null,
    forced_shocks: [{ ...TUTORIAL_SHOCK }],
  }
}

export function createGameSession(selection: SessionSelection): GameSessionState {
  return {
    mode: selection.mode,
    difficulty: selection.difficulty,
    scenarioPreset: selection.preset ?? null,
    tutorial: false,
    arsenalLimit: MODE_DETAILS[selection.mode].arsenalLimit,
    disastersUsed: 0,
    playerShocks: [],
  }
}

export function createTutorialSession(): GameSessionState {
  return {
    mode: 'sandbox',
    difficulty: 'moderate',
    scenarioPreset: null,
    tutorial: true,
    arsenalLimit: 0,
    disastersUsed: 0,
    playerShocks: [],
  }
}

export function remainingDisasters(session: GameSessionState): number | null {
  if (session.tutorial) return 0
  if (session.arsenalLimit === null) return null
  return Math.max(0, session.arsenalLimit - session.disastersUsed)
}

export function canUseDisaster(session: GameSessionState): boolean {
  if (session.tutorial) return false
  const remaining = remainingDisasters(session)
  return remaining === null || remaining > 0
}

export function recordDisaster(
  session: GameSessionState,
  shock?: ForcedShock,
): GameSessionState {
  if (!canUseDisaster(session)) return session
  return {
    ...session,
    disastersUsed: session.disastersUsed + 1,
    playerShocks: shock ? [...session.playerShocks, { ...shock }] : session.playerShocks,
  }
}

/**
 * Toolbox and restored runs predate session provenance. Player shocks are always
 * appended to forced_shocks, so only that ordered suffix can be labelled as player
 * input. Everything before it (plus the singular legacy slot) remains origin-unknown.
 */
export function storedForcedShocksBeforePlayer(
  scenario: Scenario,
  playerShocks: readonly ForcedShock[],
): ForcedShock[] {
  const additive = scenario.forced_shocks ?? []
  const storedAdditiveCount = Math.max(0, additive.length - playerShocks.length)
  return [
    ...(scenario.forced_shock ? [{ ...scenario.forced_shock }] : []),
    ...additive.slice(0, storedAdditiveCount).map((shock) => ({ ...shock })),
  ]
}
