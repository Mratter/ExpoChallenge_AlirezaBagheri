import { describe, expect, it } from 'vitest'
import { toolboxPresets } from './generated/toolboxPresets'
import { benchmarkDisplayLabel, reactiveHeuristicName } from './plannerNames'
import { defaultScenario, environmentContract, requestLimits, services } from './types'

const outcomes = ['both', 'ppo_only', 'neither'] as const

describe('toolbox preset scenarios', () => {
  const { presets, summary } = toolboxPresets

  it('ships the documented ten-case mix', () => {
    expect(presets).toHaveLength(10)
    expect(summary).toEqual({ total: 10, ppoSolved: 8, heuristicSolved: 3, neitherSolved: 2 })
    const byOutcome = Object.fromEntries(
      outcomes.map((outcome) => [outcome, presets.filter((preset) => preset.outcome === outcome).length]),
    )
    expect(byOutcome).toEqual({ both: 3, ppo_only: 5, neither: 2 })
  })

  it('records the verdicts the runtime actually returned', () => {
    expect(presets.filter((preset) => preset.observed.candidateSolved)).toHaveLength(summary.ppoSolved)
    expect(presets.filter((preset) => preset.observed.baselineSolved)).toHaveLength(summary.heuristicSolved)
    for (const preset of presets) {
      const { candidateSolved, baselineSolved } = preset.observed
      const observed = candidateSolved && baselineSolved ? 'both' : candidateSolved ? 'ppo_only' : baselineSolved ? 'heuristic_only' : 'neither'
      expect(observed).toBe(preset.outcome)
    }
  })

  it('never claims a case the heuristic solved alone', () => {
    // The stated mix is "every heuristic solve is also a PPO solve"; a
    // heuristic-only preset would silently contradict the picker's summary.
    expect(presets.filter((preset) => preset.observed.baselineSolved && !preset.observed.candidateSolved)).toEqual([])
  })

  it('keeps every preset inside the request contract the backend enforces', () => {
    const seen = new Set<string>()
    for (const preset of presets) {
      expect(seen.has(preset.id)).toBe(false)
      seen.add(preset.id)
      const scenario = preset.scenario
      expect(scenario.name).toBe(preset.label)
      expect(scenario.name.length).toBeLessThanOrEqual(requestLimits.name.maximumLength)
      expect(scenario.horizon_days).toBe(defaultScenario.horizon_days)
      expect(scenario.assessment_tail_days).toBe(environmentContract.assessmentTailDays)
      expect(preset.seed).toBeGreaterThanOrEqual(requestLimits.seed.minimum)
      expect(preset.seed).toBeLessThanOrEqual(requestLimits.seed.maximum)
      expect(scenario.daily_budget).toBeGreaterThanOrEqual(requestLimits.dailyBudget.minimum)
      expect(scenario.daily_budget).toBeLessThanOrEqual(requestLimits.dailyBudget.maximum)
      expect(scenario.daily_crew_pool).toBeGreaterThanOrEqual(requestLimits.dailyCrewPool.minimum)
      expect(scenario.daily_crew_pool).toBeLessThanOrEqual(requestLimits.dailyCrewPool.maximum)
      expect(scenario.shock_probability).toBeLessThanOrEqual(requestLimits.shockProbability.maximum)
      expect(scenario.severity_min).toBeLessThanOrEqual(scenario.severity_max)
      for (const vector of [scenario.initial_services, scenario.priorities, scenario.recovery_targets]) {
        expect(vector).toHaveLength(services.length)
      }
    }
  })

  it('is bound to the shipped policy and outcome definition', () => {
    expect(toolboxPresets.policySha256).toMatch(/^[0-9a-f]{64}$/)
    expect(toolboxPresets.outcomeDefinitionSha256).toMatch(/^[0-9a-f]{64}$/)
    for (const preset of presets) {
      expect(preset.observed.shockScheduleSha256).toMatch(/^[0-9a-f]{64}$/)
    }
  })

  it('only offers presets whose class is typical of their configuration', () => {
    // A preset that only lands in its class on one lucky seed would misrepresent
    // the configuration it is named after.
    for (const preset of presets) {
      expect(preset.classShareFirst25Seeds).toBeGreaterThanOrEqual(9)
    }
  })
})

describe('planner display names', () => {
  it('names the baseline as a tuned rule', () => {
    expect(reactiveHeuristicName).toBe('Fine tuned reactive heuristic')
  })

  it('maps the frozen retrospective label without altering other rows', () => {
    expect(benchmarkDisplayLabel('Reactive heuristic')).toBe(reactiveHeuristicName)
    expect(benchmarkDisplayLabel('v4 PPO (shipped)')).toBe('v4 PPO (shipped)')
    expect(benchmarkDisplayLabel('Tuned constant rule')).toBe('Tuned constant rule')
  })
})
