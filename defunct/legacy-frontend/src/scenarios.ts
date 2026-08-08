import type { Scenario } from './types'

export const defaultScenario: Scenario = {
  name: 'Relay City central restart',
  horizon_days: 14,
  daily_budget: 180,
  initial_services: [0.34, 0.26, 0.41, 0.38, 0.3],
  priorities: [1, 1.1, 1.2, 1.4, 1],
  shock_probability: 0.2,
  severity_min: 0.1,
  severity_max: 0.28,
  forced_shock: { day: 5, type: 'utility', severity: 0.26 },
  forced_shocks: [],
}

export const defaultSeed = 424242
