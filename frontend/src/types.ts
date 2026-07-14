export const services = ['transport', 'housing', 'food', 'healthcare', 'public_services'] as const
export type Service = (typeof services)[number]

export type ForcedShock = { day: number; type: 'utility'; severity: number }
export type Scenario = {
  name: string
  horizon_days: number
  daily_budget: number
  initial_services: number[]
  priorities: number[]
  shock_probability: number
  severity_min: number
  severity_max: number
  forced_shock: ForcedShock | null
}

export type Shock = {
  day: number
  type: string | null
  severity: number
  impact: number[]
  budget_factor: number
  forced: boolean
}

export type DayResult = {
  day: number
  shock: Shock
  available_budget: number
  services_before: number[]
  services_after_shock: number[]
  raw_proposal: number[]
  allocation: number[]
  projection: {
    distance: number
    sum: number
    constraint_violations: number
    bindings: { service: Service; lower: boolean; upper: boolean }[]
  }
  support: number[]
  gain: number[]
  strain: number[]
  services_end: number[]
  resilience: number
}

export type PlannerResult = {
  planner: string
  rauc: number
  final_resilience: number
  minimum_resilience: number
  total_projection_distance: number
  constraint_violations: number
  trajectory: DayResult[]
}

export type CompareResponse = {
  schema_version: string
  seed: number
  generator: string
  scenario: Scenario
  services: Service[]
  shock_schedule: Shock[]
  shock_schedule_sha256: string
  policy: { id: string; artifact_type: string; sha256: string; disclosure: string }
  baseline: PlannerResult
  candidate: PlannerResult
  comparison: {
    primary_metric: string
    candidate_minus_baseline: number
    outcome: string
  }
  limitations: string[]
}
