export const services = ['transport', 'housing', 'food', 'healthcare', 'public_services'] as const
export type Service = (typeof services)[number]
export const shockTypes = ['aftershock', 'supply', 'epidemic', 'utility', 'weather'] as const
export type ShockType = (typeof shockTypes)[number]

export type ForcedShock = {
  day: number
  type: ShockType
  severity: number
}
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
  forced_shocks?: ForcedShock[]
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
  raw_action: number[] | null
  raw_proposal: number[]
  lower_bounds: number[]
  upper_bounds: number[]
  allocation: number[]
  projection: {
    distance: number
    sum: number
    constraint_violations: number
    violation_breakdown: {
      sum_violations: number
      budget_violations: number
      lower_violations: number
      upper_violations: number
      total: number
    }
    bindings: { service: Service; lower: boolean; upper: boolean }[]
  }
  planner_evidence: Record<string, unknown> | null
  support: number[]
  gain: number[]
  strain: number[]
  services_end: number[]
  resilience: number
  reward: number
}

export type PlannerResult = {
  planner: string
  rauc: number
  final_resilience: number
  minimum_resilience: number
  post_shock_recovery_shortfall_auc: number
  days_to_pre_shock_recovery_after_largest_loss: number
  critical_service_days: number
  total_projection_distance: number
  constraint_violations: number
  trajectory: DayResult[]
}

export type CompareResponse = {
  schema_version: string
  result_id: string
  persistence: { format: string; idempotent: boolean; result_id: string }
  seed: number
  generator: string
  scenario: Scenario
  services: Service[]
  shock_schedule: Shock[]
  shock_schedule_sha256: string
  policy: {
    id: string
    artifact_type: string
    algorithm: string
    runtime: string
    sha256: string
    sb3_checkpoint_sha256: string
    parity_report_sha256: string
    disclosure: string
    legacy_candidate: {
      id: string
      artifact_type: string
      is_ppo: boolean
      sha256: string
      disclosure: string
    }
  }
  baseline_spec: {
    id: string
    library: string
    library_version: string
    solver: string
    objective: string
    future_shocks_visible: boolean
  }
  baseline: PlannerResult
  candidate: PlannerResult
  comparison: {
    primary_metric: string
    candidate_minus_baseline: number
    outcome: string
  }
  recommendations: RunRecommendations
  limitations: string[]
}

export type DailyRecommendation = {
  day: number
  priority_service: Service
  priority_rationale: string
  risk_alerts: { service: Service; level: 'critical' | 'strained' | 'district_dark'; detail: string }[]
  allocation_focus: Service
  allocation_focus_share: number
}

export type RunRecommendations = {
  winner: 'candidate' | 'baseline' | 'tie'
  winner_label: string
  winner_margin_pp: number
  winner_rationale: string
  critical_moment: { day: number; resilience: number; description: string }
  most_fragile_service: Service
  most_fragile_days_below_threshold: number
  worst_shock_type: string
  strategy_summary: string
  actionable_recommendations: string[]
  daily: DailyRecommendation[]
}

export type SavedResultSummary = {
  result_id: string
  seed: number
  scenario_name: string
  horizon_days: number
  candidate_rauc: number
  baseline_rauc: number
  outcome: string
  policy_sha256: string
}
