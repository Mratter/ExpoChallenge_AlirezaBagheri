export const services = ['transport', 'housing', 'food', 'healthcare', 'public_services'] as const
export type Service = (typeof services)[number]

export const shockTypes = ['aftershock', 'supply', 'epidemic', 'utility', 'weather'] as const
export type ShockType = (typeof shockTypes)[number]

export const observationOrderV3 = [
  ...services.map((service) => `service_${service}`),
  ...services.map((service) => `priority_${service}`),
  ...services.map((service) => `support_${service}`),
  ...services.map((service) => `shock_impact_${service}`),
  ...services.map((service) => `depot_stock_fraction_${service}`),
  ...services.map((service) => `pending_arrival_pressure_${service}`),
  ...services.map((service) => `throughput_factor_${service}`),
  ...services.map((service) => `depot_damage_penalty_${service}`),
  ...services.map((service) => `depot_damage_days_fraction_${service}`),
  ...services.map((service) => `recovery_target_${service}`),
  ...services.map((service) => `prior_day_critical_streak_fraction_${service}`),
  ...services.map((service) => `preparedness_level_${service}`),
  'available_material_budget_fraction',
  'available_crew_pool_fraction',
  'horizon_remaining_fraction',
  'assessment_tail_active',
  'current_shock_severity',
  'road_capacity',
  'days_since_last_shock_fraction',
  'previous_weighted_resilience',
  ...shockTypes.map((shock) => `public_next_day_risk_${shock}`),
] as const

export const actionOrderV3 = [
  ...services.map((service) => `material_share_${service}`),
  'material_utilization',
  ...services.map((service) => `crew_share_${service}`),
  'crew_utilization',
  ...services.map((service) => `stock_release_${service}`),
  ...services.map((service) => `preparedness_investment_${service}`),
] as const

export const actionGroupsV3 = [
  'material_share',
  'material_utilization',
  'crew_share',
  'crew_utilization',
  'stock_release',
  'preparedness_investment',
] as const

export type Vector5 = [number, number, number, number, number]
export type Vector22 = [
  number, number, number, number, number, number,
  number, number, number, number, number, number,
  number, number, number, number, number,
  number, number, number, number, number,
]

export type ForcedShock = {
  day: number
  type: ShockType
  severity: number
}

/** The exact public CityRecoveryEnv-v3 request contract. */
export type Scenario = {
  name: string
  horizon_days: 30
  daily_budget: number
  initial_services: Vector5
  priorities: Vector5
  shock_probability: number
  severity_min: number
  severity_max: number
  forced_shock: ForcedShock | null
  forced_shocks: ForcedShock[]
  daily_crew_pool: number
  recovery_targets: Vector5
  assessment_tail_days: 3
}

export type Shock = {
  day: number
  type: ShockType | null
  severity: number
  impact: Vector5
  budget_factor: number
  forced: boolean
  clustered: boolean
  cluster_parent_days: number[]
  ambient_occurrence_probability: number
  ambient_occurrence_draw: number
  cluster_hazard: number
  public_risk_next: Vector5
  assessment_tail: boolean
}

export type ProjectionViolationBreakdown = {
  sum_violations: number
  budget_violations: number
  lower_violations: number
  upper_violations: number
  total: number
}

export type ProjectionReceipt = {
  distance: number
  sum: number
  constraint_violations: number
  violation_breakdown: ProjectionViolationBreakdown
  bindings: Array<{ service: Service; lower: boolean; upper: boolean }>
}

export type MutualAidTransfer = {
  from_service: Service
  to_service: Service
  units: number
  donor_stock_fraction_before: number
  receiver_stock_fraction_before: number
}

/** The complete physical conservation ledger returned for each planner-day. */
export type LogisticsLedger = {
  depot_capacity: Vector5
  depot_stock_before: Vector5
  pending_arrivals: Vector5
  pending_arrivals_landed: Vector5
  pending_arrivals_held: Vector5
  depot_stock_after_pending: Vector5
  depot_damage_penalty: Vector5
  depot_damage_days_remaining: Vector5
  depot_damage_factor: Vector5
  road_capacity: number
  throughput_factor: Vector5
  mutual_aid_transfers: MutualAidTransfer[]
  mutual_aid_net: Vector5
  depot_stock_ready: Vector5
  preparedness_material_requested: Vector5
  preparedness_material_consumed: Vector5
  preparedness_effective_work: Vector5
  depot_stock_after_preparedness: Vector5
  repair_material_committed: Vector5
  preparedness_crew_assigned: Vector5
  preparedness_crew_utilized: Vector5
  preparedness_crew_capacity_effective: Vector5
  preparedness_crew_capacity_physical: Vector5
  repair_crew_assigned: Vector5
  same_day_delivery_scheduled: Vector5
  same_day_delivery_landed: Vector5
  same_day_delivery_held: Vector5
  delayed_delivery_scheduled: Vector5
  repair_reserve: Vector5
  repair_request: Vector5
  total_stock_release_budget: Vector5
  stock_release_remaining_after_preparedness: Vector5
  stock_release_limit: Vector5
  crew_capacity_effective: Vector5
  crew_capacity_physical: Vector5
  repair_dispatch: Vector5
  repair_supply: Vector5
  spoilage: Vector5
  depot_stock_end: Vector5
  pending_next_day: Vector5
  capacity_overflow: Vector5
  conservation_residual: Vector5
}

export type OfficialOutcomeChecks = {
  zero_hard_violations: boolean
  conservation_verified: boolean
  assessment_tail_targets_met: boolean
  resilience_auc_met: boolean
  critical_service_day_cap_met: boolean
  terminal_pending_within_capacity: boolean
}

/** Authoritative server verdict. The client never reclassifies this object. */
export type OfficialOutcome = {
  definition_id: string
  definition_sha256: string
  solved: boolean
  status: 'solved' | 'failed'
  checks: OfficialOutcomeChecks
  recovery_targets: Vector5
  target_met_by_service: [boolean, boolean, boolean, boolean, boolean]
  assessment_tail_days: 3
  tail_minimum_services: Vector5
  resilience_auc: number
  resilience_auc_floor: number
  critical_service_days: number
  critical_service_day_cap: number
  hard_violation_count: number
  max_conservation_residual: number
  terminal_pending_arrivals: Vector5
  terminal_pending_capacity: Vector5
  reason_codes: string[]
}

export type DayResult = {
  day: number
  shock: Shock
  available_budget: number
  available_crew: number
  material_used: number
  material_unspent: number
  crew_used: number
  crew_idle: number
  services_before: Vector5
  services_after_shock: Vector5
  raw_action: Vector22
  allocation: Vector5
  material_allocation: Vector5
  crew_allocation: Vector5
  stock_release: Vector5
  preparedness_requested: Vector5
  preparedness_investment: Vector5
  preparedness_before: Vector5
  preparedness_after_hazard: Vector5
  preparedness_gain_requested: Vector5
  preparedness_gain: Vector5
  preparedness_end: Vector5
  preparedness_alignment_reward: number
  backlog_pressure: number
  lower_bounds: Vector5
  upper_bounds: Vector5
  crew_lower_bounds: Vector5
  crew_upper_bounds: Vector5
  projection: ProjectionReceipt
  crew_projection: ProjectionReceipt
  planner_evidence: Record<string, unknown> | null
  support: Vector5
  throughput: Vector5
  public_next_day_risk: Vector5
  gain: Vector5
  strain: Vector5
  services_end: Vector5
  resilience: number
  reward: number
  hard_violation_count: number
  hard_violation_breakdown: Record<string, unknown>
  logistics: LogisticsLedger
  terminal_bonus?: number
  absolute_outcome?: OfficialOutcome
}

export type PlannerResult = {
  planner: string
  rauc: number
  final_resilience: number
  minimum_resilience: number
  post_shock_recovery_shortfall_auc: number
  days_to_pre_shock_recovery_after_largest_loss: number
  largest_shock_loss_day: number
  critical_service_days: number
  hard_violation_count: number
  constraint_violations: number
  max_logistics_conservation_residual: number
  final_depot_stock: Vector5
  final_pending_arrivals: Vector5
  absolute_outcome: OfficialOutcome
  trajectory_sha256: string
  trajectory: DayResult[]
}

export type CompareResponse = {
  schema_version: '4.0.0'
  engine_version: 'city-recovery-env-v3'
  result_id: string
  persistence: { format: string; idempotent: boolean; result_id: string }
  environment: {
    id: 'CityRecoveryEnv-v3'
    version: string
    observation_count: 73
    action_count: 22
    spec_sha256: string
  }
  engine_spec: Record<string, unknown>
  engine_spec_sha256: string
  outcome_definition: Record<string, unknown>
  outcome_definition_sha256: string
  seed: number
  generator: string
  scenario: Scenario
  services: Service[]
  observation_order: string[]
  action_order: string[]
  shock_schedule: Shock[]
  shock_schedule_sha256: string
  policy: {
    id: string
    artifact_type: string
    algorithm: string
    runtime: string
    sha256: string
    sb3_checkpoint_sha256: string
    manifest_sha256: string
  }
  baseline_spec: {
    id: string
    version: string
    uses_same_observation_contract: boolean
    uses_same_action_contract: boolean
    uses_public_risk_signal: boolean
    future_tape_visible: boolean
  }
  baseline: PlannerResult
  candidate: PlannerResult
  comparison: {
    primary_metric: 'independent_absolute_disaster_solved'
    candidate_solved: boolean
    baseline_solved: boolean
    absolute_outcome_pair: 'both_solved' | 'ppo_only' | 'heuristic_only' | 'neither'
    secondary_rauc_candidate_minus_baseline: number
  }
}

export type BenchmarkCandidateSummary = {
  id: 'city-recovery-sb3-ppo-v3-selected'
  solved_count: number
  failed_count: number
  solve_rate: number
  solve_rate_wilson_ci95: [number, number]
  mean_resilience_auc: number
}

export type BenchmarkBaselineSummary = {
  id: 'reactive-public-state-heuristic-v3'
  version: '3.0.0'
  solved_count: number
  failed_count: number
  solve_rate: number
  solve_rate_wilson_ci95: [number, number]
  mean_resilience_auc: number
}

export type VerifiedBenchmark = {
  artifact_sha256: string
  artifact_type: 'city_recovery_v3_final_benchmark'
  status: 'complete'
  primary_metric: 'independent_absolute_disasters_solved'
  synthetic_case_count: 40
  candidate: BenchmarkCandidateSummary
  baseline: BenchmarkBaselineSummary
  paired_absolute_outcomes: {
    both_solved: number
    ppo_only: number
    heuristic_only: number
    neither: number
  }
  secondary_head_to_head_resilience_auc: {
    candidate_wins: number
    baseline_wins: number
    ties: number
    candidate_mean_minus_baseline_mean: number
  }
  invariants: {
    all_rows_present: boolean
    unique_rows: boolean
    same_tapes: boolean
    deterministic_replay_mismatches: number
    candidate_hard_violations: number
    baseline_hard_violations: number
    maximum_conservation_residual: number
  }
  rows_sha256: string
  ledger_sha256: string
}

export type MetadataV3 = {
  app: string
  version: string
  schema_version: '4.0.0'
  commit: string
  profile: string
  default_seed: number
  services: Service[]
  model: {
    id: 'city-recovery-sb3-ppo-v3-selected'
    version: '3.0.0'
    schema_version: '4.0.0'
    manifest_schema_version: 1
    artifact_type: 'stable_baselines3_ppo'
    algorithm: 'PPO'
    training_library: 'stable-baselines3'
    trainable_parameters: number
    requested_training_transitions: number
    completed_training_transitions: number
    observation_count: 73
    action_count: 22
    observation_order: string[]
    action_order: string[]
    action_groups: string[]
    license: string
    source: string
    onnx_sha256: string
    selected_checkpoint_sha256: string
    manifest_sha256: string
    parity_sha256: string
    scientific_source_sha256: string
  }
  environment: {
    id: 'CityRecoveryEnv-v3'
    version: '3.0.0'
    observation_count: 73
    action_count: 22
    spec_sha256: string
    outcome_definition_sha256: string
    policy_neutral_transition: true
    future_tape_visible: false
  }
  outcome_definition: Record<string, unknown>
  baseline: {
    id: 'reactive-public-state-heuristic-v3'
    version: '3.0.0'
    uses_same_observation_contract: true
    uses_same_action_contract: true
    uses_public_risk_signal: true
    future_tape_visible: false
  }
  benchmark: VerifiedBenchmark
  dataset: Record<string, unknown>
  persistence: Record<string, unknown>
}

export type SavedResultSummary = {
  result_id: string
  schema_version: string
  engine_version: 'city-recovery-env-v3'
  seed: number
  scenario_name: string
  horizon_days: 30
  primary_metric: string
  candidate_solved: boolean
  baseline_solved: boolean
  candidate_rauc: number
  baseline_rauc: number
  outcome: CompareResponse['comparison']['absolute_outcome_pair']
  policy_sha256: string
}
