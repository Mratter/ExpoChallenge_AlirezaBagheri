import type {
  Scenario,
  Service,
  ShockType,
  Vector5,
  Vector22,
} from './generated/backendContract'

export {
  actionGroups,
  actionOrder,
  actionSlices,
  compareRequestFieldOrder,
  defaultCompareRequest,
  defaultScenario,
  environmentContract,
  forcedShockFieldOrder,
  observationOrder,
  requestLimits,
  scenarioFieldOrder,
  services,
  SHOCK_IMPACTS,
  shockTypes,
} from './generated/backendContract'
export type {
  CompareRequest,
  ForcedShock,
  Scenario,
  Service,
  ShockType,
  Vector5,
  Vector22,
} from './generated/backendContract'

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

export type ObservationContract = {
  source: string
  input_name: string
  dtype: string
  shape: [number]
  normalization: string
}

export type PolicyIdentity = {
  id: string
  path_stem: string
  artifact_type: string
  runtime: string
  sha256: string
  observation_contract: ObservationContract
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
  policy: PolicyIdentity
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

export type Metadata = {
  app: string
  version: string
  schema_version: '4.0.0'
  default_seed: number
  services: Service[]
  model: PolicyIdentity & {
    observation_count: 73
    action_count: 22
    observation_order: string[]
    action_order: string[]
    action_groups: string[]
  }
  environment: {
    id: 'CityRecoveryEnv-v3'
    version: '3.0.0'
    observation_count: 73
    action_count: 22
    spec_sha256: string
    policy_neutral_transition: true
    future_tape_visible: false
  }
  outcome_definition: Record<string, unknown>
  outcome_definition_sha256: string
  baseline: {
    id: 'reactive-public-state-heuristic-v3'
    version: '3.0.0'
    uses_same_observation_contract: true
    uses_same_action_contract: true
    uses_public_risk_signal: true
    future_tape_visible: false
  }
  persistence: Record<string, unknown>
  determinism: string
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
