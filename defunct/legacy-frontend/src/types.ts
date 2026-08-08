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
  clustered?: boolean
  cluster_parent_days?: number[]
  ambient_occurrence_probability?: number
  ambient_occurrence_draw?: number
  cluster_hazard?: number
}

/**
 * Engine-v2's per-day physical logistics ledger. The field is optional on a
 * day because released schema-v2 results must continue to render exactly as
 * the legacy, presentation-derived view; schema-v3 comparisons always include
 * the complete ledger.
 */
export type LogisticsLedger = {
  depot_capacity: number[]
  depot_stock_before: number[]
  pending_arrivals: number[]
  pending_arrivals_landed: number[]
  pending_arrivals_held: number[]
  depot_stock_after_pending: number[]
  depot_damage_penalty: number[]
  depot_damage_days_remaining: number[]
  depot_damage_factor: number[]
  road_capacity: number
  throughput_factor: number[]
  mutual_aid_transfers: Array<{
    from_service: Service
    to_service: Service
    units: number
    donor_stock_fraction_before: number
    receiver_stock_fraction_before: number
  }>
  mutual_aid_net: number[]
  depot_stock_ready: number[]
  pending_next_day: number[]
  same_day_delivery_scheduled: number[]
  same_day_delivery_landed: number[]
  same_day_delivery_held: number[]
  delayed_delivery_scheduled: number[]
  repair_reserve: number[]
  repair_request: number[]
  repair_dispatch: number[]
  repair_supply: number[]
  spoilage: number[]
  depot_stock_end: number[]
  capacity_overflow: number[]
  conservation_residual: number[]
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
  throughput?: number[]
  gain: number[]
  strain: number[]
  services_end: number[]
  resilience: number
  reward: number
  logistics?: LogisticsLedger
}

export type PlannerResult = {
  planner: string
  rauc: number
  final_resilience: number
  minimum_resilience: number
  post_shock_recovery_shortfall_auc: number
  days_to_pre_shock_recovery_after_largest_loss: number
  largest_shock_loss_day?: number
  critical_service_days: number
  total_projection_distance: number
  constraint_violations: number
  violation_breakdown?: {
    sum_violations: number
    budget_violations: number
    lower_violations: number
    upper_violations: number
  }
  total_food_spoilage?: number
  total_mutual_aid_units?: number
  max_logistics_conservation_residual?: number
  final_depot_stock?: number[]
  final_pending_arrivals?: number[]
  trajectory_sha256?: string
  trajectory: DayResult[]
}

export type CompareResponse = {
  schema_version: string
  engine_version?: string
  result_id: string
  persistence: { format: string; idempotent: boolean; result_id: string }
  seed: number
  generator: string
  scenario: Scenario
  services: Service[]
  shock_schedule: Shock[]
  shock_schedule_sha256: string
  observation_order?: string[]
  action_order?: string[]
  engine_spec?: Record<string, unknown>
  engine_spec_sha256?: string
  environment?: {
    id: string
    version: string
    observation_count: number
    action_count: number
    spec_sha256?: string
  }
  policy: {
    id: string
    artifact_type: string
    algorithm: string
    runtime: string
    sha256: string
    sb3_checkpoint_sha256: string
    parity_report_sha256: string
    disclosure: string
    predecessor_policy?: {
      id: string
      version: string
      onnx_sha256: string
      preserved: boolean
    }
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
    engine_spec_sha256?: string
  }
  baseline: PlannerResult
  candidate: PlannerResult
  comparison: {
    primary_metric: string
    candidate_minus_baseline: number
    recovery_shortfall_candidate_minus_baseline?: number
    recovery_days_candidate_minus_baseline?: number
    outcome: string
  }
  limitations: string[]
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

export type AuditActor = {
  operator_id?: string
  session_label?: string
}

export type CheckpointSelectionProvenance = {
  schema_version: '1.0.0'
  evaluation_manifest_sha256: string
  binding_scope: 'checkpoint_selection_only'
  final_protocol_executed: false
  checkpoint_artifact_path: 'artifacts/city_recovery_ppo.v2.onnx'
  checkpoint_sha256: string
  source_result_id: string
  source_result_sha256: string
  source_result_schema_version: '3.0.0'
  source_engine_spec_sha256: string
  source_result_policy_sha256: string
  source_result_policy_verified: true
  source_result_integrity_verified: true
  simulator_version_hash: string
  verification_status: 'verified'
}

export type PlanningSession = {
  schema_version: '1.0.0'
  session_id: string
  label: string
  source_result_id: string
  mode: 'simulation'
  simulation_only: true
  simulation_auto_execute: boolean
  simulator_version_hash: string
  checkpoint_selection_provenance: CheckpointSelectionProvenance
  created_at: string
}

export type OperatorPlanDay = {
  day: number
  available_budget: number
  original_policy_proposal: number[]
  original_feasible_allocation: number[]
  lower_bounds: number[]
  upper_bounds: number[]
}

export type OperatorProjectionDiagnostics = {
  already_feasible: boolean
  solver_rescue: boolean
  proposal_sum: number
  projected_sum: number
  l1_distance: number
  l2_distance: number
  fraction_proposed_allocation_modified: number
  allocations_changed: number
  changed_indices: number[]
  rank_correlation: number | null
  material_reranking: boolean
  scaled_only: boolean
  materially_changed_structure: boolean
  constraint_categories: string[]
  safety_slack_before: number
  safety_slack_after: number
  binding_constraints: { service: Service; lower: boolean; upper: boolean }[]
  fallback_required: false
}

export type ReprojectedPlanDay = {
  day: number
  proposal_source: 'original_feasible_plan' | 'operator_override'
  proposal: number[]
  feasible_allocation: number[]
  diagnostics: OperatorProjectionDiagnostics
}

export type PlanStatus =
  | 'proposed'
  | 'reviewed'
  | 'approved'
  | 'rejected'
  | 'overridden'
  | 'reprojected'
  | 'executed'

export type PlanRecord = {
  schema_version: '1.0.0'
  plan_id: string
  session_id: string
  source_result_id: string
  status: PlanStatus
  version: number
  simulation_only: true
  simulation_auto_execute: boolean
  simulator_version_hash: string
  checkpoint_selection_provenance: CheckpointSelectionProvenance
  original_plan: OperatorPlanDay[]
  operator_overrides: Array<{
    actor: AuditActor
    reason: string
    timestamp: string
    changes: Array<{ day: number; proposal: number[] }>
  }>
  reprojected_plan: ReprojectedPlanDay[]
  execution_id: string | null
  audit_events: Array<{
    sequence: number
    action: PlanStatus
    from_status: PlanStatus | null
    to_status: PlanStatus
    actor: AuditActor
    timestamp: string
    reason: string
    version: number
    simulator_version_hash: string
    checkpoint_selection_provenance: CheckpointSelectionProvenance
  }>
  created_at: string
  updated_at: string
}

export type ExecutedPlan = {
  plan: PlanRecord
  execution: {
    schema_version: '1.0.0'
    execution_id: string
    plan_id: string
    session_id: string
    simulation_only: true
    auto_approved: boolean
    simulator_version_hash: string
    checkpoint_selection_provenance: CheckpointSelectionProvenance
    live_binding_reverified: true
    binding_verified_at: string
    executed_at: string
    actor: AuditActor
    reason: string
  }
}

export type JudgeBranchMode =
  | 'open_loop_original_learned'
  | 'replan_baseline'
  | 'replan_learned'

export type JudgeBranchAggregate = {
  episode_count: number
  mean_cumulative_critical_service_days_lost: number
  cvar_10_weighted_unmet_need: number
  mean_worst_district_service_days_lost: number
  maximum_worst_district_service_days_lost: number
  hard_resource_constraint_violations: number
  mcse_across_matched_rollouts: Record<string, number | null>
}

export type JudgeBranchProof = {
  branch_state_sha256_before_tail: string
  branch_state_sha256_after_tail: string
  conditional_tail_sha256: string
  full_tape_sha256: string
}

export type JudgeProjectionDiagnostics = {
  method?: string
  distance: number
  sum: number
  constraint_violations: number
  violation_breakdown: Record<string, number>
  bindings: Record<string, number>
}

export type JudgeTrajectoryDay = {
  day: number
  raw_priorities: number[][]
  raw_reserve_head: number
  allocation: number[][]
  services_end: number[][]
  service_pool: number
  projection: JudgeProjectionDiagnostics
  hard_resource_constraint_violations: number
}

export type JudgeProposalAudit = {
  day: number
  policy_id: string
  policy_called: boolean
  priorities: number[][]
  reserve_head: number
  proposal_sha256: string
}

export type JudgePolicyOutcome = {
  policy_id: string
  synthetic_proxy: true
  trajectory: JudgeTrajectoryDay[]
  trajectory_sha256: string
  proposal_audit: JudgeProposalAudit[]
  domain_metrics: Record<string, unknown>
}

export type JudgeInitialOutcome = JudgePolicyOutcome & {
  tape_sha256: string
  proposal_schedule_sha256: string
  first_day_decision: {
    day: number
    raw_priorities: number[][]
    raw_reserve_head: number
    projected_allocation: number[][]
    projection_diagnostics: JudgeProjectionDiagnostics
    service_pool: number
  }
}

export type JudgeDemoReport = {
  schema_version: string
  demo_id: string
  result_class: 'preliminary'
  claim_eligible: false
  synthetic_proxy: true
  inputs: {
    held_out_seed: number
    learned_training_seed: number
    matched_rollout_seeds: number[]
    prefix_days: number
    active_branch_day: number
  }
  provenance: {
    simulator_version_hash: string
    evaluation_manifest_sha256: string
    learned_policy: { policy_id: string; sha256: string; training_seed: number }
    baseline_policy: { policy_id: string; sha256: string }
  }
  initial_comparison: {
    matched_initial_tape: true
    tape_sha256: string
    baseline: JudgeInitialOutcome
    learned: JudgeInitialOutcome
  }
  branch_point: {
    prefix_days: number
    active_day: number
    snapshot_sha256: string
    branch_state_sha256: string
    explicit_active_disruption: {
      type: string | null
      severity: number
      focus_district: number | null
      synthetic_proxy: true
    }
  }
  conditional_rollouts: Array<{
    rollout_seed: number
    matched_proof: JudgeBranchProof
    branches: Record<JudgeBranchMode, JudgePolicyOutcome & {
      mode: JudgeBranchMode
      replanning: boolean
      open_loop_proposal_replay: boolean
      proof: JudgeBranchProof
    }>
  }>
  aggregates: Record<JudgeBranchMode, JudgeBranchAggregate>
  counterfactual: {
    narrative: string
    interpretation_boundary: string
    selected_action: {
      name: string
      day: number
      district_id: string
      service: Service
      raw_priority: number
      projected_allocation: number
    }
    feasible_alternative: {
      name: string
      intervention: string
      raw_priority: number
      projected_allocation: number
      feasibility_verified: true
      post_projection_constraint_violations: number
    }
    matched_rollout_seeds: number[]
    immediate_metric_difference: {
      metric: string
      selected_minus_alternative: number
    }
    long_term_metric_difference: {
      metric: string
      mean_selected_minus_alternative: number
    }
    cvar_difference: {
      metric: string
      selected_minus_alternative: number
    }
    named_action_removal_proof: {
      actual_matched_rerun: true
      caller_assertion_used: false
    }
    geographic_dependency_path: { district_ids: string[]; path_exists: true }
    service_dependency_path: {
      from_service: Service
      to_service: Service
      dependency_weight: number
      path_exists: true
    }
  }
  claim_status_rows: Array<{
    claim: string
    evidence_status: 'supported' | 'inconclusive' | 'not_demonstrated'
    result_class: 'preliminary' | 'final'
    claim_eligible: boolean
  }>
  disclosures: string[]
  report_sha256: string
}

export type JudgeDemoResponse = {
  schema_version: '1.0.0'
  verification_status: 'verified'
  result_class: 'preliminary'
  claim_eligible: false
  artifact_manifest_sha256: string
  report: JudgeDemoReport
}

export const v5Sectors = [
  'transport',
  'housing',
  'food',
  'healthcare',
  'public_services',
  'resilience',
] as const
export type V5Sector = (typeof v5Sectors)[number]

export type V5SimulatorProfile = {
  profile_id: string
  label: string
  horizon_days: number
  district_count: number
  dry_run_only: boolean
  simulator_version_hash: string
}

export type V5ProfilesResponse = {
  schema_version: 'v5-development-dashboard-v1'
  simulator_version: string
  default_profile_id: string
  profiles: V5SimulatorProfile[]
  disclaimer: string
}

export type V5DevelopmentSnapshot = {
  schema_version: 'v5-development-dashboard-v1'
  result_class: 'development_snapshot'
  disclaimer: string
  simulator: {
    version: string
    profile_id: string
    profile_label: string
    simulator_version_hash: string
    horizon_days: number
    dry_run_only: boolean
    seed: number
    scenario_id: string
  }
  action_schema: {
    version: string
    sha256: string
    output_count: 55
    order: string[]
  }
  current_policy: {
    policy_id: string
    kind: string
    checkpoint_identity: string | null
  }
  model: {
    candidate_id: string
    trainable_parameter_count: number
    checkpoint_status: string
    checkpoint_identity: string | null
    selected_for_final: boolean
  }
  strategic_action: {
    status: 'Proposed'
    latent_output_count: 55
    latent_vector: number[]
    sector_names: V5Sector[]
    district_ids: string[]
    sector_shares: number[]
    district_shares: number[][]
    desired_allocation: number[][]
    spendable_resources: number
    unallocated_amount: number
    interpretation: string
  }
  carrying_reserve: {
    proposed_fraction: number
    proposed_amount: number
    inventory_before: number
    executed_inventory_after: number | null
    status: string
  }
  dependency_graph: {
    topology_version: string
    topology_sha256: string
    node_count: number
    directed_edge_count: number
    critical_edge_count: number
    relation_counts: Record<'dependency' | 'geographic' | 'supply' | 'redundancy', number>
    edge_orientation: string
  }
  projects: {
    mode_counts: Record<'temporary' | 'permanent' | 'resilience', number>
    top_proposals: Array<{
      project_id: string
      mode: string
      sector: V5Sector
      district_id: string
      proposed_amount: number
      status: 'Proposed'
    }>
    current_lifecycle_status: string
    lifecycle_statuses: string[]
    execution_status: string
  }
  forecast: {
    event_probability: number
    confidence: number
    severity_low: number
    severity_mean: number
    severity_high: number
    aid_arrival_day_low: number
    aid_arrival_day_high: number
    expected_aid_fraction: number
    uncertainty: number
    status: string
  }
  displacement_and_equity: {
    rows: Array<{
      district_id: string
      population: number
      displaced_people: number
      displaced_fraction: number
      observed_need_proxy: number
      critical_observed_need_proxy: number
      vulnerability: number
    }>
    total_displaced_people: number
    total_displaced_fraction: number
    worst_district: {
      district_id: string
      metric: string
      value: number
      scope: string
    }
  }
  projection: {
    distance: number | null
    status: string
    reason: string
  }
  policy_comparison: Array<{
    family: 'Baseline' | 'MPC' | 'Learned'
    policy: string
    status: string
    endpoint_metrics: Record<string, number> | null
    reason: string
  }>
  evidence: {
    status: string
    pilot: {
      label: string
      result_available: false
      metrics: null
    }
    final: {
      label: string
      result_available: false
      metrics: null
      campaign_started: false
      configuration_dry_run_only: true
    }
  }
}

export type V5PlanStatus =
  | 'proposed'
  | 'reviewed'
  | 'approved'
  | 'rejected'
  | 'overridden'
  | 'reprojected'
  | 'executed'

export type V5PlanComparisonRow = {
  project_id: string
  district_index: number
  sector_index: number
  raw_proposal: number
  solver_projection: number
  operator_requested: number
  reprojected: number | null
  approved: number | null
}

export type V5ApprovedPlan = {
  approval_id: string
  feasible_action_sha256: string
  approved_action_sha256: string
  reserve_amount: number
  allocations: Array<{ project_id: string; amount: number }>
}

export type V5OperatorPlan = {
  schema_version: 'v5-operator-lifecycle-v1'
  plan_id: string
  version: number
  status: V5PlanStatus
  execution_mode: 'approval_required' | 'simulation_only_auto_execute'
  approval_required: boolean
  created_at: string
  simulator: {
    version: string
    profile_id: string
    seed: number
    scenario_id: string
    scenario_sha256: string
    simulator_version_hash: string
  }
  raw_proposal: {
    latent_action_sha256: string
    project_proposal_sha256: string
    reserve_amount: number
    unallocated_amount: number
    allocations: Array<{ project_id: string; amount: number; maximum_amount: number }>
  }
  solver_projection: {
    feasible_action_sha256: string
    solver: string
    projection_distance: number
    reserve_amount: number
    unallocated_amount: number
    solver_change: {
      allocation_l1_distance: number
      allocation_l2_distance: number
      modified_allocation_fraction: number
      constraint_categories: string[]
      change_class: 'none' | 'scale_only' | 'structural'
      fallback_required: boolean
    }
  }
  operator_changes: Array<{
    target: string
    before: number
    requested_after: number
    operator_id: string
    session_id: string
    reason: string
    timestamp: string
    change_sha256: string
  }>
  current_reprojection: {
    feasible_action_sha256: string
    projection_distance: number
    reserve_amount: number
    unallocated_amount: number
  } | null
  approved_plan: V5ApprovedPlan | null
  reprojected_approved_plan: V5ApprovedPlan | null
  solver_change_evidence: {
    initial_projection_sha256: string
    reprojections: Array<Record<string, unknown>>
  }
  comparison: V5PlanComparisonRow[]
  audit_events: Array<{
    sequence: number
    action: string
    status: V5PlanStatus
    operator_id: string
    session_id: string
    reason: string
    timestamp: string
    evidence_sha256: string
    previous_event_sha256: string
    event_sha256: string
  }>
  execution: {
    transition_id: string
    approved_action_sha256: string
    state_before_sha256: string
    state_after_sha256: string
    record_sha256: string
  } | null
  actions_available: Array<'review' | 'approve' | 'reject' | 'override' | 'reproject' | 'execute'>
}

export type V5JudgeMetricSummary = {
  mean: number
  cvar_20_upper: number
  worst: number
}

export type V5JudgeStrategy = {
  raw_rollouts: Array<{
    event_id: string
    cumulative_critical_service_days_lost: number
    cvar_10_weighted_unmet_need: number
    worst_district_service_days_lost: number
    hard_resource_constraint_violations: number
  }>
  metrics: Record<
    'cumulative_critical_service_days_lost'
    | 'cvar_10_weighted_unmet_need'
    | 'worst_district_service_days_lost',
    V5JudgeMetricSummary
  >
}

export type V5JudgePlanPreview = {
  step: number
  raw_latent_action: Record<string, unknown>
  raw_policy_proposal: {
    reserve_amount: number
    unassigned_amount: number
    projects: Array<{
      project_id: string
      requested_amount: number
    }>
  }
  qp_projection: {
    reserve_amount: number
    unallocated_amount: number
    allocations: Array<{
      project_id: string
      amount: number
    }>
  }
  solver_change: {
    raw_proposal_already_feasible: boolean
    project_allocation_l1_distance: number
    allocation_l2_distance: number
    changed_project_count: number
    fraction_project_allocations_changed: number
    constraint_categories: string[]
    intervention_count: number
  }
  strategic_proposal_sha256: string
  raw_policy_proposal_sha256: string
  qp_projection_sha256: string
  prepared_action_sha256: string
}

export type V5JudgeExpectedTrajectory = {
  semantics: string
  policy_observed_hidden_tape: false
  hazard_tape_sha256: string
  trajectory_sha256: string
  planned_actions: V5JudgePlanPreview[]
  planned_actions_sha256: string
  points: Array<{
    step: number
    state_sha256: string
    raw_policy_proposal_sha256: string | null
    qp_projection_sha256: string | null
    reward: number | null
    spendable_resources: number
    reserve_inventory: number
    cumulative_critical_service_days_lost: number
    cvar_10_weighted_unmet_need: number
    worst_district_service_days_lost: number
    hard_resource_constraint_violations: number
  }>
}

export type V5JudgePolicyResult = {
  status: 'evaluated'
  policy_id: string
  initial_state_sha256: string
  branch_day: number
  initial_plan: V5JudgePlanPreview
  expected_trajectory: V5JudgeExpectedTrajectory
  event_injections: Array<{
    event_id: string
    severity: number
    day: number
    epicenter_district: number
    hazard_tape_sha256: string
  }>
  strategies: {
    continue_open_loop: V5JudgeStrategy
    replan_public_observation: V5JudgeStrategy
  }
  replan_minus_continue: Record<
    'cumulative_critical_service_days_lost'
    | 'cvar_10_weighted_unmet_need'
    | 'worst_district_service_days_lost',
    V5JudgeMetricSummary
  >
  checkpoint_identity?: Record<string, unknown>
}

export type V5JudgePostEventStrategy = {
  status: 'evaluated'
  policy_id: string
  raw_rollouts: Array<{
    event_id: string
    post_event_state_sha256: string
    cumulative_critical_service_days_lost: number
    cvar_10_weighted_unmet_need: number
    worst_district_service_days_lost: number
    hard_resource_constraint_violations: number
  }>
  metrics: Record<
    'cumulative_critical_service_days_lost'
    | 'cvar_10_weighted_unmet_need'
    | 'worst_district_service_days_lost',
    V5JudgeMetricSummary
  >
}

export type V5JudgeDemo = {
  schema_version: 'v5-judge-demo-v1'
  result_class: 'fixed_held_out_judge_demo'
  evidence_label: string
  validation_design: {
    split: 'validation'
    fixed_seed: number
    profile_id: string
    case_sha256: string
    simulator_version_hash: string
    frozen_manifest_membership?: Record<string, unknown>
  }
  event_protocol: {
    suite_id: string
    suite_sha256: string
    event_count: number
    selection_rule: string
    omitted_event_count: 0
    event_selection_after_results: false
    cvar_tail_fraction: number
  }
  baseline_selection: {
    status: 'verified' | 'not_evaluated'
    policy_id: string
    role: string
    strongest_baseline_verified: boolean
    reason?: string
    selection_rule?: string
    shared_validation_index: {
      path: string
      sha256: string
      scenario_tapes_sha256: string
    } | null
    context_binding?: Record<string, unknown>
  }
  synthetic_proxy_disclosure: {
    scenario_and_outcomes: string
    event_injections: string
    expected_trajectories: string
    observed_real_world_values_present: false
    operational_forecast: false
  }
  baseline: V5JudgePolicyResult
  learned: V5JudgePolicyResult | {
    status: 'not_evaluated'
    policy_id: null
    checkpoint_identity: null
    reason: string
    strategies: null
  }
  post_event_comparison: {
    origin_policy_id: string
    branch_day_before_event: number
    post_event_branch_step: number
    comparison_contract: {
      same_post_event_state_within_each_event: true
      event_realized_before_strategy_branch: true
      original_schedule_uses_future_observations: false
      original_schedule_rule: string
      policy_observed_hidden_event_tape: false
    }
    strategies: {
      continue_original_plan: V5JudgePostEventStrategy
      baseline_replan_from_current_state: V5JudgePostEventStrategy
      learned_replan_from_current_state: V5JudgePostEventStrategy | {
        status: 'not_evaluated'
        policy_id: null
        raw_rollouts: null
        metrics: null
        reason: string
      }
    }
    comparison_sha256: string
  }
  explanation: {
    status: 'evaluated'
    template_id: 'v5-matched-action-removal-v1'
    named_project_id: string
    affected_district_ids: string[]
    narrative: string
    explanation_sha256: string
    endpoint_summary: {
      rollout_count: number
      removal_minus_selected_immediate_metric_mean: number
      removal_minus_selected_critical_loss_mean: number
      removal_minus_selected_cvar_unmet_need_mean: number
      removal_minus_selected_worst_district_loss_mean: number
      critical_loss_delta_rollout_variation: number
    }
  }
  scientific_claims: Array<{
    claim_id: string
    status: 'supported' | 'inconclusive'
    statement: string
    reason?: string
    scope?: string
    supporting_json_pointers: string[]
  }>
  report_sha256: string
}
