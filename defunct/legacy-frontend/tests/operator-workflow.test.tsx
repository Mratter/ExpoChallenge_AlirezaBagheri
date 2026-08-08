import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  createPlan,
  createPlanningSession,
  executePlan,
  loadJudgeDemo,
  overridePlan,
  transitionPlan,
} from '../src/api'
import { PlanReviewPanel } from '../src/operator/PlanReviewPanel'
import type {
  CheckpointSelectionProvenance,
  CompareResponse,
  DayResult,
  JudgeDemoResponse,
  PlanRecord,
  PlanStatus,
  PlanningSession,
} from '../src/types'

vi.mock('../src/api', () => ({
  createPlan: vi.fn(),
  createPlanningSession: vi.fn(),
  executePlan: vi.fn(),
  loadJudgeDemo: vi.fn(),
  overridePlan: vi.fn(),
  transitionPlan: vi.fn(),
}))

function day(dayNumber: number, allocation: number[]): DayResult {
  return {
    day: dayNumber,
    shock: { day: dayNumber, type: null, severity: 0, impact: [0, 0, 0, 0, 0], budget_factor: 0, forced: false },
    available_budget: 180,
    services_before: [0.3, 0.3, 0.3, 0.3, 0.3],
    services_after_shock: [0.3, 0.3, 0.3, 0.3, 0.3],
    raw_action: [0, 0, 0, 0, 0],
    raw_proposal: [90, 20, 20, 25, 25],
    lower_bounds: [10, 10, 10, 10, 10],
    upper_bounds: [80, 80, 80, 80, 80],
    allocation,
    projection: {
      distance: 12,
      sum: 180,
      constraint_violations: 0,
      violation_breakdown: { sum_violations: 0, budget_violations: 0, lower_violations: 0, upper_violations: 0, total: 0 },
      bindings: [{ service: 'transport', lower: false, upper: true }],
    },
    planner_evidence: null,
    support: [0.7, 0.7, 0.7, 0.7, 0.7],
    gain: [0, 0, 0, 0, 0],
    strain: [0, 0, 0, 0, 0],
    services_end: [0.4, 0.4, 0.4, 0.4, 0.4],
    resilience: 0.4,
    reward: 0.4,
  }
}

function resultFixture(): CompareResponse {
  const candidate = [day(1, [80, 25, 25, 25, 25]), day(2, [60, 30, 30, 30, 30])]
  const baseline = [day(1, [36, 36, 36, 36, 36]), day(2, [36, 36, 36, 36, 36])]
  const planner = (name: string, trajectory: DayResult[]) => ({
    planner: name,
    rauc: 0.4,
    final_resilience: 0.4,
    minimum_resilience: 0.3,
    post_shock_recovery_shortfall_auc: 0.1,
    days_to_pre_shock_recovery_after_largest_loss: 3,
    critical_service_days: 4,
    total_projection_distance: 12,
    constraint_violations: 0,
    trajectory,
  })
  return {
    schema_version: '3.0.0',
    engine_version: 'city-recovery-env-v2',
    engine_spec: {},
    engine_spec_sha256: '8'.repeat(64),
    environment: {
      id: 'CityRecoveryEnv-v2', version: '2.0.0', observation_count: 33,
      action_count: 5, spec_sha256: '8'.repeat(64),
    },
    result_id: 'c'.repeat(64),
    persistence: { format: 'canonical-json-v1', idempotent: true, result_id: 'c'.repeat(64) },
    seed: 424242,
    generator: 'numpy.PCG64',
    scenario: {
      name: 'Operator fixture', horizon_days: 7, daily_budget: 180,
      initial_services: [0.3, 0.3, 0.3, 0.3, 0.3], priorities: [1, 1, 1, 1, 1],
      shock_probability: 0, severity_min: 0.1, severity_max: 0.2, forced_shock: null,
    },
    services: ['transport', 'housing', 'food', 'healthcare', 'public_services'],
    shock_schedule: candidate.map((item) => item.shock),
    shock_schedule_sha256: 'a'.repeat(64),
    policy: {
      id: 'ppo', artifact_type: 'stable_baselines3_ppo', algorithm: 'PPO', runtime: 'ONNX',
      sha256: 'b'.repeat(64), sb3_checkpoint_sha256: 'd'.repeat(64), parity_report_sha256: 'e'.repeat(64),
      disclosure: 'Synthetic only.',
      legacy_candidate: { id: 'legacy', artifact_type: 'deterministic_linear_policy_candidate', is_ppo: false, sha256: 'f'.repeat(64), disclosure: 'Not PPO.' },
    },
    baseline_spec: { id: 'glop', library: 'OR-Tools', library_version: '9', solver: 'GLOP', objective: 'visible', future_shocks_visible: false, engine_spec_sha256: '8'.repeat(64) },
    baseline: planner('baseline', baseline),
    candidate: planner('candidate', candidate),
    comparison: { primary_metric: 'rauc', candidate_minus_baseline: 0, outcome: 'tie' },
    limitations: ['Synthetic only.'],
  }
}

function provenanceFixture(): CheckpointSelectionProvenance {
  return {
    schema_version: '1.0.0',
    evaluation_manifest_sha256: '4'.repeat(64),
    binding_scope: 'checkpoint_selection_only',
    final_protocol_executed: false,
    checkpoint_artifact_path: 'artifacts/city_recovery_ppo.v2.onnx',
    checkpoint_sha256: 'b'.repeat(64),
    source_result_id: 'c'.repeat(64),
    source_result_sha256: '7'.repeat(64),
    source_result_schema_version: '3.0.0',
    source_engine_spec_sha256: '8'.repeat(64),
    source_result_policy_sha256: 'b'.repeat(64),
    source_result_policy_verified: true,
    source_result_integrity_verified: true,
    simulator_version_hash: '9'.repeat(64),
    verification_status: 'verified',
  }
}

function sessionFixture(): PlanningSession {
  return {
    schema_version: '1.0.0',
    session_id: '2'.repeat(32),
    label: 'review',
    source_result_id: 'c'.repeat(64),
    mode: 'simulation',
    simulation_only: true,
    simulation_auto_execute: false,
    simulator_version_hash: '9'.repeat(64),
    checkpoint_selection_provenance: provenanceFixture(),
    created_at: '2026-07-22T00:00:00Z',
  }
}

function planFixture(status: PlanStatus, version: number): PlanRecord {
  const result = resultFixture()
  const statuses: PlanStatus[] = ['proposed', 'reviewed', 'approved', 'executed']
  const used = statuses.slice(0, statuses.indexOf(status) + 1)
  if (!used.includes(status)) used.splice(0, used.length, 'proposed', status)
  return {
    schema_version: '1.0.0', plan_id: '1'.repeat(32), session_id: '2'.repeat(32),
    source_result_id: result.result_id, status, version, simulation_only: true,
    simulation_auto_execute: false, simulator_version_hash: '9'.repeat(64),
    checkpoint_selection_provenance: provenanceFixture(),
    original_plan: result.candidate.trajectory.map((item) => ({
      day: item.day, available_budget: item.available_budget,
      original_policy_proposal: item.raw_proposal,
      original_feasible_allocation: item.allocation,
      lower_bounds: item.lower_bounds, upper_bounds: item.upper_bounds,
    })),
    operator_overrides: [], reprojected_plan: [],
    execution_id: status === 'executed' ? '3'.repeat(32) : null,
    audit_events: used.map((item, index) => ({
      sequence: index + 1, action: item, from_status: index ? used[index - 1] : null,
      to_status: item, actor: { session_label: 'test' }, timestamp: '2026-07-22T00:00:00Z',
      reason: 'test', version: index + 1, simulator_version_hash: '9'.repeat(64),
      checkpoint_selection_provenance: provenanceFixture(),
    })),
    created_at: '2026-07-22T00:00:00Z', updated_at: '2026-07-22T00:00:00Z',
  }
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

beforeEach(() => {
  vi.mocked(loadJudgeDemo).mockRejectedValue(
    new Error('The completed v4 evaluation is not ready.'),
  )
})

function judgeFixture(): JudgeDemoResponse {
  const matrix = (value: number) => Array.from(
    { length: 4 },
    () => Array.from({ length: 5 }, () => value),
  )
  const proof = {
    branch_state_sha256_before_tail: '1'.repeat(64),
    branch_state_sha256_after_tail: '1'.repeat(64),
    conditional_tail_sha256: '2'.repeat(64),
    full_tape_sha256: '3'.repeat(64),
  }
  const aggregate = (offset: number) => ({
    episode_count: 3,
    mean_cumulative_critical_service_days_lost: 1 + offset,
    cvar_10_weighted_unmet_need: 2 + offset,
    mean_worst_district_service_days_lost: 3 + offset,
    maximum_worst_district_service_days_lost: 4 + offset,
    hard_resource_constraint_violations: 0,
    mcse_across_matched_rollouts: {},
  })
  const projection = {
    method: 'analytic QP',
    distance: 1.25,
    sum: 100,
    constraint_violations: 0,
    violation_breakdown: { sum: 0, lower: 0, upper: 0, total: 0 },
    bindings: { lower_count: 1, upper_count: 0 },
  }
  const trajectory = [1, 2].map((day) => ({
    day,
    raw_priorities: matrix(0.05),
    raw_reserve_head: 0,
    allocation: matrix(5),
    services_end: matrix(day === 1 ? 0.72 : 0.78),
    service_pool: 100,
    projection,
    hard_resource_constraint_violations: 0,
  }))
  const proposalAudit = [{
    day: 1,
    policy_id: 'policy',
    policy_called: true,
    priorities: matrix(0.05),
    reserve_head: 0,
    proposal_sha256: 'c'.repeat(64),
  }]
  const initialOutcome = (policyId: string) => ({
    policy_id: policyId,
    synthetic_proxy: true as const,
    tape_sha256: '8'.repeat(64),
    trajectory,
    trajectory_sha256: 'd'.repeat(64),
    proposal_audit: proposalAudit,
    proposal_schedule_sha256: 'e'.repeat(64),
    first_day_decision: {
      day: 1,
      raw_priorities: matrix(0.05),
      raw_reserve_head: 0,
      projected_allocation: matrix(5),
      projection_diagnostics: projection,
      service_pool: 100,
    },
    domain_metrics: {},
  })
  const branch = (
    mode: 'open_loop_original_learned' | 'replan_baseline' | 'replan_learned',
    policyId: string,
  ) => ({
    mode,
    policy_id: policyId,
    synthetic_proxy: true as const,
    replanning: mode !== 'open_loop_original_learned',
    open_loop_proposal_replay: mode === 'open_loop_original_learned',
    trajectory,
    trajectory_sha256: 'd'.repeat(64),
    proposal_audit: proposalAudit,
    domain_metrics: {},
    proof,
  })
  return {
    schema_version: '1.0.0',
    verification_status: 'verified',
    result_class: 'preliminary',
    claim_eligible: false,
    artifact_manifest_sha256: 'a'.repeat(64),
    report: {
      schema_version: '1.0.0-development',
      demo_id: 'judge-demo-v4-matched-conditional-branches',
      result_class: 'preliminary',
      claim_eligible: false,
      synthetic_proxy: true,
      inputs: {
        held_out_seed: 472100,
        learned_training_seed: 47017,
        matched_rollout_seeds: [88001, 88002, 88003],
        prefix_days: 3,
        active_branch_day: 4,
      },
      provenance: {
        simulator_version_hash: '4'.repeat(64),
        evaluation_manifest_sha256: '5'.repeat(64),
        learned_policy: {
          policy_id: 'relational_gnn_ppo_qp', sha256: '6'.repeat(64), training_seed: 47017,
        },
        baseline_policy: {
          policy_id: 'needs_weighted_heuristic_fixed_reserve_v4', sha256: '7'.repeat(64),
        },
      },
      initial_comparison: {
        matched_initial_tape: true,
        tape_sha256: '8'.repeat(64),
        baseline: initialOutcome('needs_weighted_heuristic_fixed_reserve_v4'),
        learned: initialOutcome('relational_gnn_ppo_qp'),
      },
      branch_point: {
        prefix_days: 3,
        active_day: 4,
        snapshot_sha256: '9'.repeat(64),
        branch_state_sha256: '1'.repeat(64),
        explicit_active_disruption: {
          type: 'aftershock', severity: 0.32, focus_district: 2, synthetic_proxy: true,
        },
      },
      conditional_rollouts: [{
        rollout_seed: 88001,
        matched_proof: proof,
        branches: {
          open_loop_original_learned: branch(
            'open_loop_original_learned', 'relational_gnn_ppo_qp',
          ),
          replan_baseline: branch(
            'replan_baseline', 'needs_weighted_heuristic_fixed_reserve_v4',
          ),
          replan_learned: branch('replan_learned', 'relational_gnn_ppo_qp'),
        },
      }],
      aggregates: {
        open_loop_original_learned: aggregate(0),
        replan_baseline: aggregate(0.25),
        replan_learned: aggregate(-0.25),
      },
      counterfactual: {
        narrative: 'On day 5, the selected healthcare cell differed across matched branches.',
        interpretation_boundary: 'This is a within-simulator difference, not a real-city estimate.',
        selected_action: {
          name: 'learned_healthcare_priority', day: 4, district_id: 'D2',
          service: 'healthcare', raw_priority: 0.08, projected_allocation: 19.5,
        },
        feasible_alternative: {
          name: 'remove_learned_healthcare_priority',
          intervention: 'zero and reproject', raw_priority: 0,
          projected_allocation: 15.3, feasibility_verified: true,
          post_projection_constraint_violations: 0,
        },
        matched_rollout_seeds: [88001, 88002, 88003],
        immediate_metric_difference: {
          metric: 'cumulative_critical_service_days_lost',
          selected_minus_alternative: -0.01,
        },
        long_term_metric_difference: {
          metric: 'cumulative_critical_service_days_lost',
          mean_selected_minus_alternative: -0.02,
        },
        cvar_difference: {
          metric: 'weighted_unmet_need', selected_minus_alternative: 0.03,
        },
        named_action_removal_proof: {
          actual_matched_rerun: true, caller_assertion_used: false,
        },
        geographic_dependency_path: { district_ids: ['D1', 'D2'], path_exists: true },
        service_dependency_path: {
          from_service: 'transport', to_service: 'healthcare',
          dependency_weight: 0.4, path_exists: true,
        },
      },
      claim_status_rows: [{
        claim: 'Learned replanning improves outcomes.',
        evidence_status: 'inconclusive',
        result_class: 'preliminary',
        claim_eligible: false,
      }],
      disclosures: ['Synthetic only.'],
      report_sha256: 'b'.repeat(64),
    },
  }
}

describe('operator workflow', () => {
  it('shows raw, projected, operator, and an honest unavailable Judge fallback', async () => {
    render(<PlanReviewPanel result={resultFixture()} />)

    expect(screen.getByText('Raw neural proposal')).toBeVisible()
    expect(screen.getByText('Solver-projected plan')).toBeVisible()
    expect(screen.getByText('Operator-approved plan')).toBeVisible()
    expect(screen.getByText('Judge Demo Mode')).toBeVisible()
    expect(await screen.findByText('Verified three-branch evidence is not ready')).toBeVisible()
    expect(screen.getByText(/no v4 branch outcome is inferred/)).toBeVisible()
  })

  it('loads verified matched evidence and switches among all three branch views', async () => {
    vi.mocked(loadJudgeDemo).mockResolvedValue(judgeFixture())
    render(<PlanReviewPanel result={resultFixture()} />)

    expect(await screen.findByText('Verified artifact')).toBeVisible()
    const original = screen.getByRole('tab', { name: /Continue original/ })
    const baseline = screen.getByRole('tab', { name: /Replan · baseline/ })
    const learned = screen.getByRole('tab', { name: /Replan · learned/ })
    expect(original).toHaveAttribute('aria-selected', 'true')
    expect(baseline).toBeEnabled()
    expect(learned).toBeEnabled()

    fireEvent.click(baseline)
    expect(baseline).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tabpanel')).toHaveTextContent(
      'needs_weighted_heuristic_fixed_reserve_v4',
    )
    expect(screen.getByRole('tabpanel')).toHaveTextContent('1.25')
    fireEvent.keyDown(baseline, { key: 'ArrowRight' })
    expect(learned).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByText(/0 listed claims supported; 1 remain inconclusive/)).toBeVisible()
    expect(screen.getByText('Baseline and learned plans on one frozen tape')).toBeVisible()
    expect(screen.getByText('Baseline initial plan')).toBeVisible()
    expect(screen.getByText('Learned initial plan')).toBeVisible()
    expect(screen.getAllByText('Raw policy priority')).toHaveLength(2)
    expect(screen.getAllByText('QP allocation')).toHaveLength(2)
    expect(screen.getByLabelText('Baseline initial plan initial trajectory')).toBeVisible()
    expect(screen.getByLabelText('Learned initial plan initial trajectory')).toBeVisible()
    expect(screen.getByText('Learned replanning improves outcomes.')).toBeVisible()
    expect(screen.getByText('not claim-eligible')).toBeVisible()
    expect(screen.getByText('actual matched rerun')).toBeVisible()
  })

  it('opens a verified current-state replan without mutating the v2 lifecycle', async () => {
    vi.mocked(loadJudgeDemo).mockResolvedValue(judgeFixture())
    vi.mocked(createPlanningSession).mockResolvedValue(sessionFixture())
    vi.mocked(createPlan).mockResolvedValue(planFixture('proposed', 1))
    render(<PlanReviewPanel result={resultFixture()} />)

    expect(await screen.findByText('Verified artifact')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Start operator review' }))
    await screen.findByText(/Plan proposed/)
    const replan = screen.getByRole('button', { name: 'Replan from current state' })
    expect(replan).toBeEnabled()
    fireEvent.click(replan)

    expect(screen.getByRole('tab', { name: /Replan · learned/ })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    expect(await screen.findByText(/v2 proposed plan remains unchanged at version 1/)).toBeVisible()
    expect(transitionPlan).not.toHaveBeenCalled()
    expect(screen.getByText('version 1')).toBeVisible()
  })

  it('creates a review record then performs explicit review and approval', async () => {
    vi.mocked(createPlanningSession).mockResolvedValue(sessionFixture())
    vi.mocked(createPlan).mockResolvedValue(planFixture('proposed', 1))
    vi.mocked(transitionPlan)
      .mockResolvedValueOnce(planFixture('reviewed', 2))
      .mockResolvedValueOnce(planFixture('approved', 3))
    render(<PlanReviewPanel result={resultFixture()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Start operator review' }))
    await screen.findByText(/Plan proposed/)
    expect(createPlanningSession).toHaveBeenCalledWith(
      'Review Operator fixture',
      'c'.repeat(64),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }))

    await waitFor(() => expect(transitionPlan).toHaveBeenCalledTimes(2))
    expect(vi.mocked(transitionPlan).mock.calls[0][1]).toBe('review')
    expect(vi.mocked(transitionPlan).mock.calls[1][1]).toBe('approve')
    expect(await screen.findByText('Plan approved for simulation execution only.')).toBeVisible()
  })

  it('keeps simulation execution disabled until approval', async () => {
    vi.mocked(createPlanningSession).mockResolvedValue(sessionFixture())
    vi.mocked(createPlan).mockResolvedValue(planFixture('proposed', 1))
    render(<PlanReviewPanel result={resultFixture()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Start operator review' }))
    await screen.findByText(/Plan proposed/)
    expect(screen.getByRole('button', { name: 'Execute approved simulation plan' })).toBeDisabled()
    expect(screen.getByRole('button', { name: /Replan from current state/ })).toBeDisabled()
    expect(executePlan).not.toHaveBeenCalled()
    expect(overridePlan).not.toHaveBeenCalled()
  })

  it('fails closed when the API plan differs from the result displayed to the operator', async () => {
    const divergent = planFixture('proposed', 1)
    divergent.original_plan[0] = {
      ...divergent.original_plan[0],
      original_feasible_allocation: [79, 26, 25, 25, 25],
    }
    vi.mocked(createPlanningSession).mockResolvedValue(sessionFixture())
    vi.mocked(createPlan).mockResolvedValue(divergent)
    render(<PlanReviewPanel result={resultFixture()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Start operator review' }))

    expect(await screen.findByText(
      'The authoritative plan does not match the simulation result on screen.',
    )).toBeVisible()
    expect(screen.getByText('untracked')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Start operator review' })).toBeEnabled()
  })
})
