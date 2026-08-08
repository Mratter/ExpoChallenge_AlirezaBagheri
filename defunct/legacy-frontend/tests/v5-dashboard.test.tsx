import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type {
  V5DevelopmentSnapshot,
  V5JudgeDemo,
  V5OperatorPlan,
  V5PlanStatus,
  V5ProfilesResponse,
} from '../src/types'
import { V5DevelopmentPanel } from '../src/v5/V5DevelopmentPanel'

const DISCLAIMER = 'Developmental v5 simulator and model-selection evidence. Not operationally validated.'

const profiles: V5ProfilesResponse = {
  schema_version: 'v5-development-dashboard-v1',
  simulator_version: 'v5-city-recovery-simulator-v1',
  default_profile_id: 'v5_diagnostic',
  disclaimer: DISCLAIMER,
  profiles: [
    {
      profile_id: 'v5_diagnostic',
      label: 'Diagnostic',
      horizon_days: 24,
      district_count: 8,
      dry_run_only: false,
      simulator_version_hash: 'a'.repeat(64),
    },
    {
      profile_id: 'v5_final',
      label: 'Final dry-run configuration',
      horizon_days: 120,
      district_count: 8,
      dry_run_only: true,
      simulator_version_hash: 'b'.repeat(64),
    },
  ],
}

function snapshot(profileId = 'v5_diagnostic'): V5DevelopmentSnapshot {
  const sectors = [
    'transport',
    'housing',
    'food',
    'healthcare',
    'public_services',
    'resilience',
  ] as const
  return {
    schema_version: 'v5-development-dashboard-v1',
    result_class: 'development_snapshot',
    disclaimer: DISCLAIMER,
    simulator: {
      version: 'v5-city-recovery-simulator-v1',
      profile_id: profileId,
      profile_label: profileId === 'v5_final' ? 'Final dry-run configuration' : 'Diagnostic',
      simulator_version_hash: profileId === 'v5_final' ? 'b'.repeat(64) : 'a'.repeat(64),
      horizon_days: profileId === 'v5_final' ? 120 : 24,
      dry_run_only: profileId === 'v5_final',
      seed: 520000,
      scenario_id: `balanced_recovery:${profileId}:520000`,
    },
    action_schema: {
      version: 'v5-sector-conditioned-district-reserve-v1',
      sha256: 'c'.repeat(64),
      output_count: 55,
      order: ['6 sector logits', '6 x 8 district logits', '1 reserve logit'],
    },
    current_policy: {
      policy_id: 'dependency_aware_v5',
      kind: 'non_learning_development_baseline',
      checkpoint_identity: null,
    },
    model: {
      candidate_id: 'relational_gnn_ppo',
      trainable_parameter_count: 635371,
      checkpoint_status: 'unavailable_pilot_pending',
      checkpoint_identity: null,
      selected_for_final: false,
    },
    strategic_action: {
      status: 'Proposed',
      latent_output_count: 55,
      latent_vector: Array.from({ length: 55 }, (_, index) => index / 100),
      sector_names: [...sectors],
      district_ids: Array.from({ length: 8 }, (_, index) => `district_0${index}`),
      sector_shares: Array(6).fill(1 / 6),
      district_shares: Array.from({ length: 6 }, () => Array(8).fill(1 / 8)),
      desired_allocation: Array.from({ length: 6 }, (_, sector) => (
        Array.from({ length: 8 }, (_, district) => 2 + sector + district / 10)
      )),
      spendable_resources: 160,
      unallocated_amount: 0,
      interpretation: 'Pre-projection strategic proposal; latent logits are not resource units.',
    },
    carrying_reserve: {
      proposed_fraction: 0.15,
      proposed_amount: 24,
      inventory_before: 18,
      executed_inventory_after: null,
      status: 'proposed_not_executed',
    },
    dependency_graph: {
      topology_version: 'v5-typed-district-sector-graph-v1',
      topology_sha256: 'd'.repeat(64),
      node_count: 48,
      directed_edge_count: 224,
      critical_edge_count: 72,
      relation_counts: { dependency: 64, geographic: 96, supply: 16, redundancy: 48 },
      edge_orientation: 'directed source to target',
    },
    projects: {
      mode_counts: { temporary: 14, permanent: 26, resilience: 8 },
      top_proposals: [],
      current_lifecycle_status: 'Proposed',
      lifecycle_statuses: [
        'Proposed',
        'Reviewed',
        'Approved',
        'Rejected',
        'Overridden',
        'Reprojected',
        'Executed',
      ],
      execution_status: 'not_executed_read_only_preview',
    },
    forecast: {
      event_probability: 0.32,
      confidence: 0.65,
      severity_low: 0.18,
      severity_mean: 0.28,
      severity_high: 0.39,
      aid_arrival_day_low: 1,
      aid_arrival_day_high: 4,
      expected_aid_fraction: 0.2,
      uncertainty: 0.35,
      status: 'public_forecast_only',
    },
    displacement_and_equity: {
      rows: [],
      total_displaced_people: 0,
      total_displaced_fraction: 0,
      worst_district: {
        district_id: 'district_06',
        metric: 'critical_observed_need_proxy',
        value: 0.54,
        scope: 'pre-decision developmental snapshot',
      },
    },
    projection: {
      distance: null,
      status: 'unavailable_not_executed',
      reason: 'The read-only preview does not run feasibility projection or physics.',
    },
    policy_comparison: [
      {
        family: 'Baseline',
        policy: 'dependency_aware_v5',
        status: 'development_policy_available',
        endpoint_metrics: null,
        reason: 'No evaluation result is implied.',
      },
      {
        family: 'MPC',
        policy: 'three_step_mpc_v5 / six_step_mpc_v5',
        status: 'evaluation_pending',
        endpoint_metrics: null,
        reason: 'Matched pilot evidence is pending.',
      },
      {
        family: 'Learned',
        policy: 'relational_gnn_ppo candidate',
        status: 'pilot_pending',
        endpoint_metrics: null,
        reason: 'No selected checkpoint is available.',
      },
    ],
    evidence: {
      status: 'pre_pilot_pending',
      pilot: { label: 'PILOT / PENDING', result_available: false, metrics: null },
      final: {
        label: 'FINAL / NOT RUN',
        result_available: false,
        metrics: null,
        campaign_started: false,
        configuration_dry_run_only: true,
      },
    },
  }
}

const actionByStatus: Record<V5PlanStatus, V5OperatorPlan['actions_available']> = {
  proposed: ['review', 'reject', 'override'],
  reviewed: ['approve', 'reject', 'override'],
  approved: ['execute', 'override', 'reject'],
  rejected: [],
  overridden: ['reproject', 'reject'],
  reprojected: ['approve', 'override', 'reject'],
  executed: [],
}

function operatorPlan(status: V5PlanStatus, version: number): V5OperatorPlan {
  const statuses: V5PlanStatus[] = ['proposed', 'reviewed', 'overridden', 'reprojected', 'approved', 'executed']
  const currentIndex = statuses.indexOf(status)
  const audited = currentIndex < 0 ? ['proposed', 'rejected'] as V5PlanStatus[] : statuses.slice(0, currentIndex + 1)
  const approved = status === 'approved' || status === 'executed'
  const reprojected = ['reprojected', 'approved', 'executed'].includes(status)
  return {
    schema_version: 'v5-operator-lifecycle-v1',
    plan_id: '1'.repeat(32),
    version,
    status,
    execution_mode: 'approval_required',
    approval_required: true,
    created_at: '2026-07-22T10:00:00.000Z',
    simulator: {
      version: 'v5-city-recovery-simulator-v1',
      profile_id: 'v5_diagnostic',
      seed: 520000,
      scenario_id: 'scenario',
      scenario_sha256: '2'.repeat(64),
      simulator_version_hash: '3'.repeat(64),
    },
    raw_proposal: {
      latent_action_sha256: '4'.repeat(64),
      project_proposal_sha256: '5'.repeat(64),
      reserve_amount: 12,
      unallocated_amount: 2,
      allocations: [{ project_id: 'district_00/transport-project', amount: 8, maximum_amount: 20 }],
    },
    solver_projection: {
      feasible_action_sha256: '6'.repeat(64),
      solver: 'osqp',
      projection_distance: 1.25,
      reserve_amount: 12,
      unallocated_amount: 3,
      solver_change: {
        allocation_l1_distance: 2,
        allocation_l2_distance: 1.25,
        modified_allocation_fraction: 0.25,
        constraint_categories: ['capacity_projection'],
        change_class: 'scale_only',
        fallback_required: false,
      },
    },
    operator_changes: currentIndex >= 2 ? [{
      target: 'district_00/transport-project',
      before: 8,
      requested_after: 4,
      operator_id: 'local-operator',
      session_id: 'v5-dashboard-session',
      reason: 'Preserve reserve.',
      timestamp: '2026-07-22T10:01:00.000Z',
      change_sha256: '7'.repeat(64),
    }] : [],
    current_reprojection: reprojected || approved ? {
      feasible_action_sha256: '8'.repeat(64),
      projection_distance: 0,
      reserve_amount: 12,
      unallocated_amount: 7,
    } : null,
    approved_plan: approved ? {
      approval_id: 'approval',
      feasible_action_sha256: '8'.repeat(64),
      approved_action_sha256: '9'.repeat(64),
      reserve_amount: 12,
      allocations: [{ project_id: 'district_00/transport-project', amount: 4 }],
    } : null,
    reprojected_approved_plan: approved ? {
      approval_id: 'approval',
      feasible_action_sha256: '8'.repeat(64),
      approved_action_sha256: '9'.repeat(64),
      reserve_amount: 12,
      allocations: [{ project_id: 'district_00/transport-project', amount: 4 }],
    } : null,
    solver_change_evidence: { initial_projection_sha256: '6'.repeat(64), reprojections: [] },
    comparison: [{
      project_id: 'district_00/transport-project',
      district_index: 0,
      sector_index: 0,
      raw_proposal: 8,
      solver_projection: 7,
      operator_requested: currentIndex >= 2 ? 4 : 8,
      reprojected: reprojected || approved ? 4 : null,
      approved: approved ? 4 : null,
    }],
    audit_events: audited.map((eventStatus, index) => ({
      sequence: index,
      action: eventStatus,
      status: eventStatus,
      operator_id: 'local-operator',
      session_id: 'v5-dashboard-session',
      reason: `Recorded ${eventStatus}.`,
      timestamp: `2026-07-22T10:0${index}:00.000Z`,
      evidence_sha256: `${index}`.repeat(64),
      previous_event_sha256: index ? `${index - 1}`.repeat(64) : '0'.repeat(64),
      event_sha256: `${index + 1}`.repeat(64),
    })),
    execution: status === 'executed' ? {
      transition_id: 'a'.repeat(64),
      approved_action_sha256: '9'.repeat(64),
      state_before_sha256: 'b'.repeat(64),
      state_after_sha256: 'c'.repeat(64),
      record_sha256: 'd'.repeat(64),
    } : null,
    actions_available: actionByStatus[status],
  }
}

function judgeDemo(): V5JudgeDemo {
  const metrics = {
    cumulative_critical_service_days_lost: { mean: 2, cvar_20_upper: 3, worst: 3 },
    cvar_10_weighted_unmet_need: { mean: 0.3, cvar_20_upper: 0.4, worst: 0.4 },
    worst_district_service_days_lost: { mean: 4, cvar_20_upper: 5, worst: 5 },
  }
  const raw = [{
    event_id: 'event_01_moderate',
    cumulative_critical_service_days_lost: 2,
    cvar_10_weighted_unmet_need: 0.3,
    worst_district_service_days_lost: 4,
    hard_resource_constraint_violations: 0,
  }]
  const postRaw = raw.map((row) => ({ ...row, post_event_state_sha256: '8'.repeat(64) }))
  return {
    schema_version: 'v5-judge-demo-v1',
    result_class: 'fixed_held_out_judge_demo',
    evidence_label: DISCLAIMER,
    validation_design: {
      split: 'validation', fixed_seed: 530000, profile_id: 'v5_pilot',
      case_sha256: 'a'.repeat(64), simulator_version_hash: 'b'.repeat(64),
    },
    event_protocol: {
      suite_id: 'v5-fixed-held-out-event-suite-v1', suite_sha256: 'c'.repeat(64),
      event_count: 1, selection_rule: 'all registered events are evaluated and reported',
      omitted_event_count: 0, event_selection_after_results: false, cvar_tail_fraction: 0.2,
    },
    baseline_selection: {
      status: 'not_evaluated',
      policy_id: 'dependency_aware_v5',
      role: 'preregistered_default_baseline',
      strongest_baseline_verified: false,
      reason: 'No shared-validation index was supplied.',
      shared_validation_index: null,
    },
    baseline: {
      status: 'evaluated', policy_id: 'dependency_aware_v5', initial_state_sha256: '0'.repeat(64), branch_day: 4,
      initial_plan: {
        step: 0,
        raw_latent_action: {},
        raw_policy_proposal: {
          reserve_amount: 2,
          unassigned_amount: 0,
          projects: [{ project_id: 'district_00/transport-project', requested_amount: 8 }],
        },
        qp_projection: {
          reserve_amount: 2,
          unallocated_amount: 1,
          allocations: [{ project_id: 'district_00/transport-project', amount: 7 }],
        },
        solver_change: {
          raw_proposal_already_feasible: false,
          project_allocation_l1_distance: 1,
          allocation_l2_distance: 1,
          changed_project_count: 1,
          fraction_project_allocations_changed: 1,
          constraint_categories: ['project_upper_bound'],
          intervention_count: 1,
        },
        strategic_proposal_sha256: '1'.repeat(64),
        raw_policy_proposal_sha256: '2'.repeat(64),
        qp_projection_sha256: '3'.repeat(64),
        prepared_action_sha256: '4'.repeat(64),
      },
      expected_trajectory: {
        semantics: 'Deterministic synthetic projection conditional on the fixed tape.',
        policy_observed_hidden_tape: false,
        hazard_tape_sha256: '5'.repeat(64),
        trajectory_sha256: '6'.repeat(64),
        planned_actions: [],
        planned_actions_sha256: 'a'.repeat(64),
        points: [{
          step: 0,
          state_sha256: '7'.repeat(64),
          raw_policy_proposal_sha256: null,
          qp_projection_sha256: null,
          reward: null,
          spendable_resources: 10,
          reserve_inventory: 0,
          cumulative_critical_service_days_lost: 0,
          cvar_10_weighted_unmet_need: 0,
          worst_district_service_days_lost: 0,
          hard_resource_constraint_violations: 0,
        }],
      },
      event_injections: [{ event_id: 'event_01_moderate', severity: 0.2, day: 4, epicenter_district: 0, hazard_tape_sha256: 'd'.repeat(64) }],
      strategies: {
        continue_open_loop: { raw_rollouts: raw, metrics },
        replan_public_observation: { raw_rollouts: raw, metrics },
      },
      replan_minus_continue: {
        cumulative_critical_service_days_lost: { mean: 0, cvar_20_upper: 0, worst: 0 },
        cvar_10_weighted_unmet_need: { mean: 0, cvar_20_upper: 0, worst: 0 },
        worst_district_service_days_lost: { mean: 0, cvar_20_upper: 0, worst: 0 },
      },
    },
    learned: {
      status: 'not_evaluated', policy_id: null, checkpoint_identity: null,
      reason: 'No learned checkpoint path and exact SHA-256 were supplied.', strategies: null,
    },
    post_event_comparison: {
      origin_policy_id: 'dependency_aware_v5',
      branch_day_before_event: 4,
      post_event_branch_step: 5,
      comparison_contract: {
        same_post_event_state_within_each_event: true,
        event_realized_before_strategy_branch: true,
        original_schedule_uses_future_observations: false,
        original_schedule_rule: 'repeat the last pre-event action',
        policy_observed_hidden_event_tape: false,
      },
      strategies: {
        continue_original_plan: {
          status: 'evaluated', policy_id: 'dependency_aware_v5', raw_rollouts: postRaw, metrics,
        },
        baseline_replan_from_current_state: {
          status: 'evaluated', policy_id: 'dependency_aware_v5', raw_rollouts: postRaw, metrics,
        },
        learned_replan_from_current_state: {
          status: 'not_evaluated', policy_id: null, raw_rollouts: null, metrics: null,
          reason: 'No checksum-verified learned checkpoint was supplied.',
        },
      },
      comparison_sha256: '9'.repeat(64),
    },
    synthetic_proxy_disclosure: {
      scenario_and_outcomes: 'local synthetic simulator values',
      event_injections: 'registered synthetic stress events',
      expected_trajectories: 'fixed-tape synthetic projections',
      observed_real_world_values_present: false,
      operational_forecast: false,
    },
    explanation: {
      status: 'evaluated', template_id: 'v5-matched-action-removal-v1',
      named_project_id: 'district_00/transport-project', affected_district_ids: ['district_00'],
      narrative: 'Verified matched action-removal explanation.', explanation_sha256: 'e'.repeat(64),
      endpoint_summary: {
        rollout_count: 1, removal_minus_selected_immediate_metric_mean: 0,
        removal_minus_selected_critical_loss_mean: 0.1,
        removal_minus_selected_cvar_unmet_need_mean: 0.01,
        removal_minus_selected_worst_district_loss_mean: 0.2,
        critical_loss_delta_rollout_variation: 0,
      },
    },
    scientific_claims: [{
      claim_id: 'learned_policy_outperforms_baseline',
      status: 'inconclusive',
      statement: 'Learned superiority is not established.',
      reason: 'No verified learned checkpoint was supplied.',
      supporting_json_pointers: ['/learned'],
    }],
    report_sha256: 'f'.repeat(64),
  }
}

function installFetch() {
  const fetchMock = vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url === '/api/v5/profiles') {
      return { ok: true, json: async () => profiles }
    }
    const profile = url.includes('profile=v5_final') ? 'v5_final' : 'v5_diagnostic'
    return { ok: true, json: async () => snapshot(profile) }
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('v5 developmental dashboard', () => {
  it('exposes the exact strategic contract and explicit pending evidence', async () => {
    installFetch()
    render(<V5DevelopmentPanel />)

    expect(await screen.findByRole('heading', { name: 'v5 strategic preview' })).toBeVisible()
    expect(screen.getByText(DISCLAIMER)).toBeVisible()
    expect(screen.getByLabelText('Simulator version')).toHaveValue('v5_diagnostic')
    expect(screen.getByText('55 outputs')).toBeVisible()
    expect(screen.getAllByText('dependency_aware_v5')[0]).toBeVisible()
    expect(screen.getByText('635,371')).toBeVisible()
    expect(screen.getByLabelText('Proposed carrying reserve')).toHaveTextContent('24.00 units')

    const heatmap = screen.getByRole('table', {
      name: 'Sector × district desired allocation before feasibility projection',
    })
    expect(within(heatmap).getAllByRole('row')).toHaveLength(7)
    expect(heatmap).toHaveTextContent('Resilience')
    expect(heatmap).toHaveTextContent('D07')

    fireEvent.click(screen.getByText('Inspect all 55 ordered outputs'))
    expect(screen.getByLabelText('Exact 55-output latent action').children).toHaveLength(55)

    expect(screen.getByRole('heading', { name: '48 nodes · 224 directed edges' })).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Temporary and permanent restoration' })).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Forecast uncertainty' })).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Worst-district snapshot' })).toBeVisible()
    expect(screen.getByText('Pending execution')).toBeVisible()

    const comparison = screen.getByRole('table', { name: 'v5 policy comparison' })
    expect(comparison).toHaveTextContent('Baseline')
    expect(comparison).toHaveTextContent('MPC')
    expect(comparison).toHaveTextContent('Learned')
    expect(within(comparison).getAllByText('Unavailable / pending')).toHaveLength(3)
    expect(screen.getByText('PILOT / PENDING')).toBeVisible()
    expect(screen.getByText('FINAL / NOT RUN')).toBeVisible()

    const lifecycle = screen.getByLabelText('Plan lifecycle statuses')
    for (const status of ['Proposed', 'Reviewed', 'Approved', 'Rejected', 'Overridden', 'Reprojected', 'Executed']) {
      expect(within(lifecycle).getByText(status)).toBeVisible()
    }
  })

  it('loads the selected simulator profile and labels final as dry-run only', async () => {
    const fetchMock = installFetch()
    render(<V5DevelopmentPanel />)
    await screen.findByText('55 outputs')

    fireEvent.change(screen.getByLabelText('Simulator version'), { target: { value: 'v5_final' } })

    expect(await screen.findByText('Dry-run configuration only')).toBeVisible()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/v5/development/snapshot?profile=v5_final',
      expect.objectContaining({}),
    ))
  })

  it('drives the audited override, reprojection, approval, and execution lifecycle', async () => {
    let lifecycleStatus: V5PlanStatus = 'proposed'
    let version = 1
    const fetchMock = vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url === '/api/v5/profiles') return { ok: true, json: async () => profiles }
      if (url.startsWith('/api/v5/development/snapshot')) return { ok: true, json: async () => snapshot() }
      if (url === '/api/v5/operator/plans') {
        lifecycleStatus = 'proposed'
      } else if (url.endsWith('/review')) {
        lifecycleStatus = 'reviewed'
      } else if (url.endsWith('/override')) {
        lifecycleStatus = 'overridden'
      } else if (url.endsWith('/reproject')) {
        lifecycleStatus = 'reprojected'
      } else if (url.endsWith('/approve')) {
        lifecycleStatus = 'approved'
      } else if (url.endsWith('/execute')) {
        lifecycleStatus = 'executed'
      }
      return { ok: true, json: async () => operatorPlan(lifecycleStatus, version++) }
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<V5DevelopmentPanel />)
    await screen.findByText('55 outputs')

    fireEvent.click(screen.getByRole('button', { name: 'Open auditable review' }))
    expect(await screen.findByText('Mark reviewed')).toBeVisible()
    expect(screen.getByRole('table', { name: /Raw proposal/ })).toHaveTextContent('8.00')

    fireEvent.click(screen.getByRole('button', { name: 'Mark reviewed' }))
    expect(await screen.findByRole('button', { name: 'Approve exact plan' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Reject plan' })).toBeVisible()
    fireEvent.change(screen.getByLabelText('v5 override amount'), { target: { value: '4' } })
    fireEvent.click(screen.getByRole('button', { name: 'Record override' }))

    fireEvent.click(await screen.findByRole('button', { name: 'Reproject override' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Approve exact plan' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Execute approved step' }))

    await waitFor(() => expect(screen.getByText('executed')).toBeVisible())
    const ledger = screen.getByRole('table', { name: /Raw proposal/ })
    expect(ledger).toHaveTextContent('4.00')
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/v5/operator/plans/${'1'.repeat(32)}/execute`,
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('renders the fixed no-cherry-picking Judge Demo and verified explanation', async () => {
    const fetchMock = vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url === '/api/v5/profiles') return { ok: true, json: async () => profiles }
      if (url.startsWith('/api/v5/development/snapshot')) return { ok: true, json: async () => snapshot() }
      if (url === '/api/v5/judge-demo') return { ok: true, json: async () => judgeDemo() }
      throw new Error(`Unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<V5DevelopmentPanel />)
    await screen.findByText('55 outputs')

    fireEvent.click(screen.getByRole('button', { name: 'Run fixed Judge Demo' }))

    expect(await screen.findByText('530000')).toBeVisible()
    expect(screen.getByText(/Learned \/ not evaluated/)).toBeVisible()
    expect(screen.getByText('default / not ranked')).toBeVisible()
    expect(screen.getByText('Configured default baseline plan')).toBeVisible()
    expect(screen.getByRole('region', { name: 'Baseline raw event-suite aggregates' })).toHaveTextContent('Replan CVaR')
    expect(screen.getByText(/Synthetic proxy disclosure/)).toBeVisible()
    expect(screen.getByText('1 allocation changed by QP')).toBeVisible()
    expect(screen.getByText('Scientific claim status')).toBeVisible()
    expect(screen.getByText('Learned superiority is not established.')).toBeVisible()
    const commonState = screen.getByRole('region', { name: 'Common-state post-event strategy comparison' })
    expect(within(commonState).getByText('Continue original plan')).toBeVisible()
    expect(within(commonState).getByText('Baseline replan')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Continue original plan' }))
    expect(within(commonState).queryByText('Baseline replan')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Replan from current state' }))
    expect(within(commonState).getByText('Baseline replan')).toBeVisible()
    expect(within(commonState).queryByText('Continue original plan')).not.toBeInTheDocument()
    expect(screen.getByText('Verified matched action-removal explanation.')).toBeVisible()
    expect(screen.getByText('ALL EVENTS / RAW ROWS')).toBeVisible()
    expect(screen.getByLabelText('v5 learned checkpoint candidate')).toHaveValue('relational_gnn_ppo')
  })
})
