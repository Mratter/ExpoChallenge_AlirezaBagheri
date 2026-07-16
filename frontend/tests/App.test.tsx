import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App'
import type { CompareResponse, DayResult, Shock } from '../src/types'

const shock: Shock = {
  day: 1,
  type: null,
  severity: 0,
  impact: [0, 0, 0, 0, 0],
  budget_factor: 0,
  forced: false,
}

function day(dayNumber: number, resilience: number): DayResult {
  return {
    day: dayNumber,
    shock: dayNumber === 5 ? { ...shock, day: 5, type: 'utility', severity: 0.26, forced: true } : { ...shock, day: dayNumber },
    available_budget: dayNumber === 5 ? 165.96 : 180,
    services_before: [0.34, 0.26, 0.41, 0.38, 0.3],
    services_after_shock: [0.34, 0.26, 0.41, 0.38, 0.3],
    raw_action: [0, 0, 0, 0, 0],
    raw_proposal: [36, 36, 36, 36, 36],
    lower_bounds: [0, 0, 0, 0, 0],
    upper_bounds: [90, 90, 90, 90, 90],
    allocation: [36, 36, 36, 36, 36],
    projection: {
      distance: 0,
      sum: 180,
      constraint_violations: 0,
      violation_breakdown: {
        sum_violations: 0,
        budget_violations: 0,
        lower_violations: 0,
        upper_violations: 0,
        total: 0,
      },
      bindings: [],
    },
    planner_evidence: null,
    support: [0.7, 0.7, 0.7, 0.7, 0.7],
    gain: [0.01, 0.01, 0.01, 0.01, 0.01],
    strain: [0, 0, 0, 0, 0],
    services_end: [resilience, resilience, resilience, resilience, resilience],
    resilience,
    reward: resilience,
  }
}

function responseFixture(): CompareResponse {
  const candidateDays = Array.from({ length: 14 }, (_, index) => day(index + 1, 0.4 + index * 0.01))
  const baselineDays = Array.from({ length: 14 }, (_, index) => day(index + 1, 0.39 + index * 0.009))
  return {
    schema_version: '2.0.0',
    result_id: 'c'.repeat(64),
    persistence: { format: 'canonical-json-v1', idempotent: true, result_id: 'c'.repeat(64) },
    seed: 424242,
    generator: 'numpy.PCG64',
    scenario: {
      name: 'Central district restart',
      horizon_days: 14,
      daily_budget: 180,
      initial_services: [0.34, 0.26, 0.41, 0.38, 0.3],
      priorities: [1, 1.1, 1.2, 1.4, 1],
      shock_probability: 0.2,
      severity_min: 0.1,
      severity_max: 0.28,
      forced_shock: { day: 5, type: 'utility', severity: 0.26 },
    },
    services: ['transport', 'housing', 'food', 'healthcare', 'public_services'],
    shock_schedule: candidateDays.map((entry) => entry.shock),
    shock_schedule_sha256: 'a'.repeat(64),
    policy: {
      id: 'city-recovery-sb3-ppo-v1',
      artifact_type: 'stable_baselines3_ppo',
      algorithm: 'PPO',
      runtime: 'ONNX Runtime CPUExecutionProvider',
      sha256: 'b'.repeat(64),
      sb3_checkpoint_sha256: 'd'.repeat(64),
      parity_report_sha256: 'e'.repeat(64),
      disclosure: 'Stable-Baselines3 PPO trained only on authored synthetic scenarios.',
      legacy_candidate: {
        id: 'frozen-policy-candidate-v1',
        artifact_type: 'deterministic_linear_policy_candidate',
        is_ppo: false,
        sha256: 'a'.repeat(64),
        disclosure: 'Accepted linear candidate; not PPO.',
      },
    },
    baseline_spec: {
      id: 'ortools-glop-visible-v1',
      library: 'OR-Tools',
      library_version: '9.14.6206',
      solver: 'GLOP',
      objective: 'Visible immediate recovery objective',
      future_shocks_visible: false,
    },
    baseline: {
      planner: 'ortools_glop_baseline', rauc: 0.4485, final_resilience: 0.507,
      minimum_resilience: 0.39, post_shock_recovery_shortfall_auc: 0.02,
      days_to_pre_shock_recovery_after_largest_loss: 3, critical_service_days: 4,
      total_projection_distance: 1, constraint_violations: 0,
      trajectory: baselineDays,
    },
    candidate: {
      planner: 'stable_baselines3_ppo_onnx', rauc: 0.465, final_resilience: 0.53,
      minimum_resilience: 0.4, post_shock_recovery_shortfall_auc: 0.01,
      days_to_pre_shock_recovery_after_largest_loss: 2, critical_service_days: 2,
      total_projection_distance: 1, constraint_violations: 0,
      trajectory: candidateDays,
    },
    comparison: { primary_metric: 'weighted_daily_resilience_auc', candidate_minus_baseline: 0.0165, outcome: 'candidate_higher_rauc' },
    limitations: ['Synthetic only.'],
  }
}

describe('recovery desk', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      if (String(input) === '/api/v1/simulations') {
        return { ok: true, json: async () => ({ results: [] }) }
      }
      return { ok: true, json: async () => responseFixture() }
    }))
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('renders bounded scenario controls and computed evidence', async () => {
    render(<App />)
    expect(await screen.findByRole('heading', { name: 'Central district restart' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Run comparison' })).toBeEnabled()
    expect(screen.getByText('Candidate 0 / baseline 0')).toBeVisible()
    expect(screen.getByText('Measured constraint violations')).toBeVisible()
    expect(screen.getAllByText('SB3 PPO / ONNX')[0]).toBeVisible()
    expect(screen.getByText(/Against visible OR-Tools planner/i)).toBeVisible()
    expect(screen.getByText(/not PPO/i)).toBeVisible()
    expect(screen.getByLabelText('Transport initial state')).toHaveAttribute('min', '5')
    expect(screen.getByRole('checkbox', { name: /Force utility failure/i })).toBeChecked()
  })

  it('opens the full daily audit from the result view', async () => {
    render(<App />)
    await screen.findByRole('heading', { name: 'Central district restart' })
    fireEvent.click(screen.getByRole('tab', { name: 'Daily audit' }))
    expect(screen.getByRole('table', { name: 'Full deterministic daily comparison' })).toBeVisible()
    expect(screen.getAllByRole('row')).toHaveLength(15)
  })

  it('derives candidate and baseline violation totals from daily measurements', async () => {
    const fixture = responseFixture()
    fixture.candidate.trajectory[0].projection.constraint_violations = 2
    fixture.baseline.trajectory[0].projection.constraint_violations = 1
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      if (String(input) === '/api/v1/simulations') {
        return { ok: true, json: async () => ({ results: [] }) }
      }
      return { ok: true, json: async () => fixture }
    }))

    render(<App />)

    expect(await screen.findByText('Candidate 2 / baseline 1')).toBeVisible()
    expect(
      screen.getByLabelText('Measured constraint violations: candidate 2, baseline 1'),
    ).toBeVisible()
  })

  it.each([
    [422, 'INVALID_SCENARIO', 'Days must be at most 30.', 'Scenario invalid'],
    [503, 'DEPENDENCY_NOT_READY', 'Frozen policy is unavailable.', 'Comparison blocked'],
  ])(
    'clears prior evidence after a %s error',
    async (status, code, message, heading) => {
      let comparisonCount = 0
      const fetchMock = vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
        if (String(input) === '/api/v1/simulations') {
          return { ok: true, json: async () => ({ results: [] }) }
        }
        comparisonCount += 1
        if (comparisonCount === 1) {
          return { ok: true, json: async () => responseFixture() }
        }
        return {
          ok: false,
          status,
          json: async () => ({ error: { code, message, details: [] } }),
        }
      })
      vi.stubGlobal('fetch', fetchMock)
      render(<App />)
      await screen.findByRole('heading', { name: 'Central district restart' })

      fireEvent.change(screen.getByLabelText('Days'), { target: { value: '31' } })
      fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }))

      expect(await screen.findByRole('heading', { name: heading })).toBeVisible()
      expect(screen.getByRole('alert')).toHaveTextContent(message)
      expect(screen.queryByLabelText('Comparison summary')).not.toBeInTheDocument()
      expect(screen.queryByText('Measured constraint violations')).not.toBeInTheDocument()
      expect(screen.queryByText('Shock tape')).not.toBeInTheDocument()
    },
  )

  it('restores a persisted authored result from the saved-results menu', async () => {
    const restored = responseFixture()
    restored.scenario = { ...restored.scenario, name: 'Restored corridor scenario' }
    const resultId = 'f'.repeat(64)
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url === `/api/v1/simulations/${resultId}`) {
        return { ok: true, json: async () => restored }
      }
      if (url === '/api/v1/simulations') {
        return {
          ok: true,
          json: async () => ({
            results: [{
              result_id: resultId,
              seed: 91,
              scenario_name: 'Saved corridor',
              horizon_days: 14,
              candidate_rauc: 0.46,
              baseline_rauc: 0.44,
              outcome: 'candidate_higher_rauc',
              policy_sha256: 'b'.repeat(64),
            }],
          }),
        }
      }
      return { ok: true, json: async () => responseFixture() }
    }))
    render(<App />)
    await screen.findByRole('heading', { name: 'Central district restart' })

    fireEvent.change(screen.getByLabelText('Restore saved result'), {
      target: { value: resultId },
    })

    expect(await screen.findByRole('heading', { name: 'Restored corridor scenario' })).toBeVisible()
  })
})
