import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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
    recommendations: {
      winner: 'candidate',
      winner_label: 'SB3 PPO / ONNX',
      winner_margin_pp: 1.65,
      winner_rationale: 'The learned policy outperformed the conventional planner by 1.65 resilience AUC percentage points.',
      critical_moment: { day: 5, resilience: 0.4, description: 'Day 5 recorded the lowest resilience (0.40) following a utility failure shock at 0.26 severity.' },
      most_fragile_service: 'healthcare',
      most_fragile_days_below_threshold: 3,
      worst_shock_type: 'utility',
      strategy_summary: 'Across 14 days with 180 daily units, the SB3 PPO / ONNX strategy is recommended. Resilience AUC: candidate 0.4650 vs baseline 0.4485. Most fragile service: healthcare. Critical moment: day 5.',
      actionable_recommendations: [
        'Adopt the SB3 PPO / ONNX allocation strategy for this scenario family; it yields the higher resilience trajectory.',
        'Reinforce healthcare capacity early: it spent 3 day(s) below the 0.30 stability band.',
        'Maintain reserve units for utility failure events: they caused the largest single-day service losses.',
      ],
      daily: candidateDays.map((entry) => ({
        day: entry.day,
        priority_service: 'healthcare',
        priority_rationale: `Prioritize healthcare: lowest condition relative to weight (${entry.services_end[3].toFixed(2)} state, 1.4 weight).`,
        risk_alerts: entry.services_end[3] < 0.12
          ? [{ service: 'healthcare', level: 'critical', detail: 'healthcare below 0.12 recovery floor' }]
          : [],
        allocation_focus: 'healthcare',
        allocation_focus_share: 0.25,
      })),
    },
    limitations: ['Synthetic only.'],
  }
}

describe('Analyst Toolbox', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '#/toolbox')
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      if (String(input) === '/api/v1/simulations') {
        return { ok: true, json: async () => ({ results: [] }) }
      }
      return { ok: true, json: async () => responseFixture() }
    }))
  })

  afterEach(() => {
    cleanup()
    window.history.replaceState(null, '', '#/toolbox')
    vi.unstubAllGlobals()
  })

  it('renders bounded scenario controls and computed evidence', async () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Analyst Toolbox' })).toBeVisible()
    expect(await screen.findByRole('heading', { name: 'Central district restart' })).toBeVisible()
    expect(screen.queryByText('Draft changed')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Run comparison' })).toBeEnabled()
    expect(screen.getByText('Candidate 0 / baseline 0')).toBeVisible()
    expect(screen.getByText('Measured constraint violations')).toBeVisible()
    expect(screen.getAllByText('SB3 PPO / ONNX')[0]).toBeVisible()
    expect(screen.getByText('Baseline resilience AUC')).toBeVisible()
    expect(screen.getByText('Visible OR-Tools GLOP')).toBeVisible()
    expect(screen.getByText('Measured delta +1.65 pp')).toBeVisible()
    expect(screen.getByText(/not PPO/i)).toBeVisible()
    expect(screen.getByLabelText('Transport initial state')).toHaveAttribute('min', '5')
    expect(screen.getByRole('checkbox', { name: /Force utility failure/i })).toBeChecked()
    expect(screen.getAllByLabelText(/End state: candidate/i)).toHaveLength(5)
  })

  it('opens the full daily audit from the result view', async () => {
    render(<App />)
    await screen.findByRole('heading', { name: 'Central district restart' })
    fireEvent.click(screen.getByRole('tab', { name: 'Daily audit' }))
    expect(screen.getByRole('table', { name: 'Full deterministic daily comparison' })).toBeVisible()
    expect(screen.getAllByRole('row')).toHaveLength(15)
    const scrollRegion = screen.getByRole('region', { name: 'Scrollable daily comparison table' })
    scrollRegion.focus()
    expect(scrollRegion).toHaveFocus()
  })

  it('implements keyboard navigation and labelled panels for result tabs', async () => {
    render(<App />)
    await screen.findByRole('heading', { name: 'Central district restart' })

    const trajectoryTab = screen.getByRole('tab', { name: 'Trajectory' })
    const auditTab = screen.getByRole('tab', { name: 'Daily audit' })
    expect(trajectoryTab).toHaveAttribute('tabindex', '0')
    expect(auditTab).toHaveAttribute('tabindex', '-1')

    trajectoryTab.focus()
    fireEvent.keyDown(trajectoryTab, { key: 'ArrowRight' })

    expect(auditTab).toHaveAttribute('aria-selected', 'true')
    expect(auditTab).toHaveAttribute('tabindex', '0')
    expect(auditTab).toHaveFocus()
    expect(screen.getByRole('tabpanel', { name: 'Daily audit' })).toBeVisible()
    expect(screen.getByRole('table', { name: 'Full deterministic daily comparison' })).toBeVisible()
  })

  it('keeps the primary action focused while a recompute is pending', async () => {
    render(<App />)
    await screen.findByRole('heading', { name: 'Central district restart' })
    const runButton = screen.getByRole('button', { name: 'Run comparison' })

    vi.stubGlobal('fetch', vi.fn().mockImplementation(() => new Promise(() => undefined)))
    runButton.focus()
    fireEvent.click(runButton)

    expect(runButton).toHaveFocus()
    expect(runButton).toHaveAttribute('aria-disabled', 'true')
    const resetButton = screen.getByRole('button', { name: 'Reset fixture' })
    expect(resetButton).toHaveAttribute('aria-disabled', 'true')
    fireEvent.click(resetButton)
    expect(screen.getByLabelText('Comparison summary')).toBeVisible()
    expect(screen.getByText('Running')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('Recomputing both trajectories')
  })

  it('clears computed evidence to an actionable empty state when the fixture resets', async () => {
    render(<App />)
    await screen.findByRole('heading', { name: 'Central district restart' })

    fireEvent.click(screen.getByRole('button', { name: 'Reset fixture' }))

    expect(screen.getByRole('heading', { name: 'No trajectory yet' })).toBeVisible()
    expect(screen.queryByLabelText('Comparison summary')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Run comparison' })).toBeVisible()
  })

  it('labels prior evidence when the authored draft changes', async () => {
    render(<App />)
    await screen.findByRole('heading', { name: 'Central district restart' })

    fireEvent.change(screen.getByLabelText('Seed'), { target: { value: '424243' } })

    expect(screen.getByText('Draft changed')).toBeVisible()
    expect(screen.getByText('Run to refresh evidence')).toBeVisible()
    expect(screen.getByLabelText('Comparison summary')).toBeVisible()
    expect(screen.getByRole('button', { name: 'City view' })).toBeDisabled()
  })

  it('boots the game route into setup without a comparison and applies the selected preset on Start', async () => {
    window.history.replaceState(null, '', '#/game')
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => null)
    const fetchMock = vi.mocked(fetch)

    render(<App />)

    expect(screen.getByRole('heading', { name: 'Put the city through a recovery run.' })).toBeVisible()
    expect(fetchMock).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('radio', { name: /Severe/i }))
    fireEvent.click(screen.getByRole('button', { name: 'Start Stress Test' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    const request = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string)
    expect(request.scenario).toMatchObject({
      shock_probability: 0.34,
      severity_min: 0.18,
      severity_max: 0.4,
      daily_budget: 140,
      forced_shock: null,
      forced_shocks: [],
    })
    expect(await screen.findByRole('heading', { name: 'This browser could not start WebGL.' })).toBeVisible()
    expect(screen.getByLabelText('6 disasters remaining')).toBeVisible()
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
    [422, 'INVALID_SCENARIO', 'Days must be at most 30.', 'Scenario invalid', 'Review scenario controls'],
    [503, 'DEPENDENCY_NOT_READY', 'Frozen policy is unavailable.', 'Comparison blocked', 'Try again'],
  ])(
    'clears prior evidence after a %s error',
    async (status, code, message, heading, action) => {
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
      const alert = screen.getByRole('alert')
      expect(alert).toHaveTextContent(message)
      await waitFor(() => expect(alert).toHaveFocus())
      expect(screen.queryByLabelText('Comparison summary')).not.toBeInTheDocument()
      expect(screen.queryByText('Measured constraint violations')).not.toBeInTheDocument()
      expect(screen.queryByText('Shock tape')).not.toBeInTheDocument()
      expect(screen.getByRole('button', { name: action })).toBeVisible()
    },
  )

  it('routes a cross-field scenario error back to the named control group', async () => {
    let comparisonCount = 0
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      if (String(input) === '/api/v1/simulations') {
        return { ok: true, json: async () => ({ results: [] }) }
      }
      comparisonCount += 1
      if (comparisonCount === 1) {
        return { ok: true, json: async () => responseFixture() }
      }
      return {
        ok: false,
        status: 422,
        json: async () => ({
          error: {
            code: 'INVALID_SCENARIO',
            message: 'Scenario validation failed.',
            details: [{ message: 'Value error, severity_min must be less than severity_max' }],
          },
        }),
      }
    }))

    render(<App />)
    await screen.findByRole('heading', { name: 'Central district restart' })
    fireEvent.change(screen.getByLabelText('Severity min %'), { target: { value: '25' } })
    fireEvent.change(screen.getByLabelText('Severity max %'), { target: { value: '10' } })
    fireEvent.click(screen.getByRole('button', { name: 'Run comparison' }))

    const review = await screen.findByRole('button', { name: 'Review scenario controls' })
    fireEvent.click(review)
    expect(screen.getByLabelText('Severity min %')).toHaveFocus()
  })

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

  it('switches from the Toolbox route to the WebGL-safe city route and back', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => null)
    render(<App />)
    await screen.findByRole('heading', { name: 'Central district restart' })

    fireEvent.click(screen.getByRole('button', { name: 'City view' }))

    expect(await screen.findByRole('heading', { name: 'This browser could not start WebGL.' })).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Open Analyst Toolbox' }))
    expect(await screen.findByRole('heading', { name: 'Central district restart' })).toBeVisible()
    expect(window.location.hash).toBe('#/toolbox')
  })
})
