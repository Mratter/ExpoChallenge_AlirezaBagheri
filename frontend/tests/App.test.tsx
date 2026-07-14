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
    raw_proposal: [36, 36, 36, 36, 36],
    allocation: [36, 36, 36, 36, 36],
    projection: { distance: 0, sum: 180, constraint_violations: 0, bindings: [] },
    support: [0.7, 0.7, 0.7, 0.7, 0.7],
    gain: [0.01, 0.01, 0.01, 0.01, 0.01],
    strain: [0, 0, 0, 0, 0],
    services_end: [resilience, resilience, resilience, resilience, resilience],
    resilience,
  }
}

function responseFixture(): CompareResponse {
  const candidateDays = Array.from({ length: 14 }, (_, index) => day(index + 1, 0.4 + index * 0.01))
  const baselineDays = Array.from({ length: 14 }, (_, index) => day(index + 1, 0.39 + index * 0.009))
  return {
    schema_version: '1.0.0',
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
      id: 'frozen-policy-candidate-v1',
      artifact_type: 'deterministic_linear_policy_candidate',
      sha256: 'b'.repeat(64),
      disclosure: 'Deterministic grid-selected linear heuristic; not PPO.',
    },
    baseline: {
      planner: 'urgency_baseline', rauc: 0.4485, final_resilience: 0.507,
      minimum_resilience: 0.39, total_projection_distance: 1, constraint_violations: 0,
      trajectory: baselineDays,
    },
    candidate: {
      planner: 'frozen_policy', rauc: 0.465, final_resilience: 0.53,
      minimum_resilience: 0.4, total_projection_distance: 1, constraint_violations: 0,
      trajectory: candidateDays,
    },
    comparison: { primary_metric: 'weighted_daily_resilience_auc', candidate_minus_baseline: 0.0165, outcome: 'candidate_higher_rauc' },
    limitations: ['Synthetic only.'],
  }
}

describe('recovery desk', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => responseFixture(),
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
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => fixture,
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
      const fetchMock = vi.fn()
        .mockResolvedValueOnce({ ok: true, json: async () => responseFixture() })
        .mockResolvedValueOnce({
          ok: false,
          status,
          json: async () => ({ error: { code, message, details: [] } }),
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
})
