import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  AnalysisApiError,
  createCounterfactualRequest,
  fetchDecisionExplanations,
  normalizeShareVector,
  parseCounterfactual,
  parseExplanations,
  recoveryPlanUrl,
  runCounterfactualAnalysis,
} from './analysisApi'
import { actionOrder, observationOrder, services } from './types'

type JsonObject = Record<string, unknown>

const resultId = 'a'.repeat(64)
const policySha256 = 'b'.repeat(64)
const tapeSha256 = 'c'.repeat(64)
const expectation = { resultId, policySha256, shockScheduleSha256: tapeSha256 }
const comparisonIdentity = {
  result_id: resultId,
  policy: { sha256: policySha256 },
  shock_schedule_sha256: tapeSha256,
}

function explanationFixture(id = resultId): JsonObject {
  const days = Array.from({ length: 30 }, (_, dayIndex) => ({
    day: dayIndex + 1,
    base_raw_action: Array.from({ length: 22 }, () => 0),
    channels: observationOrder.map((name, observationIndex) => ({
      observation_index: observationIndex,
      observation_name: name,
      observed_value: observationIndex / 100,
      mean_absolute_action_delta: 1 / 73,
      normalized_influence: 1 / 73,
      influence_rank: observationIndex + 1,
      most_affected_action_index: observationIndex % actionOrder.length,
      most_affected_action: actionOrder[observationIndex % actionOrder.length],
      signed_action_delta: observationIndex % 2 === 0 ? 0.01 : -0.01,
    })),
  }))
  return {
    schema_version: '1.0.0',
    result_id: id,
    method: {
      id: 'single-channel-zero-occlusion-action-sensitivity-v1',
      description: 'Each channel is independently zeroed.',
      interpretation: 'Local action sensitivity, not causal attribution.',
      causal: false,
      occlusion_value: 0,
      batch_size_per_day: 73,
      normalization: 'within-day sum',
      future_tape_visible: false,
    },
    policy: { id: 'fixture-policy', sha256: policySha256 },
    shock_schedule_sha256: tapeSha256,
    future_tape_visible: false,
    day_count: 30,
    observation_count: 73,
    action_count: 22,
    observation_order: [...observationOrder],
    action_order: [...actionOrder],
    days,
  }
}

function outcomeFixture(solved: boolean): JsonObject {
  return {
    definition_id: 'city-recovery-solved-v3',
    definition_sha256: 'd'.repeat(64),
    solved,
    status: solved ? 'solved' : 'failed',
    checks: {
      zero_hard_violations: true,
      conservation_verified: true,
      assessment_tail_targets_met: solved,
      resilience_auc_met: true,
      critical_service_day_cap_met: true,
      terminal_pending_within_capacity: true,
    },
    recovery_targets: [0.55, 0.55, 0.55, 0.55, 0.55],
    target_met_by_service: [solved, solved, solved, solved, solved],
    assessment_tail_days: 3,
    tail_minimum_services: [0.58, 0.59, 0.6, 0.61, 0.62],
    resilience_auc: 0.49,
    resilience_auc_floor: 0.44,
    critical_service_days: 0,
    critical_service_day_cap: 12,
    hard_violation_count: 0,
    max_conservation_residual: 0,
    terminal_pending_arrivals: [0, 0, 0, 0, 0],
    terminal_pending_capacity: [10, 10, 10, 10, 10],
    reason_codes: [],
  }
}

function summaryFixture(solved: boolean): JsonObject {
  return {
    solved,
    rauc: 0.49,
    final_resilience: 0.7,
    minimum_resilience: 0.31,
    critical_service_days: 0,
    hard_violation_count: 0,
    absolute_outcome: outcomeFixture(solved),
    trajectory_sha256: solved ? 'e'.repeat(64) : 'f'.repeat(64),
  }
}

function counterfactualFixture(): JsonObject {
  return {
    schema_version: '1.0.0',
    result_id: resultId,
    analysis_id: '1'.repeat(64),
    analysis_only: true,
    persisted: false,
    policy_sha256: policySha256,
    shock_schedule_sha256: tapeSha256,
    same_disaster_tape: true,
    future_tape_visible: false,
    treatment: {
      day: 5,
      material_shares: [0.1, 0.15, 0.2, 0.25, 0.3],
      crew_shares: [0.2, 0.2, 0.2, 0.2, 0.2],
    },
    unchanged_prefix: {
      days: 4,
      original_sha256: '2'.repeat(64),
      counterfactual_sha256: '2'.repeat(64),
      matches: true,
    },
    selected_day_realized_allocations: {
      services: [...services],
      original: { material: [10, 20, 30, 40, 50], crew: [5, 10, 15, 20, 25] },
      counterfactual: { material: [15, 22.5, 30, 37.5, 45], crew: [15, 15, 15, 15, 15] },
    },
    original: summaryFixture(false),
    counterfactual: summaryFixture(true),
    daily_deltas: Array.from({ length: 30 }, (_, index) => ({
      day: index + 1,
      services_end: [0, 0, 0, 0, index === 29 ? 0.02 : 0],
      preparedness_end: [0, 0, 0, 0, 0],
      resilience: index < 4 ? 0 : 0.001,
      reward: index < 4 ? 0 : 0.002,
    })),
  }
}

function child(parent: JsonObject, key: string): JsonObject {
  return parent[key] as JsonObject
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('analysis evidence parsing', () => {
  it('accepts a complete 30-day, 73-channel explanation receipt', () => {
    const parsed = parseExplanations(explanationFixture(), expectation)
    expect(parsed.days).toHaveLength(30)
    expect(parsed.days[0].channels).toHaveLength(73)
    expect(parsed.days[0].channels[0].observation_name).toBe(observationOrder[0])
  })

  it('rejects a reordered or renamed observation channel', () => {
    const fixture = explanationFixture()
    const firstDay = (fixture.days as JsonObject[])[0]
    const firstChannel = (firstDay.channels as JsonObject[])[0]
    firstChannel.observation_name = observationOrder[1]
    expect(() => parseExplanations(fixture, expectation)).toThrowError(AnalysisApiError)
  })

  it('rejects duplicate ranks and influence totals that were tampered', () => {
    const duplicateRank = explanationFixture()
    const channels = ((duplicateRank.days as JsonObject[])[0].channels as JsonObject[])
    channels[1].influence_rank = 1
    expect(() => parseExplanations(duplicateRank, expectation)).toThrowError(/duplicate influence ranks/)

    const badTotal = explanationFixture()
    const badChannels = ((badTotal.days as JsonObject[])[0].channels as JsonObject[])
    badChannels[0].normalized_influence = 0.5
    expect(() => parseExplanations(badTotal, expectation)).toThrowError(/sum to one/)
  })

  it('rejects a causal claim or mismatched policy identity', () => {
    const causal = explanationFixture()
    child(causal, 'method').causal = true
    expect(() => parseExplanations(causal, expectation)).toThrowError(/safety contract/)

    const wrongPolicy = explanationFixture()
    child(wrongPolicy, 'policy').sha256 = '9'.repeat(64)
    expect(() => parseExplanations(wrongPolicy, expectation)).toThrowError(/policy SHA-256/)
  })

  it('accepts a replay-safe counterfactual and rejects a changed prefix', () => {
    const parsed = parseCounterfactual(counterfactualFixture(), expectation)
    expect(parsed.analysis_only).toBe(true)
    expect(parsed.counterfactual.solved).toBe(true)

    const changed = counterfactualFixture()
    child(changed, 'unchanged_prefix').counterfactual_sha256 = '3'.repeat(64)
    expect(() => parseCounterfactual(changed, expectation)).toThrowError(/changed evidence before/)
  })

  it('rejects persisted or cross-tape counterfactual evidence', () => {
    const persisted = counterfactualFixture()
    persisted.persisted = true
    expect(() => parseCounterfactual(persisted, expectation)).toThrowError(/safety and persistence/)

    const wrongTape = counterfactualFixture()
    wrongTape.shock_schedule_sha256 = '4'.repeat(64)
    expect(() => parseCounterfactual(wrongTape, expectation)).toThrowError(/disaster-tape SHA-256/)
  })
})

describe('counterfactual request construction', () => {
  it('normalizes non-negative relative weights', () => {
    expect(normalizeShareVector([1, 1, 2, 2, 4], 'Shares')).toEqual([
      0.1, 0.1, 0.2, 0.2, 0.4,
    ])
    expect(createCounterfactualRequest(5, { materialShares: [1, 1, 2, 2, 4] }))
      .toEqual({ day: 5, material_shares: [0.1, 0.1, 0.2, 0.2, 0.4] })
  })

  it('rejects invalid days, negative shares, and zero-total shares', () => {
    expect(() => createCounterfactualRequest(0, { materialShares: [1, 1, 1, 1, 1] }))
      .toThrowError(/Day/)
    expect(() => normalizeShareVector([1, 1, -1, 1, 1], 'Shares')).toThrowError(/negative/)
    expect(() => normalizeShareVector([0, 0, 0, 0, 0], 'Shares')).toThrowError(/positive total/)
  })
})

describe('analysis URLs and requests', () => {
  it('encodes result ids and names export query fields explicitly', () => {
    expect(recoveryPlanUrl('result/with spaces?', 'baseline', 'pdf')).toBe(
      '/api/v1/simulations/result%2Fwith%20spaces%3F/recovery-plan?planner=baseline&format=pdf',
    )
  })

  it('uses an encoded explanations URL and validates the response identity', async () => {
    const encodedId = 'result/with spaces?'
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(
      JSON.stringify(explanationFixture(encodedId)),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    ))
    vi.stubGlobal('fetch', fetchMock)

    await fetchDecisionExplanations({ ...comparisonIdentity, result_id: encodedId })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/simulations/result%2Fwith%20spaces%3F/explanations',
      { signal: undefined },
    )
  })

  it('posts only normalized one-day treatment fields', async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(
      JSON.stringify(counterfactualFixture()),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    ))
    vi.stubGlobal('fetch', fetchMock)
    const request = createCounterfactualRequest(5, {
      materialShares: [1, 1, 1, 1, 1],
      crewShares: [1, 2, 3, 4, 5],
    })

    await runCounterfactualAnalysis(comparisonIdentity, request)
    const init = fetchMock.mock.calls[0][1]
    expect(fetchMock.mock.calls[0][0]).toBe(
      `/api/v1/simulations/${resultId}/counterfactuals`,
    )
    expect(init).toMatchObject({ method: 'POST', headers: { 'Content-Type': 'application/json' } })
    expect(JSON.parse(String(init?.body))).toEqual(request)
  })
})
