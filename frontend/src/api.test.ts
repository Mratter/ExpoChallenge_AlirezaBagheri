import { describe, expect, it } from 'vitest'
import { ComparisonError, parseComparison, parseMetadata } from './api'
import { actionGroupsV3, actionOrderV3, observationOrderV3 } from './types'

type JsonObject = Record<string, unknown>
type Tamper = (fixture: JsonObject) => void

function child(parent: JsonObject, key: string): JsonObject {
  return parent[key] as JsonObject
}

/** Pure parser fixture. These illustrative values are not release evidence. */
function metadataFixture(): JsonObject {
  return {
    schema_version: '4.0.0',
    services: ['transport', 'housing', 'food', 'healthcare', 'public_services'],
    model: {
      id: 'city-recovery-sb3-ppo-v3-selected',
      version: '3.0.0',
      schema_version: '4.0.0',
      manifest_schema_version: 1,
      artifact_type: 'stable_baselines3_ppo',
      algorithm: 'PPO',
      training_library: 'stable-baselines3',
      observation_count: 73,
      action_count: 22,
      observation_order: [...observationOrderV3],
      action_order: [...actionOrderV3],
      action_groups: [...actionGroupsV3],
      trainable_parameters: 322_733,
      requested_training_transitions: 645_120,
      completed_training_transitions: 368_640,
      onnx_sha256: '1'.repeat(64),
      selected_checkpoint_sha256: '2'.repeat(64),
      manifest_sha256: '3'.repeat(64),
      parity_sha256: '4'.repeat(64),
      scientific_source_sha256: '5'.repeat(64),
    },
    environment: {
      id: 'CityRecoveryEnv-v3',
      version: '3.0.0',
      observation_count: 73,
      action_count: 22,
      spec_sha256: '6'.repeat(64),
      outcome_definition_sha256: '7'.repeat(64),
      policy_neutral_transition: true,
      future_tape_visible: false,
    },
    baseline: {
      id: 'reactive-public-state-heuristic-v3',
      version: '3.0.0',
      uses_same_observation_contract: true,
      uses_same_action_contract: true,
      uses_public_risk_signal: true,
      future_tape_visible: false,
    },
    benchmark: {
      artifact_sha256: 'a'.repeat(64),
      artifact_type: 'city_recovery_v3_final_benchmark',
      status: 'complete',
      primary_metric: 'independent_absolute_disasters_solved',
      synthetic_case_count: 40,
      candidate: {
        id: 'city-recovery-sb3-ppo-v3-selected',
        solved_count: 22,
        failed_count: 18,
        solve_rate: 22 / 40,
        solve_rate_wilson_ci95: [0.39829092, 0.69294692],
        mean_resilience_auc: 0.51,
      },
      baseline: {
        id: 'reactive-public-state-heuristic-v3',
        version: '3.0.0',
        solved_count: 18,
        failed_count: 22,
        solve_rate: 18 / 40,
        solve_rate_wilson_ci95: [0.30705308, 0.60170908],
        mean_resilience_auc: 0.47,
      },
      paired_absolute_outcomes: {
        both_solved: 14,
        ppo_only: 8,
        heuristic_only: 4,
        neither: 14,
      },
      secondary_head_to_head_resilience_auc: {
        candidate_wins: 21,
        baseline_wins: 17,
        ties: 2,
        candidate_mean_minus_baseline_mean: 0.04,
      },
      invariants: {
        all_rows_present: true,
        unique_rows: true,
        same_tapes: true,
        deterministic_replay_mismatches: 0,
        candidate_hard_violations: 0,
        baseline_hard_violations: 0,
        maximum_conservation_residual: 0,
      },
      rows_sha256: 'b'.repeat(64),
      ledger_sha256: 'c'.repeat(64),
    },
  }
}

function expectTamperRejected(tamper: Tamper): void {
  const fixture = metadataFixture()
  tamper(fixture)
  expect(() => parseMetadata(fixture)).toThrowError(ComparisonError)
}

const metadataTamperCases: Array<[string, Tamper]> = [
  ['a string manifest schema version', (fixture) => {
    child(fixture, 'model').manifest_schema_version = '1'
  }],
  ['a stale selected-model ID', (fixture) => {
    child(fixture, 'model').id = 'city-recovery-ppo-v3'
  }],
  ['an incompatible result schema', (fixture) => {
    fixture.schema_version = '3.0.0'
  }],
  ['a mismatched model schema', (fixture) => {
    child(fixture, 'model').schema_version = '3.0.0'
  }],
  ['an unregistered selected transition count', (fixture) => {
    child(fixture, 'model').completed_training_transitions = 100_000
  }],
  ['a changed trainable parameter count', (fixture) => {
    child(fixture, 'model').trainable_parameters = 322_734
  }],
  ['a reordered public observation', (fixture) => {
    const order = child(fixture, 'model').observation_order as string[]
    ;[order[0], order[1]] = [order[1], order[0]]
  }],
  ['a changed action group', (fixture) => {
    const groups = child(fixture, 'model').action_groups as string[]
    groups[5] = 'future_override'
  }],
  ['a mismatched environment version', (fixture) => {
    child(fixture, 'environment').version = '3.1.0'
  }],
  ['a changed environment observation count', (fixture) => {
    child(fixture, 'environment').observation_count = 72
  }],
  ['a non-neutral environment flag', (fixture) => {
    child(fixture, 'environment').policy_neutral_transition = false
  }],
  ['future-tape visibility', (fixture) => {
    child(fixture, 'environment').future_tape_visible = true
  }],
  ['a changed baseline ID', (fixture) => {
    child(fixture, 'baseline').id = 'weak-heuristic'
  }],
  ['a changed baseline version', (fixture) => {
    child(fixture, 'baseline').version = '2.1.0'
  }],
  ['a baseline without the shared action contract', (fixture) => {
    child(fixture, 'baseline').uses_same_action_contract = false
  }],
  ['a baseline without the shared public-risk signal', (fixture) => {
    child(fixture, 'baseline').uses_public_risk_signal = false
  }],
  ['a baseline with future-tape visibility', (fixture) => {
    child(fixture, 'baseline').future_tape_visible = true
  }],
]

const benchmarkTamperCases: Array<[string, Tamper]> = [
  ['an extra unregistered benchmark field', (fixture) => {
    child(fixture, 'benchmark').claim = 'unsealed'
  }],
  ['a wrong benchmark artifact type', (fixture) => {
    child(fixture, 'benchmark').artifact_type = 'developmental_benchmark'
  }],
  ['an incomplete benchmark status', (fixture) => {
    child(fixture, 'benchmark').status = 'partial'
  }],
  ['a substituted primary metric', (fixture) => {
    child(fixture, 'benchmark').primary_metric = 'head_to_head_wins'
  }],
  ['a non-40 benchmark scope', (fixture) => {
    child(fixture, 'benchmark').synthetic_case_count = 39
  }],
  ['a changed benchmark candidate ID', (fixture) => {
    child(child(fixture, 'benchmark'), 'candidate').id = 'city-recovery-sb3-ppo-v3'
  }],
  ['a changed benchmark baseline version', (fixture) => {
    child(child(fixture, 'benchmark'), 'baseline').version = '3.0.1'
  }],
  ['a changed benchmark baseline ID', (fixture) => {
    child(child(fixture, 'benchmark'), 'baseline').id = 'reactive-private-state-heuristic-v3'
  }],
  ['a fractional solved count', (fixture) => {
    child(child(fixture, 'benchmark'), 'candidate').solved_count = 22.5
  }],
  ['a solve rate that disagrees with its count', (fixture) => {
    child(child(fixture, 'benchmark'), 'candidate').solve_rate = 0.9
  }],
  ['a tampered Wilson interval', (fixture) => {
    child(child(fixture, 'benchmark'), 'candidate').solve_rate_wilson_ci95 = [0.4, 0.7]
  }],
  ['a malformed Wilson interval', (fixture) => {
    child(child(fixture, 'benchmark'), 'candidate').solve_rate_wilson_ci95 = [0.4]
  }],
  ['a missing resilience mean', (fixture) => {
    delete child(child(fixture, 'benchmark'), 'candidate').mean_resilience_auc
  }],
  ['a paired table that does not cover 40 cases', (fixture) => {
    child(child(fixture, 'benchmark'), 'paired_absolute_outcomes').neither = 13
  }],
  ['paired counts that disagree with candidate solved count', (fixture) => {
    const pairs = child(child(fixture, 'benchmark'), 'paired_absolute_outcomes')
    pairs.both_solved = 13
    pairs.neither = 15
  }],
  ['head-to-head counts that do not cover 40 cases', (fixture) => {
    child(child(fixture, 'benchmark'), 'secondary_head_to_head_resilience_auc').ties = 1
  }],
  ['a resilience-AUC delta that disagrees with the means', (fixture) => {
    child(child(fixture, 'benchmark'), 'secondary_head_to_head_resilience_auc')
      .candidate_mean_minus_baseline_mean = 0.041
  }],
  ['a replay mismatch', (fixture) => {
    child(child(fixture, 'benchmark'), 'invariants').deterministic_replay_mismatches = 1
  }],
  ['an incomplete-row invariant', (fixture) => {
    child(child(fixture, 'benchmark'), 'invariants').all_rows_present = false
  }],
  ['a candidate hard violation', (fixture) => {
    child(child(fixture, 'benchmark'), 'invariants').candidate_hard_violations = 1
  }],
  ['a baseline hard violation', (fixture) => {
    child(child(fixture, 'benchmark'), 'invariants').baseline_hard_violations = 1
  }],
  ['a conservation residual above tolerance', (fixture) => {
    child(child(fixture, 'benchmark'), 'invariants').maximum_conservation_residual = 1.0001e-6
  }],
]

describe('V3 API fail-closed parsing', () => {
  it('accepts a complete, internally consistent selected-release contract', () => {
    const parsed = parseMetadata(metadataFixture())
    expect(parsed.model.id).toBe('city-recovery-sb3-ppo-v3-selected')
    expect(parsed.model.manifest_schema_version).toBe(1)
    expect(parsed.benchmark.synthetic_case_count).toBe(40)
    expect(parsed.model.observation_order).toEqual(observationOrderV3)
    expect(parsed.model.action_order).toEqual(actionOrderV3)
  })

  it.each(metadataTamperCases)('rejects metadata with %s', (_, tamper) => {
    expectTamperRejected(tamper)
  })

  it.each(benchmarkTamperCases)('rejects benchmark evidence with %s', (_, tamper) => {
    expectTamperRejected(tamper)
  })

  it('rejects incompatible comparison payloads before they reach the UI', () => {
    expect(() => parseComparison({ schema_version: '3.0.0', engine_version: 'incompatible-engine' }))
      .toThrowError(/not a CityRecoveryEnv-v3 schema 4 result/)
  })
})
