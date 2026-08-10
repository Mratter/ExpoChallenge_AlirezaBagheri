import { describe, expect, it } from 'vitest'
import { ComparisonError, parseComparison, parseMetadata } from './api'
import { actionGroups, actionOrder, observationOrder } from './types'

type JsonObject = Record<string, unknown>

function metadataFixture(): JsonObject {
  const observationContract = {
    source: 'raw_environment_observation',
    input_name: 'observation',
    dtype: 'float32',
    shape: [73],
    normalization: 'embedded_in_onnx',
  }
  return {
    app: 'Autonomous City Recovery Planner',
    version: '2.0.0',
    schema_version: '4.0.0',
    default_seed: 424242,
    services: ['transport', 'housing', 'food', 'healthcare', 'public_services'],
    model: {
      id: 'selected-policy',
      path_stem: 'selected-policy',
      artifact_type: 'onnx_policy',
      runtime: 'ONNX Runtime CPUExecutionProvider',
      sha256: '1'.repeat(64),
      observation_contract: observationContract,
      observation_count: 73,
      action_count: 22,
      observation_order: [...observationOrder],
      action_order: [...actionOrder],
      action_groups: [...actionGroups],
    },
    environment: {
      id: 'CityRecoveryEnv-v3',
      version: '3.0.0',
      observation_count: 73,
      action_count: 22,
      spec_sha256: '2'.repeat(64),
      policy_neutral_transition: true,
      future_tape_visible: false,
    },
    outcome_definition: { id: 'city-recovery-solved-v3' },
    outcome_definition_sha256: '3'.repeat(64),
    baseline: {
      id: 'reactive-public-state-heuristic-v3',
      version: '3.0.0',
      uses_same_observation_contract: true,
      uses_same_action_contract: true,
      uses_public_risk_signal: true,
      future_tape_visible: false,
    },
    persistence: { format: 'canonical-json-v1', idempotent: true },
    determinism: 'NumPy PCG64 and single-thread ONNX Runtime',
  }
}

function child(parent: JsonObject, key: string): JsonObject {
  return parent[key] as JsonObject
}

function expectMetadataRejected(change: (fixture: JsonObject) => void): void {
  const fixture = metadataFixture()
  change(fixture)
  expect(() => parseMetadata(fixture)).toThrowError(ComparisonError)
}

describe('API runtime contract parsing', () => {
  it('accepts the lean backend metadata response without release-lineage fields', () => {
    const parsed = parseMetadata(metadataFixture())
    expect(parsed.model.id).toBe('selected-policy')
    expect(parsed.model.observation_order).toEqual(observationOrder)
    expect(parsed.model.action_order).toEqual(actionOrder)
    expect(parsed.model.sha256).toBe('1'.repeat(64))
  })

  it('rejects an incompatible observation order', () => {
    expectMetadataRejected((fixture) => {
      const order = child(fixture, 'model').observation_order as string[]
      ;[order[0], order[1]] = [order[1], order[0]]
    })
  })

  it('rejects a policy whose raw observation contract is incompatible', () => {
    expectMetadataRejected((fixture) => {
      child(child(fixture, 'model'), 'observation_contract').normalization = 'missing'
    })
  })

  it('rejects incompatible environment dimensions', () => {
    expectMetadataRejected((fixture) => {
      child(fixture, 'environment').action_count = 21
    })
  })

  it('rejects future-tape visibility', () => {
    expectMetadataRejected((fixture) => {
      child(fixture, 'environment').future_tape_visible = true
    })
  })

  it('rejects incompatible comparison payloads before they reach the UI', () => {
    expect(() => parseComparison({ schema_version: '3.0.0', engine_version: 'incompatible-engine' }))
      .toThrowError(/configured environment schema/)
  })
})
