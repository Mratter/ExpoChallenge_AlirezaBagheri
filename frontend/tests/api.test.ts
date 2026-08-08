import { afterEach, describe, expect, it, vi } from 'vitest'
import { isWorkbenchOverview, loadWorkbenchOverview, OverviewError } from '../src/api'
import { measuredOverviewFixture, overviewFixture } from './fixtures'

afterEach(() => vi.unstubAllGlobals())

describe('workbench evidence client', () => {
  it('accepts the versioned overview contract', () => {
    expect(isWorkbenchOverview(overviewFixture())).toBe(true)
    expect(isWorkbenchOverview(measuredOverviewFixture())).toBe(true)
    expect(isWorkbenchOverview({ schema_version: 'model-workbench-v1', tracks: [] })).toBe(false)
  })

  it('rejects measured benchmarks whose objective or matched counts do not sum to the scenario total', () => {
    const badHeadToHead = measuredOverviewFixture()
    if (badHeadToHead.benchmark.status !== 'measured') throw new Error('Expected measured fixture')
    badHeadToHead.benchmark.head_to_head.ties = 5
    expect(isWorkbenchOverview(badHeadToHead)).toBe(false)

    const badObjective = measuredOverviewFixture()
    if (badObjective.benchmark.status !== 'measured') throw new Error('Expected measured fixture')
    badObjective.benchmark.objective.learned_policy.misses = 3
    expect(isWorkbenchOverview(badObjective)).toBe(false)
  })

  it('rejects a missing canonical synthetic disclosure', () => {
    const document = measuredOverviewFixture()
    if (document.benchmark.status !== 'measured') throw new Error('Expected measured fixture')
    document.benchmark.synthetic_disclosure = ''
    expect(isWorkbenchOverview(document)).toBe(false)
  })

  it('rejects malformed measured-evidence hashes and missing training units', () => {
    const badHash = measuredOverviewFixture()
    if (badHash.benchmark.status !== 'measured') throw new Error('Expected measured fixture')
    badHash.benchmark.provenance[0].sha256 = 'not-a-sha'
    expect(isWorkbenchOverview(badHash)).toBe(false)

    const missingUnit = measuredOverviewFixture()
    missingUnit.tracks[1].training.unit = ''
    expect(isWorkbenchOverview(missingUnit)).toBe(false)
  })

  it('rejects incompatible API documents instead of displaying stale claims', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ schema_version: 'wrong' }) }))
    await expect(loadWorkbenchOverview()).rejects.toBeInstanceOf(OverviewError)
  })
})
