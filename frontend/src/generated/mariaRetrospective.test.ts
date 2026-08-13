import { describe, expect, it } from 'vitest'
import { mariaRetrospective } from './mariaRetrospective'

const seriesKeys = ['historical', 'v4', 'reactive'] as const

describe('generated Hurricane Maria retrospective contract', () => {
  it('contains one complete day 0 through day 30 reconstruction', () => {
    expect(mariaRetrospective.days).toEqual(Array.from({ length: 31 }, (_, day) => day))
    expect(mariaRetrospective.dates).toHaveLength(31)
    expect(mariaRetrospective.serviceOrder).toEqual([
      'transport',
      'housing',
      'food',
      'healthcare',
      'public_services',
    ])
  })

  it('keeps every displayed trajectory finite, bounded, and aligned', () => {
    for (const key of seriesKeys) {
      const series = mariaRetrospective.series[key]
      expect(series.total).toHaveLength(31)
      expect(series.total.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)).toBe(true)
      for (const service of mariaRetrospective.serviceOrder) {
        expect(series.services[service]).toHaveLength(31)
        expect(series.services[service].every((value) => Number.isFinite(value) && value >= 0 && value <= 1)).toBe(true)
      }
    }
  })

  it('provides valid historical observation markers for each service', () => {
    for (const service of mariaRetrospective.serviceOrder) {
      const days = mariaRetrospective.observationDays[service]
      expect([...days].sort((a, b) => a - b)).toEqual(days)
      expect(new Set(days).size).toBe(days.length)
      expect(days.every((day) => Number.isInteger(day) && day >= 0 && day <= 30)).toBe(true)
    }
    expect(Object.values(mariaRetrospective.observationDays).flat().length).toBeGreaterThan(0)
  })

  it('contains the complete internally consistent seven-row benchmark', () => {
    expect(mariaRetrospective.benchmarkRows).toHaveLength(7)
    expect(mariaRetrospective.benchmarkRows.some((row) => /tuned/i.test(row.label))).toBe(true)
    expect(mariaRetrospective.benchmarkRows.some((row) => /oracle/i.test(row.label) && /privileged/i.test(row.classification))).toBe(true)
    for (const row of mariaRetrospective.benchmarkRows) {
      expect(row.total).toBe(200)
      expect(row.rate).toBeCloseTo(row.solved / row.total, 10)
    }
  })

  it('carries immutable evidence identities into the frontend', () => {
    const hashes = [
      mariaRetrospective.receiptSha256,
      mariaRetrospective.sourceManifestSha256,
      mariaRetrospective.reconstructionSha256,
      mariaRetrospective.artifactSha256,
    ]
    expect(hashes.every((hash) => /^[a-f0-9]{64}$/.test(hash))).toBe(true)
    expect(mariaRetrospective.caption).toMatch(/project-derived|project reconstruction/i)
    expect(mariaRetrospective.caption).toMatch(/not observed|not causal/i)
  })
})
