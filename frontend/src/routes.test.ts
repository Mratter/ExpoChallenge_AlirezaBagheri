import { describe, expect, it } from 'vitest'
import { hashForRoute, routeFromHashValue, titleForRoute } from './routes'

describe('application routes', () => {
  it.each([
    ['#/toolbox', 'toolbox'],
    ['#/game', 'game'],
    ['#/', 'landing'],
    ['', 'landing'],
    ['#/unknown', 'landing'],
    ['#/game/extra', 'landing'],
    ['#/toolbox/extra', 'landing'],
  ] as const)('routes %s to %s', (hash, expected) => {
    expect(routeFromHashValue(hash)).toBe(expected)
  })

  it('uses the canonical root hash for the landing page', () => {
    expect(hashForRoute('landing')).toBe('#/')
    expect(hashForRoute('toolbox')).toBe('#/toolbox')
    expect(hashForRoute('game')).toBe('#/game')
  })

  it('provides route-specific document titles', () => {
    expect(titleForRoute('landing')).toContain('Evidence-led')
    expect(titleForRoute('toolbox')).toContain('Analyst Toolbox')
    expect(titleForRoute('game')).toContain('3D city')
  })
})
