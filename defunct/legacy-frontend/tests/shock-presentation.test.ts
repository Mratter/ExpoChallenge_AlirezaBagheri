import { describe, expect, it } from 'vitest'
import { SHOCK_DISPLAY_NAMES, shockDisplayName } from '../src/shockPresentation'

describe('shock presentation names', () => {
  it('renames only the frozen aftershock key for visible copy', () => {
    expect(shockDisplayName('aftershock')).toBe('Earthquake')
    expect(SHOCK_DISPLAY_NAMES).toEqual({
      aftershock: 'Earthquake',
      supply: 'Supply',
      epidemic: 'Epidemic',
      utility: 'Utility',
      weather: 'Weather',
    })
  })

  it('keeps empty and legacy-safe fallback labels readable', () => {
    expect(shockDisplayName(null)).toBe('No shock')
    expect(shockDisplayName('legacy_event')).toBe('Legacy event')
  })
})
