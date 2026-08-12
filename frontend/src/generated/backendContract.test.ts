import { describe, expect, it } from 'vitest'
import {
  actionOrder,
  actionSlices,
  requestLimits,
  sectorPalette,
  services,
  SHOCK_IMPACTS,
} from './backendContract'
import {
  DISTRICTS,
  serviceIndex,
  shockImpactFor,
  SHOCK_IMPACTS as GAME_SHOCK_IMPACTS,
} from '../game/model'

function relativeLuminance(color: string): number {
  const channels = [1, 3, 5].map(
    (index) => Number.parseInt(color.slice(index, index + 2), 16) / 255,
  )
  const linear = channels.map((channel) => (
    channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4
  ))
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
}

function contrastRatio(first: string, second: string): number {
  const [darker, lighter] = [relativeLuminance(first), relativeLuminance(second)]
    .sort((left, right) => left - right)
  return (lighter + 0.05) / (darker + 0.05)
}

describe('generated backend contract', () => {
  it('is the game model single source for service order and shock impacts', () => {
    expect(GAME_SHOCK_IMPACTS).toBe(SHOCK_IMPACTS)
    services.forEach((service, index) => {
      expect(serviceIndex(service)).toBe(index)
    })
    expect(shockImpactFor('aftershock', 'housing')).toBe(1)
    expect(shockImpactFor('supply', 'food')).toBe(1)
    expect(shockImpactFor('epidemic', 'healthcare')).toBe(1)
    expect(shockImpactFor('utility', 'public_services')).toBe(1)
  })

  it('keeps action slices and custom validator limits available to consumers', () => {
    expect(actionOrder.slice(
      actionSlices.preparednessInvestment.start,
      actionSlices.preparednessInvestment.end,
    )).toEqual(services.map((service) => `preparedness_investment_${service}`))
    expect(requestLimits.initialServices).toEqual({
      length: 5,
      minimum: 0.05,
      maximum: 0.95,
    })
    expect(requestLimits.recoveryTargets).toEqual({
      length: 5,
      minimum: 0.45,
      maximum: 0.75,
    })
  })

  it('derives one contrast-safe UI color per diorama body color', () => {
    const paper = '#f7f8f8'
    const darkenFactor = 0.76
    services.forEach((service) => {
      const colors = sectorPalette[service]
      const expectedUi = `#${[1, 3, 5]
        .map((index) => (
          Math.round(Number.parseInt(colors.body.slice(index, index + 2), 16) * darkenFactor)
            .toString(16)
            .padStart(2, '0')
        ))
        .join('')}`
      expect(colors.ui).toBe(expectedUi)
      expect(contrastRatio(colors.ui, paper)).toBeGreaterThanOrEqual(3)
      const district = DISTRICTS.find((candidate) => candidate.service === service)
      expect(district).toMatchObject({ accent: colors.accent, body: colors.body })
    })
  })
})
