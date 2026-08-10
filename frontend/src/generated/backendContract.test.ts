import { describe, expect, it } from 'vitest'
import {
  actionOrder,
  actionSlices,
  requestLimits,
  services,
  SHOCK_IMPACTS,
} from './backendContract'
import {
  serviceIndex,
  shockImpactFor,
  SHOCK_IMPACTS as GAME_SHOCK_IMPACTS,
} from '../game/model'

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
})
