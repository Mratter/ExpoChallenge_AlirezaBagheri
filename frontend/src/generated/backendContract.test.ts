import { describe, expect, it } from 'vitest'
import {
  actionOrderV3,
  actionSlicesV3,
  requestLimitsV3,
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
    expect(actionOrderV3.slice(
      actionSlicesV3.preparednessInvestment.start,
      actionSlicesV3.preparednessInvestment.end,
    )).toEqual(services.map((service) => `preparedness_investment_${service}`))
    expect(requestLimitsV3.initialServices).toEqual({
      length: 5,
      minimum: 0.05,
      maximum: 0.95,
    })
    expect(requestLimitsV3.recoveryTargets).toEqual({
      length: 5,
      minimum: 0.45,
      maximum: 0.75,
    })
  })
})
