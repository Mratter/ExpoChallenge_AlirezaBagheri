import { describe, expect, it } from 'vitest'
import { preparednessResourcesForDay } from './DecisionAnalysis'
import type { DayResult } from './types'

describe('preparedness sustainability evidence', () => {
  it('reports persisted resources consumed rather than dimensionless gate fractions', () => {
    const day = {
      preparedness_investment: [1, 1, 1, 1, 1],
      logistics: {
        preparedness_material_consumed: [1, 2, 3, 4, 5],
        preparedness_crew_utilized: [0.5, 1, 1.5, 2, 2.5],
      },
    } as unknown as DayResult

    expect(preparednessResourcesForDay(day)).toEqual({ material: 15, crew: 7.5 })
  })
})
