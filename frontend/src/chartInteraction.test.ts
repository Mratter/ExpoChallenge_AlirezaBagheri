import { describe, expect, it } from 'vitest'
import { dayFromChartPointer } from './chartInteraction'

describe('shared chart pointer scale', () => {
  const scale = {
    boundsLeft: 100,
    boundsWidth: 824,
    dayStart: 0,
    dayEnd: 30,
    plotLeft: 52,
    plotWidth: 736,
    viewWidth: 824,
  }

  it('maps the exact plot edges and midpoint to days', () => {
    expect(dayFromChartPointer({ ...scale, clientX: 152 })).toBe(0)
    expect(dayFromChartPointer({ ...scale, clientX: 520 })).toBe(15)
    expect(dayFromChartPointer({ ...scale, clientX: 888 })).toBe(30)
  })

  it('clamps pointer positions outside the plot', () => {
    expect(dayFromChartPointer({ ...scale, clientX: -1000 })).toBe(0)
    expect(dayFromChartPointer({ ...scale, clientX: 5000 })).toBe(30)
  })

  it('fails closed to the first day before layout is measurable', () => {
    expect(dayFromChartPointer({ ...scale, clientX: 500, boundsWidth: 0 })).toBe(0)
  })
})
