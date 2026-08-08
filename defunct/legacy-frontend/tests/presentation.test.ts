import { describe, expect, it } from 'vitest'
import type { CompareResponse, DayResult, LogisticsLedger, Shock } from '../src/types'
import {
  PRESENTATION_INTERPOLATION_DISCLOSURE,
  SHOCK_IMPACT_WINDOW_FRACTION,
  SHOCK_RESPONSE_START_FRACTION,
  presentationEase,
  presentationIncidentStage,
  sampleRunPresentation,
} from '../src/game/presentation'

const SERVICES = ['transport', 'housing', 'food', 'healthcare', 'public_services'] as const

function vector(value: number): number[] {
  return Array.from({ length: SERVICES.length }, () => value)
}

function logistics(options: Partial<LogisticsLedger> = {}): LogisticsLedger {
  return {
    depot_capacity: vector(100),
    depot_stock_before: vector(10),
    pending_arrivals: vector(4),
    pending_arrivals_landed: vector(4),
    pending_arrivals_held: vector(0),
    depot_stock_after_pending: vector(14),
    depot_damage_penalty: vector(0.1),
    depot_damage_days_remaining: vector(2),
    depot_damage_factor: vector(0.8),
    road_capacity: 0.8,
    throughput_factor: vector(0.64),
    mutual_aid_transfers: [],
    mutual_aid_net: vector(0),
    depot_stock_ready: vector(14),
    pending_next_day: vector(6),
    same_day_delivery_scheduled: vector(10),
    same_day_delivery_landed: vector(8),
    same_day_delivery_held: vector(2),
    delayed_delivery_scheduled: vector(6),
    repair_reserve: vector(2),
    repair_request: vector(7),
    repair_dispatch: vector(6),
    repair_supply: vector(4.8),
    spoilage: vector(0.2),
    depot_stock_end: vector(18),
    capacity_overflow: vector(1),
    conservation_residual: vector(0),
    ...options,
  }
}

function shock(day: number, type: Shock['type'] = null): Shock {
  return {
    day,
    type,
    severity: type ? 0.25 : 0,
    impact: type ? [0.65, 1, 0.2, 0.35, 0.45] : vector(0),
    budget_factor: type ? 0.15 : 0,
    forced: Boolean(type),
  }
}

function day(
  dayNumber: number,
  servicesBefore: number[],
  servicesAfterShock: number[],
  servicesEnd: number[],
  availableBudget: number,
  dayShock: Shock = shock(dayNumber),
  dayLogistics?: LogisticsLedger,
): DayResult {
  const allocation = [30, 40, 35, 45, availableBudget - 150]
  return {
    day: dayNumber,
    shock: dayShock,
    available_budget: availableBudget,
    services_before: servicesBefore,
    services_after_shock: servicesAfterShock,
    raw_action: vector(0),
    raw_proposal: allocation,
    lower_bounds: vector(0),
    upper_bounds: vector(100),
    allocation,
    projection: {
      distance: 0,
      sum: availableBudget,
      constraint_violations: 0,
      violation_breakdown: {
        sum_violations: 0,
        budget_violations: 0,
        lower_violations: 0,
        upper_violations: 0,
        total: 0,
      },
      bindings: SERVICES.map((service) => ({ service, lower: false, upper: false })),
    },
    planner_evidence: null,
    support: vector(0.8),
    throughput: vector(0.8),
    gain: servicesEnd.map((value, index) => value - servicesAfterShock[index]),
    strain: vector(0),
    services_end: servicesEnd,
    resilience: servicesEnd.reduce((sum, value) => sum + value, 0) / servicesEnd.length,
    reward: 0.5,
    logistics: dayLogistics,
  }
}

function result(trajectory: DayResult[]): CompareResponse {
  return {
    schema_version: trajectory.some((entry) => entry.logistics) ? '3.0.0' : '2.1.0',
    engine_version: 'city-recovery-env-v2',
    result_id: 'a'.repeat(64),
    persistence: { format: 'canonical-json-v1', idempotent: true, result_id: 'a'.repeat(64) },
    seed: 17,
    generator: 'numpy.PCG64',
    scenario: {
      name: 'Interpolation fixture',
      horizon_days: trajectory.length,
      daily_budget: 180,
      initial_services: vector(0.2),
      priorities: [1, 2, 1, 3, 1],
      shock_probability: 0.2,
      severity_min: 0.05,
      severity_max: 0.4,
      forced_shock: null,
      forced_shocks: [],
    },
    services: [...SERVICES],
    shock_schedule: trajectory.map((entry) => entry.shock),
    shock_schedule_sha256: 'b'.repeat(64),
    policy: {
      id: 'test',
      artifact_type: 'ppo',
      algorithm: 'PPO',
      runtime: 'ONNX',
      sha256: 'c'.repeat(64),
      sb3_checkpoint_sha256: 'd'.repeat(64),
      parity_report_sha256: 'e'.repeat(64),
      disclosure: 'authored synthetic',
      legacy_candidate: {
        id: 'legacy', artifact_type: 'linear', is_ppo: false,
        sha256: 'f'.repeat(64), disclosure: 'legacy',
      },
    },
    baseline_spec: {
      id: 'baseline', library: 'OR-Tools', library_version: '1', solver: 'GLOP',
      objective: 'visible', future_shocks_visible: false,
    },
    candidate: {
      planner: 'candidate', rauc: 0.6, final_resilience: 0.6, minimum_resilience: 0.2,
      post_shock_recovery_shortfall_auc: 0.1,
      days_to_pre_shock_recovery_after_largest_loss: 3,
      critical_service_days: 0, total_projection_distance: 0,
      constraint_violations: 0, trajectory,
    },
    baseline: {
      planner: 'baseline', rauc: 0.5, final_resilience: 0.5, minimum_resilience: 0.1,
      post_shock_recovery_shortfall_auc: 0.2,
      days_to_pre_shock_recovery_after_largest_loss: 4,
      critical_service_days: 0, total_projection_distance: 0,
      constraint_violations: 0, trajectory,
    },
    comparison: {
      primary_metric: 'rauc', candidate_minus_baseline: 0.1,
      outcome: 'candidate_higher_rauc',
    },
    limitations: [],
  }
}

describe('pure run presentation sampling', () => {
  const first = day(
    1,
    vector(0.2),
    vector(0.2),
    [0.4, 0.5, 0.6, 0.7, 0.8],
    180,
    shock(1),
    logistics(),
  )
  const quakeLedger = logistics({
    depot_stock_before: vector(18),
    depot_stock_end: vector(30),
    depot_damage_factor: vector(0.5),
    depot_damage_penalty: vector(0.3),
    depot_damage_days_remaining: vector(4),
    throughput_factor: vector(0.35),
    road_capacity: 0.5,
    pending_arrivals_landed: vector(6),
    same_day_delivery_landed: vector(10),
    repair_dispatch: vector(8),
    repair_supply: vector(5),
    pending_next_day: vector(12),
  })
  const second = day(
    2,
    first.services_end,
    [0.2, 0.15, 0.5, 0.45, 0.4],
    [0.3, 0.35, 0.55, 0.6, 0.5],
    150,
    shock(2, 'aftershock'),
    quakeLedger,
  )
  const third = day(
    3,
    second.services_end,
    second.services_end,
    [0.4, 0.5, 0.62, 0.68, 0.58],
    180,
    shock(3),
    logistics({
      depot_stock_before: vector(30),
      depot_stock_end: vector(36),
      depot_damage_factor: vector(0.65),
      depot_damage_penalty: vector(0.2),
      depot_damage_days_remaining: vector(3),
      throughput_factor: vector(0.48),
      road_capacity: 0.65,
    }),
  )
  const run = result([first, second, third])

  it('uses a smooth full-day curve with exact endpoints', () => {
    expect(presentationEase(-1)).toBe(0)
    expect(presentationEase(0)).toBe(0)
    expect(presentationEase(0.25)).toBeCloseTo(0.103515625, 12)
    expect(presentationEase(0.5)).toBe(0.5)
    expect(presentationEase(1)).toBe(1)
    expect(presentationEase(2)).toBe(1)

    const start = sampleRunPresentation(run, { dayIndex: 0, progress: 0 })
    const middle = sampleRunPresentation(run, { dayIndex: 0, progress: 0.5 })
    const end = sampleRunPresentation(run, { dayIndex: 0, progress: 1 })
    expect(start.services).toEqual(first.services_before)
    expect(middle.services[0]).toBeCloseTo(0.3)
    expect(end.services).toEqual(first.services_end)
  })

  it('lands a shock at the boundary with continuous impact follow-through, then recovery', () => {
    const boundary = sampleRunPresentation(run, { dayIndex: 1, progress: 0 })
    const impactMidpoint = sampleRunPresentation(run, {
      dayIndex: 1,
      progress: SHOCK_IMPACT_WINDOW_FRACTION / 2,
    })
    const afterShock = sampleRunPresentation(run, {
      dayIndex: 1,
      progress: SHOCK_IMPACT_WINDOW_FRACTION,
    })
    const assessment = sampleRunPresentation(run, {
      dayIndex: 1,
      progress: (SHOCK_IMPACT_WINDOW_FRACTION + SHOCK_RESPONSE_START_FRACTION) / 2,
    })
    const settling = sampleRunPresentation(run, {
      dayIndex: 1,
      progress: SHOCK_RESPONSE_START_FRACTION
        + (1 - SHOCK_RESPONSE_START_FRACTION) / 2,
    })
    const end = sampleRunPresentation(run, { dayIndex: 1, progress: 1 })

    expect(boundary.shockAtBoundary).toBe(true)
    expect(boundary.incidentSegment).toBe('impact')
    expect(boundary.shockImpactProgress).toBe(0)
    expect(boundary.services).toEqual(second.services_before)
    expect(boundary.services).toEqual(first.services_end)
    expect(impactMidpoint.shockImpactProgress).toBe(0.5)
    expect(impactMidpoint.services[0]).toBeCloseTo(0.3)
    expect(afterShock.incidentSegment).toBe('assessment')
    expect(afterShock.shockImpactProgress).toBe(1)
    expect(afterShock.recoveryProgress).toBe(0)
    expect(afterShock.services).toEqual(second.services_after_shock)
    expect(assessment.incidentSegment).toBe('assessment')
    expect(assessment.recoveryProgress).toBe(0)
    expect(assessment.services).toEqual(second.services_after_shock)
    expect(settling.incidentSegment).toBe('recovery')
    expect(settling.recoveryProgress).toBeCloseTo(0.5)
    expect(settling.services[0]).toBeCloseTo(0.25)
    expect(end.services).toEqual(second.services_end)
    expect(end.services).not.toEqual(third.services_end)
  })

  it('derives impact, assessment, and response from fixed shared-cursor thresholds', () => {
    expect(presentationIncidentStage(0)).toBe('impact')
    expect(presentationIncidentStage(SHOCK_IMPACT_WINDOW_FRACTION - 1e-6)).toBe('impact')
    expect(presentationIncidentStage(SHOCK_IMPACT_WINDOW_FRACTION)).toBe('assessment')
    expect(presentationIncidentStage(SHOCK_RESPONSE_START_FRACTION - 1e-6)).toBe('assessment')
    expect(presentationIncidentStage(SHOCK_RESPONSE_START_FRACTION)).toBe('response')
    expect(presentationIncidentStage(4)).toBe('response')
  })

  it('keeps clear boundaries continuous and never samples a future day', () => {
    const priorEnd = sampleRunPresentation(run, { dayIndex: 1, progress: 1 })
    const clearStart = sampleRunPresentation(run, { dayIndex: 2, progress: 0 })
    expect(clearStart.shockAtBoundary).toBe(false)
    expect(clearStart.services).toEqual(priorEnd.services)
    expect(clearStart.serviceEndpoints.end).toEqual(third.services_end)
    expect(clearStart.logistics?.pendingArrivalsLanded).toEqual(priorEnd.logistics?.pendingArrivalsLanded)
    expect(clearStart.logistics?.sameDayDeliveryLanded).toEqual(priorEnd.logistics?.sameDayDeliveryLanded)
    expect(clearStart.logistics?.pendingNextDay).toEqual(priorEnd.logistics?.pendingNextDay)
    expect(clearStart.logistics?.repairDispatch).toEqual(priorEnd.logistics?.repairDispatch)
    expect(clearStart.logistics?.capacityOverflow).toEqual(priorEnd.logistics?.capacityOverflow)
  })

  it('lands the game HUD arrival change during the shock impact window', () => {
    const start = sampleRunPresentation(run, { dayIndex: 1, progress: 0 })
    const impactMiddle = sampleRunPresentation(run, {
      dayIndex: 1,
      progress: SHOCK_IMPACT_WINDOW_FRACTION / 2,
    })
    const afterImpact = sampleRunPresentation(run, {
      dayIndex: 1,
      progress: SHOCK_IMPACT_WINDOW_FRACTION,
    })
    const end = sampleRunPresentation(run, { dayIndex: 1, progress: 1 })
    expect(start.availableBudgetEndpoints).toEqual({ start: 180, end: 150 })
    expect(start.availableBudget).toBe(180)
    expect(impactMiddle.availableBudget).toBe(165)
    expect(afterImpact.availableBudget).toBe(150)
    expect(end.availableBudget).toBe(150)
  })

  it('samples schema-v3 depot stock and operations only between recorded endpoints', () => {
    const boundary = sampleRunPresentation(run, { dayIndex: 1, progress: 0 })
    const middle = sampleRunPresentation(run, { dayIndex: 1, progress: 0.5 })
    const end = sampleRunPresentation(run, { dayIndex: 1, progress: 1 })

    expect(boundary.logistics?.depotStock).toEqual(vector(18))
    expect(boundary.logistics?.pendingNextDay).toEqual(vector(6))
    expect(middle.logistics?.depotStock).toEqual(vector(24))
    expect(end.logistics?.depotStock).toEqual(vector(30))
    expect(boundary.logistics?.depotDamageFactor).toEqual(vector(0.8))
    expect(middle.logistics?.depotDamageFactor).toEqual(vector(0.5))
    expect(middle.logistics?.roadCapacity).toBe(0.5)
    expect(middle.logistics?.pendingArrivalsLanded).toEqual(vector(5))
    expect(middle.logistics?.sameDayDeliveryLanded).toEqual(vector(9))
    expect(middle.logistics?.landedUnits).toEqual(vector(14))
    expect(middle.logistics?.repairDispatch).toEqual(vector(7))
    expect(middle.logistics?.repairSupply).toEqual(vector(4.9))
    expect(middle.logistics?.pendingNextDay).toEqual(vector(9))
    expect(end.visualDay.logistics?.depot_stock_end).toEqual(quakeLedger.depot_stock_end)
    expect(end.visualDay.logistics?.repair_supply).toEqual(quakeLedger.repair_supply)
  })

  it('smooths adjacent clear-day depot function while retaining a discrete shock hit', () => {
    const shockImpactMidpoint = sampleRunPresentation(run, {
      dayIndex: 1,
      progress: SHOCK_IMPACT_WINDOW_FRACTION / 2,
    })
    const shockImpactEnd = sampleRunPresentation(run, {
      dayIndex: 1,
      progress: SHOCK_IMPACT_WINDOW_FRACTION,
    })
    expect(shockImpactMidpoint.logistics?.depotDamageFactor).toEqual(vector(0.65))
    expect(shockImpactMidpoint.logistics?.roadCapacity).toBeCloseTo(0.65)
    expect(shockImpactEnd.logistics?.depotDamageFactor).toEqual(vector(0.5))
    expect(shockImpactEnd.logistics?.roadCapacity).toBeCloseTo(0.5)

    const clearStart = sampleRunPresentation(run, { dayIndex: 2, progress: 0 })
    const clearMiddle = sampleRunPresentation(run, { dayIndex: 2, progress: 0.5 })
    const clearEnd = sampleRunPresentation(run, { dayIndex: 2, progress: 1 })
    expect(clearStart.logistics?.depotDamageFactor).toEqual(vector(0.5))
    expect(clearMiddle.logistics?.depotDamageFactor).toEqual(vector(0.575))
    expect(clearEnd.logistics?.depotDamageFactor).toEqual(vector(0.65))
    expect(clearMiddle.logistics?.roadCapacity).toBeCloseTo(0.575)
  })

  it('derives weighted wellbeing from the sampled services and scenario priorities', () => {
    const sample = sampleRunPresentation(run, { dayIndex: 1, progress: 0 })
    const priorities = run.scenario.priorities
    const expected = sample.services.reduce(
      (sum, value, index) => sum + value * priorities[index],
      0,
    ) / priorities.reduce((sum, value) => sum + value, 0)
    expect(sample.wellbeing).toBeCloseTo(expected, 12)
    expect(sample.visualDay.resilience).toBe(sample.wellbeing)
  })

  it('is deterministic, deep-equal, and does not mutate or reuse visual arrays from the run', () => {
    const before = structuredClone(run)
    const left = sampleRunPresentation(run, { dayIndex: 1, progress: 0.371 })
    const right = sampleRunPresentation(run, { dayIndex: 1, progress: 0.371 })
    expect(left).toEqual(right)
    expect(run).toEqual(before)
    expect(left.recordedDay).toBe(run.candidate.trajectory[1])
    expect(left.visualDay).not.toBe(left.recordedDay)
    expect(left.services).not.toBe(left.recordedDay.services_end)
    expect(left.visualDay.logistics).not.toBe(left.recordedDay.logistics)
    expect(PRESENTATION_INTERPOLATION_DISCLOSURE).toContain('returned daily states')
    expect(PRESENTATION_INTERPOLATION_DISCLOSURE).toContain('not additional simulator steps')
  })

  it('normalizes an out-of-range cursor and rejects an empty trajectory', () => {
    expect(sampleRunPresentation(run, { dayIndex: 99, progress: 2 }).cursor).toEqual({
      dayIndex: 2,
      progress: 1,
    })
    expect(sampleRunPresentation(run, { dayIndex: Number.NaN, progress: Number.NaN }).cursor).toEqual({
      dayIndex: 0,
      progress: 0,
    })
    expect(() => sampleRunPresentation(result([]), { dayIndex: 0, progress: 0 })).toThrow(RangeError)
  })
})
