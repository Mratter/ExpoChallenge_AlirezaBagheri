import { describe, expect, it } from 'vitest'
import type { CompareResponse, DayResult, Shock, ShockType } from '../src/types'
import { infrastructureRepairVisibility, roadNetworkState } from '../src/game/InfrastructureScene'
import {
  depotPresentationCopy,
  intakeHubPresentationCopy,
  IN_WORLD_INTERPOLATION_DISCLOSURE,
  palletFieldPlan,
} from '../src/game/DepotNetwork'
import {
  afterActionReports,
  buildingRepairPresentationOffset,
  buildingRepairStarted,
  depotDamageStateFromFactor,
  depotStatusesForDay,
  FUEL_POINT_STOP_COPY,
  incidentPhaseForDay,
  intensityBand,
  LEGACY_V1_DEPOT_DISCLOSURE,
  recoveryArcForService,
  recoveryMilestones,
  repairFreshnessForBuilding,
  repairFreshnessForBuildingAt,
  repairProgressForBuilding,
  repairProgressForBuildingAt,
  repairStageForBuildingAt,
  scheduledFuelServiceForDay,
  shockAdjustedArrivalShortfall,
  splitExactCargo,
  throughputVehiclesPerDay,
  vehicleDispatchCountsForDay,
  vehicleManifestsForDay,
} from '../src/game/realism'
import { REALISM_LEDGER } from '../src/game/realismLedger'
import {
  advanceVehicleMissionProgress,
  dispatchStartProgress,
  buildingCurbFor,
  MAX_VISIBLE_ROAD_VEHICLES,
  publishVehicleEvidenceTransition,
  VISIBLE_VEHICLE_LIMITS,
  VEHICLE_MISSION_STAGE_PROGRESS,
  vehicleAdvancesInMode,
  vehicleCargoCopy,
  vehicleCarriesCargo,
  vehicleCycleState,
  vehicleMissionDayKey,
  vehicleMissionProgressAt,
  vehicleMissionProgressPerDay,
  vehicleMissionSnapshotAt,
  vehicleMissionTimelinesForResult,
  vehicleMissionWeatherMultiplier,
  vehicleOperationalAbsoluteDay,
  vehiclePlansForDay,
  vehiclePlansForMode,
  vehiclePositionForCycle,
  visibleVehiclePlansForDay,
  visibleVehicleRole,
  visibleVehicleStableIdsForResult,
  type VehicleEvidenceTransition,
} from '../src/game/VehicleFleet'
import { damageStateFor, DISTRICTS, rebuildingCohortForDay, relayNarration } from '../src/game/model'
import {
  CRITICAL_SMOKE_PUFF_COUNT,
  criticalSmokePresentation,
  stagedRepairDamageState,
} from '../src/game/CityScene'
import {
  latestPresentedIncident,
  latestPresentedIncidentOfType,
  presentedIncidentRecovery,
} from '../src/game/RecoveryPhenomenology'
import { CITY_BUILDING_PLACEMENTS, CITY_DEPOTS, CITY_HUB, isPointOnCityRoad } from '../src/game/worldLayout'

const SERVICES = ['transport', 'housing', 'food', 'healthcare', 'public_services'] as const
const NO_SHOCK: Shock = { day: 1, type: null, severity: 0, impact: [0, 0, 0, 0, 0], budget_factor: 0, forced: false }

describe('phase-gated infrastructure repair presentation', () => {
  it('keeps damage state visible while withholding repair overlays before response', () => {
    const state = { debrisSegments: 4, freshPatches: 3 }
    expect(infrastructureRepairVisibility(state, false)).toEqual({ debrisSegments: 0, freshPatches: 0 })
    expect(infrastructureRepairVisibility(state, true)).toEqual(state)
  })
})

function day(dayNumber: number, options: Partial<DayResult> = {}): DayResult {
  const servicesEnd = options.services_end ?? [0.62, 0.58, 0.64, 0.61, 0.59]
  const allocation = options.allocation ?? [30, 42, 36, 40, 32]
  const shock = options.shock ?? { ...NO_SHOCK, day: dayNumber }
  return {
    day: dayNumber,
    shock,
    available_budget: options.available_budget ?? allocation.reduce((sum, value) => sum + value, 0),
    services_before: options.services_before ?? servicesEnd,
    services_after_shock: options.services_after_shock ?? servicesEnd,
    raw_action: [0, 0, 0, 0, 0],
    raw_proposal: allocation,
    lower_bounds: [0, 0, 0, 0, 0],
    upper_bounds: [90, 90, 90, 90, 90],
    allocation,
    projection: {
      distance: 0,
      sum: options.available_budget ?? allocation.reduce((sum, value) => sum + value, 0),
      constraint_violations: 0,
      violation_breakdown: { sum_violations: 0, budget_violations: 0, lower_violations: 0, upper_violations: 0, total: 0 },
      bindings: SERVICES.map((service) => ({ service, lower: false, upper: false })),
    },
    planner_evidence: null,
    support: [0.8, 0.8, 0.8, 0.8, 0.8],
    gain: options.gain ?? [0.01, 0.01, 0.01, 0.01, 0.01],
    strain: [0, 0, 0, 0, 0],
    services_end: servicesEnd,
    resilience: options.resilience ?? 0.61,
    reward: 0.61,
    logistics: options.logistics,
  }
}

function result(trajectory: DayResult[]): CompareResponse {
  return {
    schema_version: trajectory.some((entry) => entry.logistics) ? '3.0.0' : '2.1.0',
    result_id: 'a'.repeat(64),
    persistence: { format: 'canonical-json-v1', idempotent: true, result_id: 'a'.repeat(64) },
    seed: 424242,
    generator: 'numpy.PCG64',
    scenario: {
      name: 'Relay City test',
      horizon_days: trajectory.length,
      daily_budget: 180,
      initial_services: [0.62, 0.58, 0.64, 0.61, 0.59],
      priorities: [1, 1.1, 1.2, 1.4, 1],
      shock_probability: 0.2,
      severity_min: 0.1,
      severity_max: 0.28,
      forced_shock: null,
      forced_shocks: [],
    },
    services: [...SERVICES],
    shock_schedule: trajectory.map((entry) => entry.shock),
    shock_schedule_sha256: 'b'.repeat(64),
    policy: {
      id: 'test', artifact_type: 'ppo', algorithm: 'PPO', runtime: 'ONNX', sha256: 'c'.repeat(64),
      sb3_checkpoint_sha256: 'd'.repeat(64), parity_report_sha256: 'e'.repeat(64), disclosure: 'synthetic',
      legacy_candidate: { id: 'legacy', artifact_type: 'linear', is_ppo: false, sha256: 'f'.repeat(64), disclosure: 'legacy' },
    },
    baseline_spec: { id: 'baseline', library: 'OR-Tools', library_version: '1', solver: 'GLOP', objective: 'visible', future_shocks_visible: false },
    candidate: {
      planner: 'candidate', rauc: 0.6, final_resilience: 0.6, minimum_resilience: 0.4,
      post_shock_recovery_shortfall_auc: 0.1, days_to_pre_shock_recovery_after_largest_loss: 3,
      critical_service_days: 0, total_projection_distance: 0, constraint_violations: 0, trajectory,
    },
    baseline: {
      planner: 'baseline', rauc: 0.5, final_resilience: 0.5, minimum_resilience: 0.3,
      post_shock_recovery_shortfall_auc: 0.2, days_to_pre_shock_recovery_after_largest_loss: 4,
      critical_service_days: 0, total_projection_distance: 0, constraint_violations: 0, trajectory,
    },
    comparison: { primary_metric: 'rauc', candidate_minus_baseline: 0.1, outcome: 'candidate_higher_rauc' },
    limitations: [],
  }
}

describe('realism logistics selectors', () => {
  it('renders every accepted supply unit without the former 400-pallet cap', () => {
    expect(palletFieldPlan(0)).toEqual({ instanceCount: 0, finalPalletFraction: 0 })
    expect(palletFieldPlan(400)).toEqual({ instanceCount: 400, finalPalletFraction: 1 })
    expect(palletFieldPlan(500)).toEqual({ instanceCount: 500, finalPalletFraction: 1 })

    const fractional = palletFieldPlan(499.25)
    expect(fractional).toEqual({ instanceCount: 500, finalPalletFraction: 0.25 })
    expect((fractional.instanceCount - 1) + fractional.finalPalletFraction).toBe(499.25)
  })

  it('runs each returned delivery mission once, stages it, and does not restart on a same-day compare rerun', () => {
    let progress = 0.19
    for (let frame = 0; frame < 1_000; frame += 1) {
      progress = advanceVehicleMissionProgress(progress, 1 / 60)
    }
    expect(progress).toBe(VEHICLE_MISSION_STAGE_PROGRESS)
    expect(vehicleCycleState(progress)).toBe('stage')
    expect(advanceVehicleMissionProgress(progress, 10)).toBe(VEHICLE_MISSION_STAGE_PROGRESS)

    const original = result([day(1), day(2)])
    const rerun = {
      ...original,
      result_id: 'f'.repeat(64),
      scenario: {
        ...original.scenario,
        forced_shocks: [{ day: 2, type: 'weather' as const, severity: 0.2 }],
      },
    }
    expect(vehicleMissionDayKey(rerun, 0)).toBe(vehicleMissionDayKey(original, 0))
    expect(vehicleMissionDayKey(original, 1)).not.toBe(vehicleMissionDayKey(original, 0))
  })

  it('shows manifest cargo only through unload and keeps zero-cargo support missions honest', () => {
    const current = day(1, { allocation: [28.5, 38.5, 31, 44, 38] })
    const plans = vehiclePlansForDay(result([current]), 0)
    const flatbed = plans.find((plan) => plan.manifest.fleet === 'brick flatbed' && plan.active)!
    const maintenance = plans.find((plan) => plan.manifest.wave === 'maintenance')!
    const relief = plans.find((plan) => plan.relief)!

    expect(['load', 'outbound', 'dock'].every((state) => (
      vehicleCarriesCargo(flatbed.manifest, state as 'load' | 'outbound' | 'dock')
    ))).toBe(true)
    expect(vehicleCarriesCargo(flatbed.manifest, 'return')).toBe(false)
    expect(vehicleCarriesCargo(flatbed.manifest, 'stage')).toBe(false)
    expect(vehicleCargoCopy(flatbed.manifest, 'return')).toContain('Empty return')
    expect(vehicleCargoCopy(flatbed.manifest, 'return')).toContain('delivered at destination')

    for (const support of [maintenance, relief]) {
      expect(support.manifest.cargoUnits).toBe(0)
      expect(vehicleCarriesCargo(support.manifest, 'outbound')).toBe(false)
      expect(vehicleCargoCopy(support.manifest, 'outbound')).toContain('no supply cargo')
      expect(vehicleCargoCopy(support.manifest, 'outbound')).not.toContain('pallets =')
    }
    expect(vehicleCargoCopy(maintenance.manifest, 'outbound')).toContain('Maintenance mission')
    expect(vehicleCargoCopy(relief.manifest, 'outbound')).toContain('Roadside assistance mission')
    expect(relief.manifest.destination).toContain('roadside breakdown')
    expect(relief.manifest.returnLeg).toContain('roadside assistance')
  })

  it('maps the full engine-v2 depot-function range across four visible damage states', () => {
    expect(depotDamageStateFromFactor(1)).toBe('intact')
    expect(depotDamageStateFromFactor(0.85)).toBe('intact')
    expect(depotDamageStateFromFactor(0.84)).toBe('slight')
    expect(depotDamageStateFromFactor(0.68)).toBe('slight')
    expect(depotDamageStateFromFactor(0.67)).toBe('moderate')
    expect(depotDamageStateFromFactor(0.48)).toBe('moderate')
    expect(depotDamageStateFromFactor(0.47)).toBe('rubble')
    expect(depotDamageStateFromFactor(0.30)).toBe('rubble')
  })

  it('partitions exact fractional cargo without inventing or losing a unit', () => {
    const loads = splitExactCargo(38.5, 12)
    expect(loads).toEqual([12, 12, 12, 2.5])
    expect(loads.reduce((sum, value) => sum + value, 0)).toBe(38.5)
  })

  it('stages each new day with the largest allocation wave visibly rolling first', () => {
    const current = day(4, { allocation: [28, 52, 31, 40, 29] })
    const plans = vehiclePlansForDay(result([current]), 0)
      .filter((plan) => plan.manifest.wave === 'line-haul' && plan.active)
    const first = plans.find((plan) => plan.manifest.dispatchRank === 0)!
    const second = plans.find((plan) => plan.manifest.dispatchRank === 1)!
    const firstProgress = dispatchStartProgress(first, 424242, 4)
    const secondProgress = dispatchStartProgress(second, 424242, 4)
    expect(firstProgress).toBeGreaterThan(secondProgress)
    expect(vehicleCycleState(firstProgress, first.fuelStop !== null)).toBe('outbound')
    expect(['load', 'fuel']).toContain(vehicleCycleState(secondProgress, second.fuelStop !== null))
  })

  it('keeps legacy allocation presentation while leaving unrecorded depot operations unavailable', () => {
    const current = day(1, { allocation: [28.5, 38.5, 31, 44, 38], services_end: [0.7, 0.18, 0.6, 0.75, 0.55] })
    const depots = depotStatusesForDay(current, SERVICES)
    expect(depots).toHaveLength(5)
    expect(Object.fromEntries(depots.map((depot) => [depot.service, depot.allocationUnits]))).toEqual(
      Object.fromEntries(SERVICES.map((service, index) => [service, current.allocation[index]])),
    )
    expect(depots.every((depot) => (
      depot.damage === null
      && depot.placard === null
      && depot.palletUnits === null
      && depot.stockCapacity === null
      && depot.pendingUnits === null
      && depot.spoilageUnits === null
      && depot.throughputSignal === null
      && depot.dockQueue === null
      && depot.reroutedFrom === null
      && depot.inboundWindow.includes('unavailable')
    ))).toBe(true)
  })

  it('uses identical exact cargo mapping for the Toolbox manifest', () => {
    const current = day(2, { allocation: [28.5, 38.5, 31, 44, 38] })
    const manifests = vehicleManifestsForDay(current, undefined, SERVICES)
    const inbound = manifests.filter((manifest) => manifest.wave === 'inbound')
    expect(inbound.reduce((sum, item) => sum + item.cargoUnits, 0)).toBe(current.available_budget)
    for (const service of SERVICES) {
      const serviceLoads = manifests.filter((manifest) => manifest.service === service && manifest.wave === 'line-haul')
      expect(serviceLoads.reduce((sum, item) => sum + item.cargoUnits, 0)).toBe(current.allocation[SERVICES.indexOf(service)])
    }
  })

  it('keeps a stable-capacity fleet, typed jobs, full cycles, and staged inactive vehicles', () => {
    const current = day(1, { services_end: [0.3, 0.58, 0.64, 0.61, 0.59] })
    const plans = vehiclePlansForDay(result([current]), 0)
    expect(plans.some((plan) => plan.manifest.fleet === 'ambulance')).toBe(true)
    expect(plans.some((plan) => plan.manifest.fleet === 'brick flatbed')).toBe(true)
    expect(plans.some((plan) => !plan.active)).toBe(true)
    expect(['load', 'outbound', 'dock', 'return', 'stage'].map((_, index) => vehicleCycleState([0.02, 0.2, 0.5, 0.7, 0.95][index]))).toEqual(['load', 'outbound', 'dock', 'return', 'stage'])
    expect(vehiclePlansForMode(plans, 'full')).toEqual(plans)
    expect(vehiclePlansForMode(plans, 'assessment')).toEqual(plans)
    expect(plans.filter((plan) => plan.manifest.wave !== 'emergency').every((plan) => !vehicleAdvancesInMode(plan, 'assessment'))).toBe(true)
    expect(plans.filter((plan) => plan.manifest.wave === 'emergency').every((plan) => vehicleAdvancesInMode(plan, 'assessment'))).toBe(true)
  })

  it('caps the road view at 17 deterministic slots while leaving the full manifest untouched', () => {
    const currentResult = result(Array.from({ length: 20 }, (_, index) => day(index + 1)))
    const stableIds = visibleVehicleStableIdsForResult(currentResult)
    const repeatedIds = visibleVehicleStableIdsForResult({ ...currentResult, result_id: 'f'.repeat(64) })
    const visible = visibleVehiclePlansForDay(currentResult, 0, stableIds)
    const complete = vehiclePlansForDay(currentResult, 0)
    const roleCounts = visible.reduce<Partial<Record<ReturnType<typeof visibleVehicleRole>, number>>>((counts, plan) => {
      const role = visibleVehicleRole(plan)
      counts[role] = (counts[role] ?? 0) + 1
      return counts
    }, {})

    expect(MAX_VISIBLE_ROAD_VEHICLES).toBe(17)
    expect(stableIds.length).toBeLessThanOrEqual(MAX_VISIBLE_ROAD_VEHICLES)
    expect(new Set(stableIds).size).toBe(stableIds.length)
    expect(repeatedIds).toEqual(stableIds)
    expect(vehicleMissionTimelinesForResult(currentResult)).toHaveLength(stableIds.length)
    Object.entries(VISIBLE_VEHICLE_LIMITS).forEach(([role, limit]) => {
      expect(roleCounts[role as keyof typeof roleCounts] ?? 0).toBeLessThanOrEqual(limit)
    })
    visible.forEach((plan) => {
      const exact = complete.find((candidate) => candidate.stableId === plan.stableId)
      expect(exact).toBeDefined()
      expect(plan.manifest).toEqual(exact?.manifest)
    })

    const completeManifest = vehicleManifestsForDay(currentResult.candidate.trajectory[0], undefined, SERVICES)
    expect(completeManifest.filter((manifest) => manifest.wave === 'inbound')
      .reduce((total, manifest) => total + manifest.cargoUnits, 0)).toBe(180)
  })

  it('selects bounded cargo slots proportionally without changing a selected load', () => {
    const balanced = result([day(1, { allocation: [36, 36, 36, 36, 36] })])
    const skewed = result([day(1, { allocation: [1, 1, 1, 1, 176] })])
    const selectedLineCount = (current: CompareResponse, service: typeof SERVICES[number]) => (
      visibleVehiclePlansForDay(current, 0)
        .filter((plan) => plan.manifest.wave === 'line-haul' && plan.manifest.service === service)
        .length
    )

    expect(selectedLineCount(skewed, 'public_services'))
      .toBeGreaterThan(selectedLineCount(balanced, 'public_services'))
    visibleVehiclePlansForDay(skewed, 0).forEach((plan) => {
      if (plan.manifest.wave === 'line-haul') expect(plan.manifest.cargoUnits).toBeLessThanOrEqual(12)
      if (plan.manifest.wave === 'last-mile') expect(plan.manifest.cargoUnits).toBeLessThanOrEqual(4)
    })
  })

  it('keeps every selected route on the rendered road graph and docks at a visible curb', () => {
    const currentResult = result([day(1), day(2), day(3)])
    const visible = visibleVehiclePlansForDay(currentResult, 0)
    expect(visible.length).toBeGreaterThan(0)
    visible.forEach((plan) => {
      expect(plan.path[0]).toEqual(plan.stagePosition)
      plan.path.forEach((point) => expect(isPointOnCityRoad(point, 0.08)).toBe(true))
      plan.path.slice(1).forEach((point, index) => {
        const previous = plan.path[index]
        expect(Math.abs(point[0] - previous[0]) < 1e-6 || Math.abs(point[2] - previous[2]) < 1e-6)
          .toBe(true)
      })
      expect(vehiclePositionForCycle(plan, 0.90)).toEqual(plan.stagePosition)
    })

    const lastMile = visible.find((plan) => plan.manifest.wave === 'last-mile')!
    const endpoint = lastMile.path.at(-1)!
    expect(vehiclePositionForCycle(lastMile, 0.45)).toEqual(endpoint)
    expect(vehiclePositionForCycle(lastMile, 0.52)).toEqual(endpoint)
    expect(vehiclePositionForCycle(lastMile, 0.589999)).toEqual(endpoint)
    const originDepot = CITY_DEPOTS.find((depot) => lastMile.manifest.origin.includes(
      DISTRICTS.find((district) => district.service === depot.service)?.shortLabel ?? '',
    ))
    expect(originDepot).toBeDefined()
    expect(Math.hypot(
      lastMile.stagePosition[0] - originDepot!.curb[0],
      lastMile.stagePosition[2] - originDepot!.curb[2],
    )).toBeLessThan(1.5)

    const detoured = vehiclePlansForDay(result([day(1, {
      services_end: [0.3, 0.58, 0.64, 0.61, 0.59],
    })]), 0).filter((plan) => plan.manifest.wave === 'line-haul')
    detoured.forEach((plan) => plan.path.forEach((point) => {
      expect(isPointOnCityRoad(point, 0.08)).toBe(true)
    }))
  })

  it('uses distance-based table speed so a route three times longer takes three times longer', () => {
    const short = { speed: 0.7, pathLength: 10, fuelStop: null }
    const long = { speed: 0.7, pathLength: 30, fuelStop: null }
    const shortRate = vehicleMissionProgressPerDay(short)
    const longRate = vehicleMissionProgressPerDay(long)
    const shortWorldUnitsPerDay = short.pathLength * shortRate / 0.32
    const longWorldUnitsPerDay = long.pathLength * longRate / 0.32

    expect(shortWorldUnitsPerDay).toBeCloseTo(4.9, 8)
    expect(longWorldUnitsPerDay).toBeCloseTo(shortWorldUnitsPerDay, 8)
    expect(shortRate / longRate).toBeCloseTo(3, 8)
    expect(0.45 / longRate).toBeCloseTo((0.45 / shortRate) * 3, 8)
  })

  it('samples each convoy mission from deterministic presentation time without looping', () => {
    const currentResult = result([day(1), day(2)])
    const plan = vehiclePlansForDay(currentResult, 0).find((entry) => entry.active)!
    const start = vehicleMissionProgressAt(plan, currentResult.seed, 1, 0, 0)
    const mid = vehicleMissionProgressAt(plan, currentResult.seed, 1, 0, 0.5)
    const repeated = vehicleMissionProgressAt(plan, currentResult.seed, 1, 0, 0.5)
    const staged = vehicleMissionProgressAt(plan, currentResult.seed, 1, 0, 20)
    expect(mid).toBeGreaterThan(start)
    expect(repeated).toBe(mid)
    expect(staged).toBe(VEHICLE_MISSION_STAGE_PROGRESS)
    expect(vehicleCycleState(staged, plan.fuelStop !== null)).toBe('stage')
  })

  it('reconstructs identical fleet positions after backward seeks and late mounts', () => {
    const currentResult = result([day(1), day(2), day(3)])
    const firstTimeline = vehicleMissionTimelinesForResult(currentResult)
      .find((entry) => entry.stableId === 'housing-line-0')!
    const forward = vehicleMissionSnapshotAt(firstTimeline, 1.42)
    const rewound = vehicleMissionSnapshotAt(firstTimeline, 0.31)
    const replayed = vehicleMissionSnapshotAt(firstTimeline, 1.42)
    const remountedTimeline = vehicleMissionTimelinesForResult(currentResult)
      .find((entry) => entry.stableId === 'housing-line-0')!
    const remounted = vehicleMissionSnapshotAt(remountedTimeline, 1.42)

    expect(rewound.progress).toBeLessThan(forward.progress)
    expect(replayed.progress).toBe(forward.progress)
    expect(replayed.missionKey).toBe(forward.missionKey)
    expect(remounted.progress).toBe(forward.progress)
    expect(remounted.missionKey).toBe(forward.missionKey)
    expect(vehiclePositionForCycle(remounted.plan, remounted.progress))
      .toEqual(vehiclePositionForCycle(forward.plan, forward.progress))
  })

  it('starts each queued mission from its existing staging position without a boundary jump', () => {
    const currentResult = result(Array.from({ length: 30 }, (_, index) => day(index + 1)))
    const timeline = vehicleMissionTimelinesForResult(currentResult)
      .find((entry) => entry.stableId === 'housing-line-0')!
    const nextMission = timeline.missions[1]
    const boundary = nextMission.startOperationalDay
    const epsilon = 1e-7
    const before = vehicleMissionSnapshotAt(timeline, boundary - epsilon)
    const atBoundary = vehicleMissionSnapshotAt(timeline, boundary)
    const after = vehicleMissionSnapshotAt(timeline, boundary + epsilon)
    const beforePosition = vehiclePositionForCycle(before.plan, before.progress)
    const boundaryPosition = vehiclePositionForCycle(atBoundary.plan, atBoundary.progress)
    const afterPosition = vehiclePositionForCycle(after.plan, after.progress)
    const distance = (left: readonly number[], right: readonly number[]) => Math.hypot(
      left[0] - right[0],
      left[1] - right[1],
      left[2] - right[2],
    )

    expect(atBoundary.missionKey).toBe(nextMission.missionKey)
    expect(nextMission.initialProgress).toBe(0)
    expect(boundaryPosition).toEqual(atBoundary.plan.stagePosition)
    expect(distance(beforePosition, boundaryPosition)).toBeLessThan(0.0001)
    expect(distance(boundaryPosition, afterPosition)).toBeLessThan(0.0001)

    // A seek away and a fresh timeline mount reconstruct the same boundary pose.
    vehicleMissionSnapshotAt(timeline, Math.min(timeline.dayCount, boundary + 0.4))
    const replayed = vehicleMissionSnapshotAt(timeline, boundary)
    const remountedTimeline = vehicleMissionTimelinesForResult(currentResult)
      .find((entry) => entry.stableId === 'housing-line-0')!
    const remounted = vehicleMissionSnapshotAt(remountedTimeline, boundary)
    expect(replayed).toEqual(atBoundary)
    expect(remounted).toEqual(atBoundary)
    expect(vehiclePositionForCycle(remounted.plan, remounted.progress)).toEqual(boundaryPosition)
  })

  it('joins the return route to staging continuously at the stage boundary', () => {
    const currentResult = result([day(1)])
    const activePlans = vehiclePlansForDay(currentResult, 0).filter((entry) => entry.active)
    const representative = activePlans.find((entry) => entry.fuelStop === null)!
    const plans = [...activePlans, { ...representative, broken: true }]
    const epsilon = 1e-7
    expect(activePlans.length).toBeGreaterThan(0)
    for (const plan of plans) {
      const justReturning = vehiclePositionForCycle(plan, 0.90 - epsilon)
      const staged = vehiclePositionForCycle(plan, 0.90)
      const stagedLater = vehiclePositionForCycle(plan, VEHICLE_MISSION_STAGE_PROGRESS)
      const boundaryDistance = Math.hypot(
        justReturning[0] - staged[0],
        justReturning[1] - staged[1],
        justReturning[2] - staged[2],
      )

      expect(vehicleCycleState(0.90, plan.fuelStop !== null)).toBe('stage')
      expect(staged).toEqual(plan.stagePosition)
      expect(stagedLater).toEqual(plan.stagePosition)
      expect(boundaryDistance).toBeLessThan(0.0001)
    }
  })

  it('keeps one returned dispatch per day across same-day compare reruns', () => {
    const original = result([day(1), day(2), day(3)])
    const rerun: CompareResponse = {
      ...original,
      result_id: 'f'.repeat(64),
      scenario: {
        ...original.scenario,
        forced_shocks: [{ day: 3, type: 'utility', severity: 0.4 }],
      },
    }
    const originalTimeline = vehicleMissionTimelinesForResult(original)
      .find((entry) => entry.stableId === 'housing-line-0')!
    const rerunTimeline = vehicleMissionTimelinesForResult(rerun)
      .find((entry) => entry.stableId === 'housing-line-0')!
    const cursor = 1.37
    const before = vehicleMissionSnapshotAt(originalTimeline, cursor)
    const after = vehicleMissionSnapshotAt(rerunTimeline, cursor)

    expect(originalTimeline.missions).toHaveLength(3)
    expect(rerunTimeline.missions).toHaveLength(3)
    expect(rerunTimeline.missions.map((mission) => mission.missionKey))
      .toEqual(originalTimeline.missions.map((mission) => mission.missionKey))
    expect(after.progress).toBe(before.progress)
    expect(after.missionKey).toBe(before.missionKey)
  })

  it('freezes weather speed to the mission day instead of the current display day', () => {
    const weatherShock: Shock = {
      day: 2,
      type: 'weather',
      severity: 0.6,
      impact: [0.2, 0.1, 0.1, 0.1, 0.1],
      budget_factor: 0,
      forced: true,
    }
    const laterWeather = result([day(1), day(2, { shock: weatherShock }), day(3)])
    const alwaysClear = result([day(1), day(2), day(3)])
    const laterWeatherTimeline = vehicleMissionTimelinesForResult(laterWeather)
      .find((entry) => entry.stableId === 'housing-line-0')!
    const alwaysClearTimeline = vehicleMissionTimelinesForResult(alwaysClear)
      .find((entry) => entry.stableId === 'housing-line-0')!

    expect(vehicleMissionWeatherMultiplier(laterWeather, 0)).toBe(1)
    expect(vehicleMissionWeatherMultiplier(laterWeather, 1)).toBe(0.58)
    expect(vehicleMissionSnapshotAt(laterWeatherTimeline, 0.8).progress)
      .toBe(vehicleMissionSnapshotAt(alwaysClearTimeline, 0.8).progress)
    expect(vehicleMissionSnapshotAt(laterWeatherTimeline, 1.25).progress)
      .toBe(vehicleMissionSnapshotAt(laterWeatherTimeline, 1).progress)
    expect(vehicleMissionSnapshotAt(laterWeatherTimeline, 1.61).progress)
      .toBe(vehicleMissionSnapshotAt(alwaysClearTimeline, 1.25).progress)

    const firstDayWeather = result([
      day(1, { shock: { ...weatherShock, day: 1 } }),
      day(2),
      day(3),
    ])
    const firstDayWeatherTimeline = vehicleMissionTimelinesForResult(firstDayWeather)
      .find((entry) => entry.stableId === 'housing-line-0')!
    const beforeBoundary = vehicleMissionSnapshotAt(firstDayWeatherTimeline, 0.8)
    const afterBoundary = vehicleMissionSnapshotAt(firstDayWeatherTimeline, 1.2)
    const firstMission = firstDayWeatherTimeline.missions[0]
    expect(afterBoundary.missionKey).toBe(beforeBoundary.missionKey)
    expect(afterBoundary.progress - beforeBoundary.progress)
      .toBeCloseTo(firstMission.progressPerDay * 0.4, 10)
  })

  it('holds ordinary traffic continuously through impact and assessment while emergency traffic advances', () => {
    const earthquake: Shock = {
      day: 2,
      type: 'aftershock',
      severity: 0.65,
      impact: [0.22, 0.18, 0.12, 0.2, 0.16],
      budget_factor: 0,
      forced: true,
    }
    const shockedResult = result([day(1), day(2, { shock: earthquake }), day(3)])
    const timelines = vehicleMissionTimelinesForResult(shockedResult)
    const ordinary = timelines.find((timeline) => (
      timeline.stableId === 'housing-line-0'
    ))!
    const emergency = timelines.find((timeline) => (
      [...timeline.dailyPlans.values()].some((plan) => plan.manifest.wave === 'emergency')
    ))!

    expect(vehicleOperationalAbsoluteDay(1, [1])).toBe(1)
    expect(vehicleOperationalAbsoluteDay(1.18, [1])).toBe(1)
    expect(vehicleOperationalAbsoluteDay(1.359, [1])).toBe(1)
    expect(vehicleOperationalAbsoluteDay(1.61, [1])).toBeCloseTo(1.25, 10)

    const ordinaryAtImpact = vehicleMissionSnapshotAt(ordinary, 1)
    const ordinaryAtAssessment = vehicleMissionSnapshotAt(ordinary, 1.18)
    const ordinaryBeforeResponse = vehicleMissionSnapshotAt(ordinary, 1.359)
    const ordinaryAfterResponse = vehicleMissionSnapshotAt(ordinary, 1.61)
    expect(ordinaryAtAssessment.progress).toBe(ordinaryAtImpact.progress)
    expect(ordinaryBeforeResponse.progress).toBe(ordinaryAtImpact.progress)
    expect(ordinaryAfterResponse.progress).toBeGreaterThan(ordinaryAtImpact.progress)

    const emergencyAtImpact = vehicleMissionSnapshotAt(emergency, 1)
    const emergencyAtAssessment = vehicleMissionSnapshotAt(emergency, 1.18)
    expect(emergency.pausesForAssessment).toBe(false)
    expect(emergencyAtAssessment.progress).toBeGreaterThan(emergencyAtImpact.progress)
    expect(vehicleMissionSnapshotAt(ordinary, 1.61)).toEqual(ordinaryAfterResponse)
  })

  it('keeps the verification-only vehicle transition hook read-only and explicitly selected', () => {
    const transition: VehicleEvidenceTransition = {
      stableId: 'housing-last-0',
      result_id: 'a'.repeat(64),
      presentation_day: 3,
      day: 2,
      manifest: {
        id: 'd2-housing-last-0',
        wave: 'last-mile',
        fleet: 'brick flatbed',
        origin: 'Residential point of distribution',
        destination: 'Residential site 12',
        cargo_units: 6,
        cargo_pallets: 6,
        return_leg: 'returns empty to district depot',
        scheduled_stop: null,
      },
      cycle_state: 'outbound',
      progress: 0.25,
      position: [1, 0.42, 2],
      timestamp_ms: 123.5,
    }
    const captured: VehicleEvidenceTransition[] = []

    expect(publishVehicleEvidenceTransition(transition, undefined)).toBe(false)
    expect(publishVehicleEvidenceTransition(transition, {
      vehicleId: 'food-last-0',
      onVehicleTransition: (event) => captured.push(event),
    })).toBe(false)
    expect(publishVehicleEvidenceTransition(transition, {
      vehicleId: transition.stableId,
      onVehicleTransition: (event) => captured.push(event),
    })).toBe(true)
    expect(captured).toEqual([transition])
  })

  it('routes active 3D last-mile vehicles to the exact same rebuilding cohort sites as manifests', () => {
    const previous = day(1, {
      services_end: [0.5, 0.42, 0.5, 0.5, 0.5],
      services_after_shock: [0.5, 0.42, 0.5, 0.5, 0.5],
    })
    const current = day(2, {
      services_before: previous.services_end,
      services_after_shock: previous.services_end,
      services_end: [0.51, 0.49, 0.51, 0.51, 0.51],
      allocation: [30, 42, 36, 40, 32],
    })
    const manifests = vehicleManifestsForDay(current, previous, SERVICES)
      .filter((manifest) => manifest.service === 'housing' && manifest.wave === 'last-mile')
    const plans = vehiclePlansForDay(result([previous, current]), 1)
      .filter((plan) => plan.active && plan.manifest.service === 'housing' && plan.manifest.wave === 'last-mile')
    expect(plans.map((plan) => plan.manifest.destination)).toEqual(manifests.map((manifest) => manifest.destination))
    plans.forEach((plan) => {
      const buildingIndex = Number(plan.manifest.destination.match(/site (\d+)$/)?.[1]) - 1
      const building = CITY_BUILDING_PLACEMENTS
        .filter((placement) => placement.service === 'housing')[buildingIndex]
      const curb = buildingCurbFor('housing', buildingIndex)
      expect(plan.path.at(-1)).toEqual(curb)
      expect(isPointOnCityRoad(curb)).toBe(true)
      expect(Math.hypot(curb[0] - building.position[0], curb[2] - building.position[2]))
        .toBeGreaterThanOrEqual(1.29)
      expect(Math.hypot(curb[0] - building.position[0], curb[2] - building.position[2]))
        .toBeLessThanOrEqual(1.31)
    })
  })

  it('uses the same deterministic fallback destination and an empty hub return when no cohort is active', () => {
    const current = day(1, {
      services_end: [0.62, 0.58, 0.64, 0.61, 0.59],
      gain: [0, 0, 0, 0, 0],
    })
    const manifests = vehicleManifestsForDay(current, undefined, SERVICES)
      .filter((manifest) => manifest.wave === 'last-mile')
    const plans = vehiclePlansForDay(result([current]), 0)
      .filter((plan) => plan.active && plan.manifest.wave === 'last-mile')
    expect(plans.map((plan) => plan.manifest.destination)).toEqual(manifests.map((manifest) => manifest.destination))
    const lineHaul = vehiclePlansForDay(result([current]), 0)
      .filter((plan) => plan.manifest.wave === 'line-haul')
    expect(lineHaul.every((plan) => plan.manifest.returnLeg === 'returns empty to hub staging')).toBe(true)
    expect(lineHaul.some((plan) => plan.manifest.returnLeg.includes('fuel'))).toBe(false)
  })

  it('routes one deterministic active line-haul mission through a visible hub fuel dwell', () => {
    const current = day(1, {
      allocation: [30, 42, 36, 40, 32],
      services_end: [0.62, 0.58, 0.64, 0.61, 0.59],
    })
    const currentResult = result([current])
    const manifests = vehicleManifestsForDay(current, undefined, SERVICES)
      .filter((manifest) => manifest.scheduledStop)
    const plans = vehiclePlansForDay(currentResult, 0)
      .filter((plan) => plan.manifest.scheduledStop)

    expect(scheduledFuelServiceForDay(current, SERVICES)).toBe('housing')
    expect(manifests).toHaveLength(1)
    expect(plans).toHaveLength(1)
    expect(manifests[0].scheduledStop).toBe(FUEL_POINT_STOP_COPY)
    expect(plans[0].manifest.scheduledStop).toBe(manifests[0].scheduledStop)
    expect(plans[0].path).toContainEqual(CITY_HUB.fuelLane)
    expect(vehicleCycleState(0.18, true)).toBe('fuel')
    expect(vehiclePositionForCycle(plans[0], 0.18)).toEqual(CITY_HUB.fuelLane)

    const nextDay = day(2, { allocation: current.allocation })
    expect(scheduledFuelServiceForDay(nextDay, SERVICES)).toBe('healthcare')
    expect(vehicleManifestsForDay(nextDay, current, SERVICES).filter((manifest) => manifest.scheduledStop)).toHaveLength(1)
  })

  it('uses recorded v2 stock, throughput, queues, repair supply, and mutual aid without synthesizing them', () => {
    const current = day(3, {
      allocation: [30, 42, 36, 40, 32],
      logistics: {
        depot_capacity: [400, 400, 400, 400, 400],
        depot_stock_before: [130, 90, 40, 125, 145],
        pending_arrivals: [8, 9, 10, 11, 12],
        pending_arrivals_landed: [8, 9, 10, 11, 12],
        pending_arrivals_held: [0, 0, 0, 0, 0],
        depot_stock_after_pending: [138, 99, 50, 136, 157],
        depot_damage_penalty: [0.1, 0.35, 0.2, 0.3, 0.4],
        depot_damage_days_remaining: [2, 5, 3, 4, 4],
        depot_damage_factor: [0.3, 0.65, 0.8, 0.7, 0.6],
        road_capacity: 0.77,
        throughput_factor: [0.9, 0.5005, 0.616, 0.539, 0.462],
        mutual_aid_transfers: [{
          from_service: 'transport',
          to_service: 'food',
          units: 20,
          donor_stock_fraction_before: 0.345,
          receiver_stock_fraction_before: 0.125,
        }],
        mutual_aid_net: [-20, 0, 20, 0, 0],
        depot_stock_ready: [118, 99, 70, 136, 157],
        pending_next_day: [10.5, 14.7, 12.6, 14, 11.2],
        same_day_delivery_scheduled: [19.5, 27.3, 23.4, 26, 20.8],
        same_day_delivery_landed: [19.5, 27.3, 23.4, 26, 20.8],
        same_day_delivery_held: [0, 0, 0, 0, 0],
        delayed_delivery_scheduled: [10.5, 14.7, 12.6, 14, 11.2],
        repair_reserve: [6.08, 6.72, 5.76, 6.24, 6.56],
        repair_request: [36.08, 48.72, 41.76, 46.24, 38.56],
        repair_dispatch: [36.08, 48.72, 41.76, 46.24, 38.56],
        repair_supply: [32.472, 24.38436, 25.72416, 24.92336, 17.81472],
        spoilage: [0, 0, 0.40605504, 0, 0],
        depot_stock_end: [105.028, 101.91564, 67.26978496, 137.07664, 159.98528],
        capacity_overflow: [0, 0, 0, 0, 0],
        conservation_residual: [0, 0, 0, 0, 0],
      },
    })
    const depots = depotStatusesForDay(current, SERVICES)
    expect(depots.find((depot) => depot.service === 'transport')?.damage).toBe('rubble')
    const food = depots.find((depot) => depot.service === 'food')!
    expect(food).toMatchObject({
      engineTruth: true,
      palletUnits: 67.26978496,
      stockCapacity: 400,
      pendingUnits: 12.6,
      spoilageUnits: 0.40605504,
      throughputSignal: 0.616,
      reroutedFrom: null,
      mutualAidFrom: 'transport',
    })
    expect(food.recorded).toEqual({
      stockBeforeUnits: 40,
      stockReadyUnits: 70,
      stockEndUnits: 67.26978496,
      pendingLandedUnits: 10,
      sameDayLandedUnits: 23.4,
      landedUnits: 33.4,
      capacityHeldUnits: 0,
      repairDispatchUnits: 41.76,
      repairSupplyUnits: 25.72416,
      damagePenalty: 0.2,
      damageDaysRemaining: 3,
      damageFactor: 0.8,
      throughputFactor: 0.616,
      scheduledOrHeldNextDayUnits: 12.6,
      spoilageUnits: 0.40605504,
      mutualAidNetUnits: 20,
    })
    expect(food.source).toContain('recorded engine-v2')
    const inWorldCopy = depotPresentationCopy(food)
    expect(inWorldCopy.condition).toContain('visual interpolation')
    expect(inWorldCopy.stock).toContain('visual stock interpolation')
    expect(inWorldCopy.flow).toContain('Visual interpolation: 33.4 units landed')
    expect(inWorldCopy.flow).toContain('25.7 effective repair supply')
    expect(Object.values(inWorldCopy).filter(Boolean).join(' ')).not.toMatch(/recorded/i)
    expect(IN_WORLD_INTERPOLATION_DISCLOSURE).toContain('exact daily values in Analyst Toolbox')
    expect(intakeHubPresentationCopy(current.available_budget)).toContain('visual budget interpolation')
    expect(intakeHubPresentationCopy(current.available_budget)).toContain('exact daily value in Analyst Toolbox')

    const manifests = vehicleManifestsForDay(current, undefined, SERVICES)
    const foodLineHaul = manifests.filter((manifest) => manifest.service === 'food' && manifest.wave === 'line-haul' && !manifest.fleet.includes('mutual-aid'))
    const foodLastMile = manifests.filter((manifest) => manifest.service === 'food' && manifest.wave === 'last-mile')
    const mutualAid = manifests.filter((manifest) => manifest.fleet === 'mutual-aid line-haul truck')
    expect(foodLineHaul.reduce((sum, item) => sum + item.cargoUnits, 0)).toBe(33.4)
    expect(foodLastMile.reduce((sum, item) => sum + item.cargoUnits, 0)).toBe(25.72416)
    expect(mutualAid.reduce((sum, item) => sum + item.cargoUnits, 0)).toBe(20)
    expect(mutualAid.every((manifest) => manifest.wave === 'mutual-aid')).toBe(true)
    const rubbleDepotLastMile = vehiclePlansForDay(result([current]), 0)
      .filter((plan) => plan.manifest.service === 'transport' && plan.manifest.wave === 'last-mile')
    expect(rubbleDepotLastMile.length).toBeGreaterThan(0)
    expect(rubbleDepotLastMile.every((plan) => plan.active)).toBe(true)
    expect(vehicleDispatchCountsForDay(current, undefined, SERVICES)).toEqual({
      lineHaulHeavyTrucks: 17,
      lastMileVehicles: 35,
      mutualAidVehicles: 2,
    })
    expect(throughputVehiclesPerDay(current)).toBe(54)
    expect(relayNarration(result([current]), 0)).toContain('DEPOT LEDGER:')
    expect(relayNarration(result([current]), 0)).toContain('36.3 UNITS LANDED')
    expect(relayNarration(result([current]), 0)).toContain('MUTUAL AID: 20.0 UNITS TRANSPORT TO FOOD')
    expect(relayNarration(result([current]), 0)).toContain('0.41 FOOD UNITS EXPIRED IN STORAGE')

    const inconsistentNet: DayResult = {
      ...current,
      logistics: { ...current.logistics!, mutual_aid_net: [0, 0, 0, 0, 0] },
    }
    expect(depotStatusesForDay(inconsistentNet, SERVICES).find((depot) => depot.service === 'food')?.mutualAidFrom).toBeNull()
    expect(vehicleManifestsForDay(inconsistentNet, undefined, SERVICES).some((manifest) => manifest.wave === 'mutual-aid')).toBe(false)
    expect(vehiclePlansForDay(result([inconsistentNet]), 0).some((plan) => plan.manifest.wave === 'mutual-aid' && plan.active)).toBe(false)
    expect(relayNarration(result([inconsistentNet]), 0)).not.toContain('MUTUAL AID:')
  })

  it('keeps schema-v2 depot truth absent and uses the exact legacy disclosure', () => {
    const legacy = day(4, { allocation: [28.5, 38.5, 31, 44, 38] })
    const depots = depotStatusesForDay(legacy, SERVICES)
    expect(legacy.logistics).toBeUndefined()
    expect(depots.every((depot) => depot.recorded === null)).toBe(true)
    expect(depots.every((depot) => depot.source === LEGACY_V1_DEPOT_DISCLOSURE)).toBe(true)
    expect(depots.every((depot) => depot.damage === null && depot.throughputSignal === null && depot.dockQueue === null)).toBe(true)
    expect(depots.every((depot) => depot.dispatchedVehicles > 0 && depot.lastMileVehicles > 0)).toBe(true)
    expect(vehicleManifestsForDay(legacy, undefined, SERVICES).some((manifest) => manifest.wave === 'mutual-aid')).toBe(false)
    expect(relayNarration(result([legacy]), 0)).not.toContain('DEPOT LEDGER:')
  })
})

describe('real incident and recovery derivations', () => {
  const earthquake: Shock = {
    day: 2, type: 'aftershock', severity: 0.32, impact: [0.65, 1, 0.2, 0.35, 0.45], budget_factor: 0.15, forced: true,
  }
  const trajectory = [
    day(1, { services_end: [0.7, 0.7, 0.7, 0.7, 0.7] }),
    day(2, { shock: earthquake, services_before: [0.7, 0.7, 0.7, 0.7, 0.7], services_after_shock: [0.55, 0.42, 0.65, 0.62, 0.6], services_end: [0.57, 0.45, 0.67, 0.64, 0.62], available_budget: 171.36, allocation: [30, 42, 33, 36, 30.36] }),
    day(3, { services_before: [0.57, 0.45, 0.67, 0.64, 0.62], services_after_shock: [0.57, 0.45, 0.67, 0.64, 0.62], services_end: [0.61, 0.53, 0.7, 0.68, 0.66] }),
    day(4, { services_before: [0.61, 0.53, 0.7, 0.68, 0.66], services_after_shock: [0.61, 0.53, 0.7, 0.68, 0.66], services_end: [0.66, 0.62, 0.72, 0.71, 0.7] }),
    day(5, { services_before: [0.66, 0.62, 0.72, 0.71, 0.7], services_after_shock: [0.66, 0.62, 0.72, 0.71, 0.7], services_end: [0.71, 0.71, 0.74, 0.73, 0.72] }),
  ]
  const run = result(trajectory)

  it('preserves the true multi-day recovery arc and ordered incident phases', () => {
    const arc = recoveryArcForService(run, 2, 'housing')
    expect(arc.startedDay).toBe(2)
    expect(arc.completionDay).toBe(5)
    expect(arc.durationDays).toBe(4)
    expect(arc.progress).toBeGreaterThan(0)
    expect(arc.progress).toBeLessThan(1)
    expect(incidentPhaseForDay({ result: run, dayIndex: 1, telegraph: true, impact: false })).toBe('TELEGRAPH')
    expect(incidentPhaseForDay({ result: run, dayIndex: 1, telegraph: false, impact: true })).toBe('IMPACT')
    expect(incidentPhaseForDay({ result: run, dayIndex: 1, telegraph: false, impact: false, postImpactPhase: 'assessment' })).toBe('ASSESSMENT')
    expect(incidentPhaseForDay({ result: run, dayIndex: 1, telegraph: false, impact: false, postImpactPhase: 'response' })).toBe('RESPONSE')
    expect(incidentPhaseForDay({ result: run, dayIndex: 2, telegraph: false, impact: false })).toBe('RESPONSE')
  })

  it('derives road closure, detour, rail state, and later repair patches from transport truth', () => {
    const closedRun = result([day(1, { services_end: [0.2, 0.6, 0.6, 0.6, 0.6] })])
    expect(roadNetworkState(closedRun, 0)).toMatchObject({ condition: 'closed', detourActive: true, railAvailable: false })
    const recovered = roadNetworkState(run, 3)
    expect(recovered.freshPatches).toBeGreaterThan(0)
  })

  it('keeps quake debris and road-repair residue on the shared continuous service sample', () => {
    const beforeImpact = roadNetworkState(run, 1, trajectory[1].services_before)
    const duringImpact = roadNetworkState(run, 1, [0.60, 0.54, 0.68, 0.66, 0.65])
    expect(beforeImpact).toMatchObject({ quakeProgress: 1, debrisSegments: 0 })
    expect(duringImpact.quakeProgress).toBeCloseTo((0.60 - 0.55) / (0.70 - 0.55))
    expect(duringImpact.debrisSegments).toBeGreaterThan(0)

    // Day 4 starts at day 3's exact end. Current-day gain is still zero at
    // this cursor, so visible patches can only be truthful residue from day 3.
    const dayFourStart = roadNetworkState(run, 3, trajectory[2].services_end)
    expect(dayFourStart.freshPatches).toBeGreaterThan(0)
    expect(dayFourStart.quakeProgress).toBeCloseTo((0.61 - 0.55) / (0.70 - 0.55))
  })

  it('does not reclassify an unchanged earthquake-damaged road from integer incident age at midnight', () => {
    const unchangedPresentedServices = [0.60, 0.54, 0.68, 0.66, 0.65]
    const dayThreeEnd = roadNetworkState(run, 2, unchangedPresentedServices)
    const dayFourStart = roadNetworkState(run, 3, unchangedPresentedServices)

    expect(dayThreeEnd.quakeAge).toBe(1)
    expect(dayFourStart.quakeAge).toBe(2)
    expect(dayFourStart.quakeLoad).toBeCloseTo(dayThreeEnd.quakeLoad)
    expect(dayFourStart.condition).toBe(dayThreeEnd.condition)
    expect(dayFourStart.condition).toBe('restricted')
  })

  it('keeps typed quake infrastructure until its own transport arc recovers across a later shock', () => {
    const supplyShock: Shock = {
      day: 3, type: 'supply', severity: 0.2, impact: [0.08, 0.02, 0.3, 0.15, 0.03], budget_factor: 0.2, forced: true,
    }
    const overlap = result([
      trajectory[0],
      trajectory[1],
      day(3, {
        shock: supplyShock,
        services_before: trajectory[1].services_end,
        services_after_shock: [0.57, 0.45, 0.55, 0.60, 0.61],
        services_end: [0.60, 0.50, 0.58, 0.63, 0.64],
      }),
    ])
    expect(roadNetworkState(overlap, 2)).toMatchObject({ quakeAge: 1 })
    expect(roadNetworkState(overlap, 2).debrisSegments).toBeGreaterThan(0)
  })

  it('keeps older typed residue live on the shared sample while withholding only the current incident', () => {
    const supplyShock: Shock = {
      day: 3, type: 'supply', severity: 0.2, impact: [0.08, 0.02, 0.3, 0.15, 0.03], budget_factor: 0.2, forced: true,
    }
    const overlap = result([
      trajectory[0],
      trajectory[1],
      day(3, {
        shock: supplyShock,
        services_before: trajectory[1].services_end,
        services_after_shock: [0.55, 0.43, 0.53, 0.59, 0.60],
        services_end: [0.60, 0.50, 0.58, 0.63, 0.64],
      }),
    ])

    const olderQuake = latestPresentedIncidentOfType(overlap, 2, 'aftershock', false)
    expect(olderQuake).toMatchObject({ dayIndex: 1, ageDays: 1, type: 'aftershock' })
    expect(latestPresentedIncident(overlap, 2, false)).toMatchObject({ dayIndex: 1, type: 'aftershock' })
    expect(latestPresentedIncident(overlap, 2, true)).toMatchObject({ dayIndex: 2, type: 'supply' })

    const lowSample = [0.56, 0.46, 0.55, 0.60, 0.61]
    const recoveringSample = [0.60, 0.52, 0.58, 0.63, 0.64]
    expect(presentedIncidentRecovery(overlap, olderQuake!, 'housing', recoveringSample))
      .toBeGreaterThan(presentedIncidentRecovery(overlap, olderQuake!, 'housing', lowSample))
  })

  it('keeps critical smoke topology stable and derives every visual parameter continuously from strength', () => {
    const clear = criticalSmokePresentation(0)
    const middle = criticalSmokePresentation(0.5)
    const dark = criticalSmokePresentation(1)

    expect([clear.puffCount, middle.puffCount, dark.puffCount]).toEqual([
      CRITICAL_SMOKE_PUFF_COUNT,
      CRITICAL_SMOKE_PUFF_COUNT,
      CRITICAL_SMOKE_PUFF_COUNT,
    ])
    expect(middle.opacity).toBeGreaterThan(clear.opacity)
    expect(middle.opacity).toBeLessThan(dark.opacity)
    expect(middle.rise).toBeGreaterThan(clear.rise)
    expect(middle.rise).toBeLessThan(dark.rise)
    expect(criticalSmokePresentation(0.500001).opacity - middle.opacity).toBeLessThan(0.000001)
  })

  it('parks a started crew through stalled ticks and scales its visual repair by the recorded arc', () => {
    const recovery = result([
      day(1, { services_end: [0.7, 0.7, 0.7, 0.7, 0.7] }),
      day(2, { shock: earthquake, services_before: [0.7, 0.7, 0.7, 0.7, 0.7], services_after_shock: [0.55, 0.4, 0.65, 0.62, 0.6], services_end: [0.57, 0.42, 0.67, 0.64, 0.62] }),
      day(3, { services_before: [0.57, 0.42, 0.67, 0.64, 0.62], services_after_shock: [0.57, 0.42, 0.67, 0.64, 0.62], services_end: [0.60, 0.48, 0.69, 0.66, 0.64], allocation: [30, 60, 30, 30, 30] }),
      day(4, { services_before: [0.60, 0.48, 0.69, 0.66, 0.64], services_after_shock: [0.60, 0.48, 0.69, 0.66, 0.64], services_end: [0.60, 0.48, 0.69, 0.66, 0.64], allocation: [30, 60, 30, 30, 30] }),
      day(5, { services_before: [0.60, 0.48, 0.69, 0.66, 0.64], services_after_shock: [0.60, 0.48, 0.69, 0.66, 0.64], services_end: [0.64, 0.58, 0.71, 0.69, 0.67], allocation: [30, 60, 30, 30, 30] }),
      day(6, { services_before: [0.64, 0.58, 0.71, 0.69, 0.67], services_after_shock: [0.64, 0.58, 0.71, 0.69, 0.67], services_end: [0.71, 0.71, 0.74, 0.73, 0.72], allocation: [30, 60, 30, 30, 30] }),
      day(7, { services_end: [0.71, 0.71, 0.74, 0.73, 0.72] }),
      day(8, { services_end: [0.71, 0.71, 0.74, 0.73, 0.72] }),
    ])
    const assigned = rebuildingCohortForDay(recovery.candidate.trajectory[2], recovery.candidate.trajectory[1], 'housing')[0]
    expect(assigned).toBeTypeOf('number')
    expect(buildingRepairStarted(recovery, 3, 'housing', assigned)).toBe(true)
    expect(repairProgressForBuilding(recovery, 2, 'housing', assigned)).toBe(0)
    expect(repairProgressForBuilding(recovery, 3, 'housing', assigned)).toBe(0)
    expect(repairProgressForBuilding(recovery, 4, 'housing', assigned)).toBeGreaterThan(0)
    expect(repairProgressForBuilding(recovery, 5, 'housing', assigned)).toBe(1)
    expect(repairProgressForBuildingAt(recovery, 4, 0, 'housing', assigned)).toBe(0)
    expect(repairProgressForBuildingAt(recovery, 4, 1, 'housing', assigned)).toBe(
      repairProgressForBuilding(recovery, 4, 'housing', assigned),
    )
    expect(repairProgressForBuildingAt(recovery, 4, 0.9, 'housing', assigned)).toBeGreaterThan(0)
    expect(repairStageForBuildingAt(recovery, 4, 0, 'housing', assigned)).toBe('assessment')
    const initial = damageStateFor(0.4, assigned)
    expect(stagedRepairDamageState(initial, 'intact', true, 0)).toBe(initial)
    expect(stagedRepairDamageState(initial, 'intact', true, 1)).toBe('intact')
    expect(repairFreshnessForBuilding(recovery, 5, 'housing', assigned)).toBe(1)
    expect(repairFreshnessForBuilding(recovery, 7, 'housing', assigned)).toBe(0.5)
    expect(repairFreshnessForBuildingAt(recovery, 7, 0, 'housing', assigned)).toBe(
      repairFreshnessForBuilding(recovery, 6, 'housing', assigned),
    )
    expect(repairFreshnessForBuildingAt(recovery, 7, 1, 'housing', assigned)).toBe(0.5)

    const joiningCohort = rebuildingCohortForDay(
      recovery.candidate.trajectory[4],
      recovery.candidate.trajectory[3],
      'housing',
    )
    const offsets = joiningCohort.map((buildingIndex) => buildingRepairPresentationOffset(
      recovery,
      4,
      'housing',
      buildingIndex,
    ))
    expect(new Set(offsets.map((value) => value.toFixed(6))).size).toBeGreaterThan(1)
    const latest = joiningCohort[offsets.indexOf(Math.max(...offsets))]
    expect(repairProgressForBuildingAt(recovery, 4, Math.max(...offsets) - 0.001, 'housing', latest)).toBe(0)
    expect(repairProgressForBuildingAt(recovery, 4, Math.max(...offsets) + 0.08, 'housing', latest)).toBeGreaterThan(0)
  })

  it('pairs qualitative incident bands with raw-number reporting inputs', () => {
    expect(intensityBand('aftershock', 0.32)).toBe('severe shaking')
    expect(intensityBand('weather', 0.2)).toBe('major storm')
    for (const type of ['aftershock', 'supply', 'epidemic', 'utility', 'weather'] as ShockType[]) {
      expect(intensityBand(type, 0.2).length).toBeGreaterThan(5)
    }
  })

  it('derives debrief milestones, exact arrival shortfall, and per-incident AAR from the same run', () => {
    expect(shockAdjustedArrivalShortfall(run)).toBeCloseTo(8.64)
    expect(recoveryMilestones(run).some((milestone) => milestone.service === 'housing')).toBe(false)
    const reports = afterActionReports(run)
    expect(reports).toHaveLength(1)
    expect(reports[0]).toMatchObject({
      day: 2,
      type: 'aftershock',
      strongestService: 'housing',
      emergencyWaveVehicles: 4,
      lineHaulHeavyTrucks: 16,
      lastMileVehicles: 45,
      mutualAidVehicles: 0,
      logisticsRecorded: false,
    })
    expect(reports[0].recoveryDays.housing).toBe(3)
  })
})

describe('anti-fabrication ledger', () => {
  it('documents a source, derivation, limit, and contract item for every visible behavior family', () => {
    expect(REALISM_LEDGER.length).toBeGreaterThanOrEqual(30)
    for (const entry of REALISM_LEDGER) {
      expect(entry.source.length).toBeGreaterThan(3)
      expect(entry.derivation.length).toBeGreaterThan(10)
      expect(entry.doesNotMean.length).toBeGreaterThan(10)
      expect(entry.items).toMatch(/\d/)
    }
    expect(REALISM_LEDGER.some((entry) => entry.items.includes('137'))).toBe(true)
    expect(JSON.stringify(REALISM_LEDGER)).not.toMatch(/\baftershock\b/i)
    expect(JSON.stringify(REALISM_LEDGER)).toContain('presentation-only aim/drop district')
  })
})
