import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { Group, InstancedMesh } from 'three'
import type { CompareResponse, DayResult, Service } from '../types'
import { DISTRICTS } from './model'
import {
  HEAVY_TRUCK_CAPACITY,
  LAST_MILE_CAPACITY,
  depotStatusesForDay,
  deterministicUnit,
  incidentRecoveryProgress,
  latestIncidentOfType,
  rebuildingDestinationsForDay,
  FUEL_POINT_STOP_COPY,
  scheduledFuelServiceForDay,
  serviceDestination,
  splitExactCargo,
  strongestShockService,
  type VehicleManifest,
} from './realism'
import {
  CITY_BUILDING_PLACEMENTS,
  CITY_DEPOTS,
  CITY_DISTRICTS,
  CITY_HUB,
  CITY_LOCAL_STREET_OFFSETS,
  CITY_ROUTE_WAYPOINTS,
  type WorldPoint,
} from './worldLayout'
import type { RenderQualityProfile } from './renderQuality'
import { GAME_DAY_DURATION_MS } from './pacing'
import { SHOCK_RESPONSE_START_FRACTION } from './presentation'

export type VehiclePlan = {
  stableId: string
  manifest: VehicleManifest
  path: readonly WorldPoint[]
  segmentLengths: readonly number[]
  pathLength: number
  active: boolean
  stagePosition: WorldPoint
  speed: number
  scale: readonly [number, number, number]
  color: string
  cabinColor: string
  emergency: boolean
  broken: boolean
  relief: boolean
  fuelStop: WorldPoint | null
}

export type VehicleCycleState = 'load' | 'fuel' | 'outbound' | 'dock' | 'return' | 'stage'
export type VehicleFleetMode = 'full' | 'assessment'
export type VehicleDockDwell = { id: string; active: boolean; strength: number }
export const VEHICLE_MISSION_STAGE_PROGRESS = 0.95
export const VISIBLE_VEHICLE_LIMITS = {
  inbound: 2,
  lineHaul: 4,
  lastMile: 4,
  mutualAid: 1,
  emergency: 2,
  support: 1,
  civilian: 2,
  commuter: 1,
} as const
export const MAX_VISIBLE_ROAD_VEHICLES = Object.values(VISIBLE_VEHICLE_LIMITS)
  .reduce((sum, count) => sum + count, 0)

export type VehicleEvidenceTransition = {
  stableId: string
  result_id: string
  /** Returned day currently visible while this (possibly earlier) mission continues. */
  presentation_day: number
  /** Returned manifest day; a slow mission can honestly span several presentation days. */
  day: number
  manifest: {
    id: string
    wave: VehicleManifest['wave']
    fleet: string
    origin: string
    destination: string
    cargo_units: number
    cargo_pallets: number
    return_leg: string
    scheduled_stop: string | null
  }
  cycle_state: VehicleCycleState
  progress: number
  position: WorldPoint
  timestamp_ms: number
}

export type RelayVehicleEvidenceHook = {
  /** Verification-only selector; production never installs this hook. */
  vehicleId: string
  onVehicleTransition: (transition: VehicleEvidenceTransition) => void
}

declare global {
  interface Window {
    /** Read-only Playwright evidence hook. Absent in normal application use. */
    __RELAY_EVIDENCE__?: RelayVehicleEvidenceHook
  }
}

/** Publishes only to an explicitly installed verification hook and never mutates game state. */
export function publishVehicleEvidenceTransition(
  transition: VehicleEvidenceTransition,
  hook: RelayVehicleEvidenceHook | undefined = typeof window === 'undefined'
    ? undefined
    : window.__RELAY_EVIDENCE__,
): boolean {
  if (!hook || hook.vehicleId !== transition.stableId) return false
  try {
    hook.onVehicleTransition(transition)
    return true
  } catch {
    // Evidence collection must never affect the simulation or renderer.
    return false
  }
}

const SERVICE_COLORS: Readonly<Record<Service, string>> = {
  transport: '#5a8290',
  housing: '#bd6b52',
  food: '#d49a3d',
  healthcare: '#d9ded7',
  public_services: '#71866a',
}

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, value))
}

/** Dawn staging offset: the largest returned allocation is the only ordinary wave already outbound. */
export function dispatchStartProgress(
  plan: Pick<VehiclePlan, 'stableId' | 'manifest'>,
  seed: number,
  day: number,
): number {
  const rank = plan.manifest.dispatchRank
  const base = rank < 0 ? 0.24 : rank === 0 ? 0.19 : Math.max(0.025, 0.135 - rank * 0.022)
  const salt = [...plan.stableId].reduce((total, character) => total + character.charCodeAt(0), 0)
  return Math.min(0.245, base + deterministicUnit(seed, day, 830 + salt) * 0.004)
}

export function vehicleCycleState(progress: number, visitsFuel = false): VehicleCycleState {
  // Keep exact authored phase boundaries exact; the modulo expression alone
  // turns 0.90 into 0.899999… on some engines and delays staging by a frame.
  const normalized = progress >= 0 && progress <= 1
    ? progress
    : ((progress % 1) + 1) % 1
  if (visitsFuel) {
    if (normalized < 0.10) return 'load'
    if (normalized < 0.21) return 'fuel'
    if (normalized < 0.48) return 'outbound'
    if (normalized < 0.62) return 'dock'
    if (normalized < 0.90) return 'return'
    return 'stage'
  }
  if (normalized < 0.13) return 'load'
  if (normalized < 0.45) return 'outbound'
  if (normalized < 0.59) return 'dock'
  if (normalized < 0.90) return 'return'
  return 'stage'
}

/** A returned daily mission advances once, then remains staged until its day key changes. */
export function advanceVehicleMissionProgress(progress: number, increment: number): number {
  const current = Number.isFinite(progress) ? clamp(progress, 0, VEHICLE_MISSION_STAGE_PROGRESS) : 0
  const forward = Number.isFinite(increment) ? Math.max(0, increment) : 0
  return Math.min(VEHICLE_MISSION_STAGE_PROGRESS, current + forward)
}

/**
 * Samples one mission from run identity plus the shared presentation time.
 * Playback speed, pausing, and aiming are already encoded in absoluteDay, so
 * changing those controls never resets or double-scales a journey.
 */
export function vehicleMissionProgressAt(
  plan: Pick<VehiclePlan, 'stableId' | 'manifest' | 'speed' | 'pathLength' | 'fuelStop'>,
  seed: number,
  engineDay: number,
  missionDayIndex: number,
  absoluteDay: number,
  weatherMultiplier = 1,
  accessibilityMultiplier = 1,
): number {
  const start = dispatchStartProgress(plan, seed, engineDay)
  const elapsedDays = Math.max(0, Number.isFinite(absoluteDay) ? absoluteDay - missionDayIndex : 0)
  const progressPerDay = vehicleMissionProgressPerDay(plan, weatherMultiplier, accessibilityMultiplier)
  return advanceVehicleMissionProgress(start, elapsedDays * progressPerDay)
}

export function vehicleMissionProgressPerDay(
  plan: Pick<VehiclePlan, 'speed' | 'pathLength' | 'fuelStop'>,
  weatherMultiplier = 1,
  accessibilityMultiplier = 1,
): number {
  const travelProgressSpan = plan.fuelStop ? 0.315 : 0.32
  return (GAME_DAY_DURATION_MS / 1_000)
    * Math.max(0, plan.speed)
    * travelProgressSpan
    / Math.max(1e-6, plan.pathLength)
    * Math.max(0, weatherMultiplier)
    * Math.max(0, accessibilityMultiplier)
}

/** Cargo remains aboard through destination dwell, then is absent on return/staging. */
export function vehicleCarriesCargo(
  manifest: Pick<VehicleManifest, 'cargoUnits'>,
  state: VehicleCycleState,
): boolean {
  return manifest.cargoUnits > 0
    && state !== 'return'
    && state !== 'stage'
}

/** Current-leg copy: manifest totals stay auditable without describing an empty leg as loaded. */
export function vehicleCargoCopy(manifest: VehicleManifest, state: VehicleCycleState): string {
  if (manifest.cargoUnits <= 0) {
    if (manifest.wave === 'emergency') return 'Damage assessment mission · no supply cargo'
    if (manifest.fleet === 'relief truck') return 'Roadside assistance mission · no supply cargo'
    if (manifest.wave === 'maintenance') return 'Maintenance mission · no supply cargo'
    return 'No supply cargo'
  }
  const load = `${manifest.cargoPallets.toFixed(1)} pallets = ${manifest.cargoUnits.toFixed(1)} supply units`
  if (state === 'return') return `Empty return · ${load} delivered at destination`
  if (state === 'stage') return `Staged empty · ${load} delivered at destination`
  return load
}

/** Compare reruns within a presentation day do not authorize a second daily dispatch. */
export function vehicleMissionDayKey(result: CompareResponse, dayIndex: number): string {
  return `${result.seed}:${result.candidate.trajectory[dayIndex].day}`
}

function serviceFleet(service: Service): string {
  if (service === 'healthcare') return 'ambulance'
  if (service === 'housing') return 'brick flatbed'
  if (service === 'food') return 'refrigerated truck'
  if (service === 'public_services') return 'bucket truck'
  return 'recovery bus'
}

export type VehicleSilhouette =
  | 'ambulance'
  | 'rapid-assessment'
  | 'flatbed'
  | 'refrigerated'
  | 'bucket-truck'
  | 'bus'
  | 'road-sweeper'
  | 'loader'
  | 'dump-truck'
  | 'rail'
  | 'car'
  | 'van'
  | 'truck'

/**
 * Presentation-only silhouette classification. The manifest remains the sole
 * source of fleet truth; this helper only chooses procedural bodywork.
 */
export function vehicleSilhouetteForFleet(fleet: string): VehicleSilhouette {
  const label = fleet.toLowerCase()
  if (label.includes('ambulance')) return 'ambulance'
  if (label.includes('rapid assessment')) return 'rapid-assessment'
  if (label.includes('flatbed')) return 'flatbed'
  if (label.includes('refrigerated')) return 'refrigerated'
  if (label.includes('bucket')) return 'bucket-truck'
  if (label.includes('bus')) return 'bus'
  if (label.includes('sweeper')) return 'road-sweeper'
  if (label.includes('loader')) return 'loader'
  if (label.includes('dump')) return 'dump-truck'
  if (label.includes('rail') || label.includes('wagon')) return 'rail'
  if (label.includes('car')) return 'car'
  if (label.includes('van')) return 'van'
  return 'truck'
}

function scaleForFleet(fleet: string): readonly [number, number, number] {
  const silhouette = vehicleSilhouetteForFleet(fleet)
  if (silhouette === 'bus') return [0.72, 0.58, 1.25]
  if (silhouette === 'ambulance') return [0.62, 0.62, 0.9]
  if (silhouette === 'rapid-assessment') return [0.58, 0.52, 0.82]
  if (silhouette === 'flatbed') return [0.72, 0.48, 1.08]
  if (silhouette === 'refrigerated') return [0.72, 0.72, 1.08]
  if (silhouette === 'bucket-truck') return [0.68, 0.62, 1]
  if (silhouette === 'road-sweeper') return [0.64, 0.5, 0.92]
  if (silhouette === 'loader') return [0.78, 0.64, 0.9]
  if (silhouette === 'dump-truck') return [0.78, 0.65, 1.08]
  if (silhouette === 'rail') return [0.82, 0.64, 1.38]
  if (silhouette === 'car') return [0.48, 0.38, 0.68]
  return [0.68, 0.6, 1]
}

function depotFor(service: Service): (typeof CITY_DEPOTS)[number] {
  return CITY_DEPOTS.find((depot) => depot.service === service)!
}

function compactPath(points: readonly WorldPoint[]): WorldPoint[] {
  return points.filter((point, index) => {
    if (index === 0) return true
    const previous = points[index - 1]
    return Math.hypot(point[0] - previous[0], point[2] - previous[2]) > 1e-6
  })
}

function hubStagePosition(slot: number): WorldPoint {
  return [
    -5.2 + (slot % 4) * 0.72,
    0.43,
    -1.35 + (Math.floor(slot / 4) % 5) * 0.82,
  ]
}

function depotStagePosition(service: Service, slot: number): WorldPoint {
  const depot = depotFor(service)
  if (slot <= 0) return depot.curb
  const directionToHub = -Math.sign(depot.curb[0])
  return [
    depot.curb[0] + directionToHub * slot * 0.68,
    0.43,
    depot.curb[2],
  ]
}

function hubStageToDispatch(slot: number): WorldPoint[] {
  const stage = hubStagePosition(slot)
  return compactPath([
    stage,
    [stage[0], 0.43, -2],
    CITY_HUB.dispatchGate,
  ])
}

function routeFromHubToDistrict(service: Service): WorldPoint[] {
  const district = CITY_DISTRICTS.find((entry) => entry.service === service)!
  return compactPath([
    CITY_HUB.dispatchGate,
    [0, 0.43, -2],
    [0, 0.43, district.center[2]],
    [district.center[0], 0.43, district.center[2]],
  ])
}

function nearestStreetOffset(value: number): number {
  return CITY_LOCAL_STREET_OFFSETS.reduce((nearest, candidate) => (
    Math.abs(candidate - value) < Math.abs(nearest - value) ? candidate : nearest
  ), CITY_LOCAL_STREET_OFFSETS[0])
}

/** Road-side bay for a real rebuilding-site destination; it remains outside the building body. */
export function buildingCurbFor(service: Service, buildingIndex: number): WorldPoint {
  const buildings = CITY_BUILDING_PLACEMENTS.filter((building) => building.service === service)
  const target = buildings[((buildingIndex % buildings.length) + buildings.length) % buildings.length].position
  const district = CITY_DISTRICTS.find((entry) => entry.service === service)!
  const localX = target[0] - district.center[0]
  const localZ = target[2] - district.center[2]
  const streetX = district.center[0] + nearestStreetOffset(localX)
  const streetZ = district.center[2] + nearestStreetOffset(localZ)
  return Math.abs(target[0] - streetX) < Math.abs(target[2] - streetZ)
    ? [streetX, 0.43, target[2]]
    : [target[0], 0.43, streetZ]
}

function districtRouteToBuilding(service: Service, buildingIndex: number): WorldPoint[] {
  const district = CITY_DISTRICTS.find((entry) => entry.service === service)!
  const curb = buildingCurbFor(service, buildingIndex)
  const horizontalStreet = Math.abs(curb[2] - district.center[2]) > 1e-6
  return compactPath(horizontalStreet
    ? [
        [district.center[0], 0.43, district.center[2]],
        [district.center[0], 0.43, curb[2]],
        curb,
      ]
    : [
        [district.center[0], 0.43, district.center[2]],
        [curb[0], 0.43, district.center[2]],
        curb,
      ])
}

function routeFromDepotToBuilding(
  originService: Service,
  service: Service,
  buildingIndex: number,
  slot: number,
): readonly WorldPoint[] {
  const stage = depotStagePosition(originService, slot)
  const originDistrict = CITY_DISTRICTS.find((entry) => entry.service === originService)!
  const localDestination = districtRouteToBuilding(service, buildingIndex)
  if (originService === service) {
    return compactPath([
      stage,
      [originDistrict.center[0], 0.43, stage[2]],
      [originDistrict.center[0], 0.43, originDistrict.center[2]],
      ...localDestination.slice(1),
    ])
  }
  const toHub = [...CITY_ROUTE_WAYPOINTS[originService].direct].reverse()
  const toDistrict = routeFromHubToDistrict(service)
  return compactPath([
    stage,
    ...toHub.slice(1),
    ...toDistrict.slice(1),
    ...localDestination.slice(1),
  ])
}

function routeFromHubToDepot(
  service: Service,
  detour: boolean,
  slot: number,
  visitsFuel: boolean,
): readonly WorldPoint[] {
  const serviceRoute = detour ? CITY_ROUTE_WAYPOINTS[service].detour : CITY_ROUTE_WAYPOINTS[service].direct
  const staging = hubStageToDispatch(slot)
  if (!visitsFuel) return compactPath([...staging, ...serviceRoute.slice(1)])
  return compactPath([
    ...staging,
    [0, 0.43, -2],
    [0, 0.43, 2.7],
    CITY_HUB.fuelLane,
    [0, 0.43, 2.7],
    [0, 0.43, -2],
    ...serviceRoute.slice(1),
  ])
}

function activeLoads(allocation: number, capacity: number, maximum: number): number[] {
  const loads = splitExactCargo(allocation, capacity)
  return Array.from({ length: maximum }, (_, index) => loads[index] ?? 0)
}

function planFromManifest({
  stableId,
  manifest,
  path,
  active,
  index,
  serviceIndex,
  speed,
  broken = false,
  relief = false,
}: {
  stableId: string
  manifest: VehicleManifest
  path: readonly WorldPoint[]
  active: boolean
  index: number
  serviceIndex?: number
  speed: number
  broken?: boolean
  relief?: boolean
}): VehiclePlan {
  const service = manifest.service === 'hub' ? null : manifest.service
  const color = service ? SERVICE_COLORS[service] : '#b9ad94'
  const cabinColor = manifest.fleet.includes('ambulance')
    ? '#e7e5dd'
    : manifest.fleet.includes('refrigerated')
      ? '#d8ddd7'
      : manifest.fleet.includes('car')
        ? '#7c8783'
        : '#cbc5b8'
  const segmentLengths = path.slice(1).map((point, pathIndex) => Math.hypot(
    point[0] - path[pathIndex][0],
    point[2] - path[pathIndex][2],
  ))
  return {
    stableId,
    manifest,
    path,
    segmentLengths,
    pathLength: segmentLengths.reduce((sum, length) => sum + length, 0) || 1,
    active,
    stagePosition: path[0] ?? hubStagePosition(index + (serviceIndex ?? 0)),
    speed,
    scale: scaleForFleet(manifest.fleet),
    color,
    cabinColor,
    emergency: manifest.wave === 'emergency',
    broken,
    relief,
    fuelStop: manifest.scheduledStop ? CITY_HUB.fuelLane : null,
  }
}

/** Stable-capacity fleet: inactive vehicles remain staged instead of disappearing at a day boundary. */
export function vehiclePlansForDay(
  result: CompareResponse,
  dayIndex: number,
): VehiclePlan[] {
  const day = result.candidate.trajectory[dayIndex]
  const previous = result.candidate.trajectory[dayIndex - 1]
  const transportIndex = result.services.indexOf('transport')
  const transport = day.services_end[transportIndex]
  const quakeIncident = latestIncidentOfType(result, dayIndex, 'aftershock')
  const quakeTransportRecovery = quakeIncident
    ? incidentRecoveryProgress(result, quakeIncident.dayIndex, dayIndex, 'transport')
    : 1
  const quakeDebrisActive = quakeTransportRecovery < 1
  const detour = transport < 0.42 || quakeDebrisActive
  const statuses = depotStatusesForDay(day, result.services)
  const plans: VehiclePlan[] = []

  const inboundMaximum = Math.max(1, Math.ceil(result.scenario.daily_budget / HEAVY_TRUCK_CAPACITY))
  const inboundLoads = activeLoads(day.available_budget, HEAVY_TRUCK_CAPACITY, inboundMaximum)
  const rail = transport >= 0.55
  inboundLoads.forEach((cargo, index) => {
    const manifest: VehicleManifest = {
      id: `inbound-${index}`,
      service: 'hub',
      fleet: rail ? 'rail freight wagon' : 'inbound freight truck',
      origin: 'regional supply edge',
      destination: 'central intake hub',
      cargoUnits: cargo,
      cargoPallets: cargo,
      returnLeg: rail ? 'returns empty on the regional rail spur' : 'returns empty beyond the plate edge',
      wave: 'inbound',
      dispatchRank: index,
    }
    plans.push(planFromManifest({
      stableId: `inbound-${index}`,
      manifest,
      path: rail
        ? [CITY_HUB.railIngress, CITY_HUB.railDock]
        : [CITY_HUB.roadIngress, [0, 0.43, -2] as WorldPoint, CITY_HUB.dispatchGate],
      active: cargo > 0,
      index,
      speed: rail ? 0.90 : 0.68,
    }))
  })

  const allocationOrder = result.services
    .map((service, index) => ({ service, allocation: day.allocation[index] }))
    .sort((left, right) => right.allocation - left.allocation)
  const scheduledFuelService = scheduledFuelServiceForDay(day, result.services)

  allocationOrder.forEach(({ service, allocation }, dispatchRank) => {
    const serviceIndex = result.services.indexOf(service)
    const lineHaulUnits = day.logistics
      ? Math.max(0, (day.logistics.pending_arrivals_landed[serviceIndex] ?? 0) + (day.logistics.same_day_delivery_landed[serviceIndex] ?? 0))
      : allocation
    const lastMileUnits = Math.max(0, day.logistics?.repair_supply[serviceIndex] ?? allocation)
    const maxLineHaul = Math.max(...result.candidate.trajectory.map((entry) => entry.logistics
      ? Math.max(0, (entry.logistics.pending_arrivals_landed[serviceIndex] ?? 0) + (entry.logistics.same_day_delivery_landed[serviceIndex] ?? 0))
      : entry.allocation[serviceIndex]))
    const maxLastMile = Math.max(...result.candidate.trajectory.map((entry) => (
      entry.logistics?.repair_supply[serviceIndex] ?? entry.allocation[serviceIndex]
    )))
    const lineMaximum = Math.max(1, Math.ceil(maxLineHaul / HEAVY_TRUCK_CAPACITY))
    const lastMaximum = Math.max(1, Math.ceil(maxLastMile / LAST_MILE_CAPACITY))
    const lineLoads = activeLoads(lineHaulUnits, HEAVY_TRUCK_CAPACITY, lineMaximum)
    const lastLoads = activeLoads(lastMileUnits, LAST_MILE_CAPACITY, lastMaximum)
    const status = statuses.find((entry) => entry.service === service)!
    const throughputMotion = day.logistics ? 0.45 + 0.55 * (status.throughputSignal ?? 0) : 1
    const receivingService = status.reroutedFrom ?? service
    const buildingCount = CITY_BUILDING_PLACEMENTS.filter((building) => building.service === service).length
    const lastMileDestinations = rebuildingDestinationsForDay(
      day,
      previous,
      service,
      dispatchRank,
      buildingCount,
    )
    const breakdown = transport < 0.35
      && deterministicUnit(result.seed, day.day, 260 + serviceIndex) < 0.22

    lineLoads.forEach((cargo, index) => {
      const visitsFuel = cargo > 0 && service === scheduledFuelService && index === 0
      const rerouteDisclosure = status.reroutedFrom
        ? `Recorded local depot rubble: nearest-healthy ${DISTRICTS.find((item) => item.service === receivingService)?.shortLabel} point of distribution presentation route for ${DISTRICTS.find((item) => item.service === service)?.shortLabel}`
        : null
      const manifest: VehicleManifest = {
        id: `${service}-line-${index}`,
        service,
        fleet: 'line-haul truck',
        origin: 'central intake hub',
        destination: status.reroutedFrom
          ? `${DISTRICTS.find((item) => item.service === receivingService)?.shortLabel} point of distribution · support route for ${DISTRICTS.find((item) => item.service === service)?.shortLabel}`
          : `${DISTRICTS.find((item) => item.service === service)?.shortLabel} point of distribution`,
        cargoUnits: cargo,
        cargoPallets: cargo,
        returnLeg: 'returns empty to hub staging',
        scheduledStop: [visitsFuel ? FUEL_POINT_STOP_COPY : null, rerouteDisclosure].filter(Boolean).join(' · ') || undefined,
        wave: 'line-haul',
        dispatchRank,
      }
      plans.push(planFromManifest({
        stableId: `${service}-line-${index}`,
        manifest,
        path: routeFromHubToDepot(
          receivingService,
          detour,
          serviceIndex * 2 + index,
          visitsFuel,
        ),
        active: cargo > 0,
        index,
        serviceIndex,
        speed: (0.68 - dispatchRank * 0.012) * throughputMotion,
        broken: breakdown && index === 0,
      }))
    })

    lastLoads.forEach((cargo, index) => {
      const buildingIndex = lastMileDestinations[index % lastMileDestinations.length]
      const manifest: VehicleManifest = {
        id: `${service}-last-${index}`,
        service,
        fleet: serviceFleet(service),
        origin: `${DISTRICTS.find((item) => item.service === receivingService)?.shortLabel} point of distribution`,
        destination: serviceDestination(service, buildingIndex),
        cargoUnits: cargo,
        cargoPallets: cargo,
        returnLeg: status.reroutedFrom ? 'returns empty to assisting district depot' : 'returns empty to district depot',
        wave: 'last-mile',
        dispatchRank,
      }
      plans.push(planFromManifest({
        stableId: `${service}-last-${index}`,
        manifest,
        path: routeFromDepotToBuilding(receivingService, service, buildingIndex, index),
        active: cargo > 0 && (day.logistics !== undefined || status.damage !== 'rubble'),
        index,
        serviceIndex,
        speed: (service === 'healthcare' ? 0.78 : 0.62) * throughputMotion,
      }))
    })

    const level = day.services_end[serviceIndex]
    const lowAllocation = allocation <= [...day.allocation].sort((left, right) => left - right)[1]
    const maintenanceManifest: VehicleManifest = {
      id: `${service}-maintenance`,
      service,
      fleet: service === 'transport' ? 'road sweeper' : 'inspection van',
      origin: `${DISTRICTS.find((item) => item.service === service)?.shortLabel} point of distribution`,
      destination: `${DISTRICTS.find((item) => item.service === service)?.shortLabel} maintenance loop`,
      cargoUnits: 0,
      cargoPallets: 0,
      returnLeg: 'returns to district depot',
      wave: 'maintenance',
      dispatchRank: 5,
    }
    plans.push(planFromManifest({
      stableId: `${service}-maintenance`,
      manifest: maintenanceManifest,
      path: routeFromDepotToBuilding(service, service, 31, 0),
      active: level > 0.68 && lowAllocation,
      index: 30 + serviceIndex,
      serviceIndex,
      speed: 0.50,
    }))

    const reliefManifest: VehicleManifest = {
      ...maintenanceManifest,
      id: `${service}-relief`,
      fleet: 'relief truck',
      origin: 'central intake hub staging',
      destination: `${DISTRICTS.find((item) => item.service === service)?.shortLabel} roadside breakdown`,
      returnLeg: 'returns to hub staging after roadside assistance',
    }
    plans.push(planFromManifest({
      stableId: `${service}-relief`,
      manifest: reliefManifest,
      path: routeFromHubToDepot(service, detour, 15 + serviceIndex, false),
      active: breakdown,
      index: 40 + serviceIndex,
      serviceIndex,
      speed: 0.72,
      relief: true,
    }))
  })

  const mutualAidPairs = new Map<string, { from_service: Service; to_service: Service }>()
  result.candidate.trajectory.forEach((entry) => entry.logistics?.mutual_aid_transfers.forEach((transfer) => {
    mutualAidPairs.set(`${transfer.from_service}:${transfer.to_service}`, {
      from_service: transfer.from_service,
      to_service: transfer.to_service,
    })
  }))
  if (day.logistics) {
    ;[...mutualAidPairs.values()].forEach((pair, pairIndex) => {
      const transfer = day.logistics?.mutual_aid_transfers.find((event) => (
        event.from_service === pair.from_service && event.to_service === pair.to_service
      ))
      const transferUnits = transfer?.units ?? 0
      const receiverIndex = result.services.indexOf(pair.to_service)
      const donorIndex = result.services.indexOf(pair.from_service)
      const netMatches = Boolean(transfer)
        && receiverIndex >= 0
        && donorIndex >= 0
        && Math.abs((day.logistics?.mutual_aid_net[receiverIndex] ?? 0) - transferUnits) <= 1e-6
        && Math.abs((day.logistics?.mutual_aid_net[donorIndex] ?? 0) + transferUnits) <= 1e-6
      const maximumTransfer = Math.max(0, ...result.candidate.trajectory.flatMap((entry) => (
        entry.logistics?.mutual_aid_transfers
          .filter((event) => event.from_service === pair.from_service && event.to_service === pair.to_service)
          .map((event) => event.units) ?? []
      )))
      const transferThroughput = Math.min(
        day.logistics?.throughput_factor[donorIndex] ?? 0,
        day.logistics?.throughput_factor[receiverIndex] ?? 0,
      )
      const loads = activeLoads(
        netMatches ? transferUnits : 0,
        HEAVY_TRUCK_CAPACITY,
        Math.max(1, Math.ceil(maximumTransfer / HEAVY_TRUCK_CAPACITY)),
      )
      loads.forEach((cargo, index) => {
        const manifest: VehicleManifest = {
          id: `${pair.from_service}-${pair.to_service}-mutual-aid-${index}`,
          service: pair.to_service,
          fleet: 'mutual-aid line-haul truck',
          origin: `${DISTRICTS.find((item) => item.service === pair.from_service)?.shortLabel} point of distribution`,
          destination: `${DISTRICTS.find((item) => item.service === pair.to_service)?.shortLabel} point of distribution`,
          cargoUnits: cargo,
          cargoPallets: cargo,
          returnLeg: 'returns empty to donor depot',
          wave: 'mutual-aid',
          dispatchRank: result.services.length,
        }
        plans.push(planFromManifest({
          stableId: `${pair.from_service}-${pair.to_service}-mutual-aid-${index}`,
          manifest,
          path: compactPath([
            depotStagePosition(pair.from_service, index),
            ...[...CITY_ROUTE_WAYPOINTS[pair.from_service].direct].reverse().slice(1),
            ...CITY_ROUTE_WAYPOINTS[pair.to_service].direct.slice(1),
          ]),
          active: cargo > 0,
          index: 90 + pairIndex * 4 + index,
          serviceIndex: receiverIndex,
          speed: 0.60 * (0.45 + 0.55 * transferThroughput),
        }))
      })
    })
  }

  const currentShock = day.shock.type ? day.shock : null
  const emergencyService = currentShock ? strongestShockService(currentShock, result.services) : null
  const emergencyCount = currentShock ? Math.min(4, 1 + Math.round(currentShock.severity * 8)) : 0
  Array.from({ length: 4 }, (_, index) => {
    const service = emergencyService ?? result.services[index % result.services.length]
    const manifest: VehicleManifest = {
      id: `emergency-${index}`,
      service,
      fleet: service === 'healthcare' ? 'ambulance' : 'rapid assessment vehicle',
      origin: 'central intake hub staging',
      destination: `${DISTRICTS.find((item) => item.service === service)?.shortLabel} incident focus`,
      cargoUnits: 0,
      cargoPallets: 0,
      returnLeg: 'returns to incident staging after assessment',
      wave: 'emergency',
      dispatchRank: -1,
    }
    plans.push(planFromManifest({
      stableId: `emergency-${index}`,
      manifest,
      path: compactPath([
        ...hubStageToDispatch(10 + index),
        ...routeFromHubToDistrict(service).slice(1),
      ]),
      active: index < emergencyCount,
      index: 50 + index,
      speed: 0.90,
    }))
  })

  DISTRICTS.forEach((district, districtIndex) => {
    const serviceIndex = result.services.indexOf(district.service)
    const level = day.services_end[serviceIndex]
    const prior = previous?.services_end[serviceIndex] ?? level
    const impacted = currentShock ? currentShock.impact[serviceIndex] * currentShock.severity > 0.08 : false
    const returning = !currentShock && level > prior + 0.002
    Array.from({ length: 1 }, (_, index) => {
      const edgeX = district.center[0] < 0 ? -26 : 26
      const localPath: readonly WorldPoint[] = impacted
        ? compactPath([
            [district.center[0], 0.43, district.center[2]],
            [0, 0.43, district.center[2]],
            [0, 0.43, -2],
            [edgeX, 0.43, -2],
          ])
        : returning
          ? compactPath([
              [edgeX, 0.43, -2],
              [0, 0.43, -2],
              [0, 0.43, district.center[2]],
              [district.center[0], 0.43, district.center[2]],
            ])
          : [
              [district.center[0] - 7.7, 0.43, district.center[2]],
              [district.center[0] + 7.7, 0.43, district.center[2]],
            ]
      const manifest: VehicleManifest = {
        id: `${district.service}-civilian-${index}`,
        service: district.service,
        fleet: 'civilian car',
        origin: impacted ? district.label : 'district road loop',
        destination: impacted ? 'plate-edge outbound road' : returning ? district.label : 'district road loop',
        cargoUnits: 0,
        cargoPallets: 0,
        returnLeg: impacted ? 'return pulse waits for service recovery' : 'continues district loop',
        wave: 'civilian',
        dispatchRank: 6,
      }
      plans.push(planFromManifest({
        stableId: `${district.service}-civilian-${index}`,
        manifest,
        path: localPath,
        active: !quakeDebrisActive && index < Math.ceil(level),
        index: 60 + districtIndex + index,
        serviceIndex,
        speed: impacted ? 0.62 : 0.48 + level * 0.10,
      }))
    })
  })

  const housing = day.services_end[result.services.indexOf('housing')]
  const regularLoop = housing >= 0.65 && transport >= 0.65
  Array.from({ length: 3 }, (_, index) => {
    const manifest: VehicleManifest = {
      id: `commuter-${index}`,
      service: 'transport',
      fleet: index === 2 ? 'regular pickup bus' : 'transit bus',
      origin: 'Transit works',
      destination: index === 2 ? 'Residential quarter pickup loop' : 'Relay City commuter loop',
      cargoUnits: 0,
      cargoPallets: 0,
      returnLeg: 'completes the scheduled loop',
      wave: 'civilian',
      dispatchRank: 7,
    }
    plans.push(planFromManifest({
      stableId: `commuter-${index}`,
      manifest,
      path: compactPath([
        depotStagePosition('transport', index),
        [17, 0.43, 0.5],
        [17, 0.43, 7.5],
        [0, 0.43, 7.5],
        [0, 0.43, -11.5],
        [-17, 0.43, -11.5],
      ]),
      active: !quakeDebrisActive && (index < 2 ? transport >= 0.45 : regularLoop),
      index: 80 + index,
      serviceIndex: transportIndex,
      speed: 0.52,
    }))
  })

  return plans
}

export type VisibleVehicleRole = keyof typeof VISIBLE_VEHICLE_LIMITS

export function visibleVehicleRole(plan: Pick<VehiclePlan, 'stableId' | 'manifest' | 'relief'>): VisibleVehicleRole {
  if (plan.stableId.startsWith('commuter-')) return 'commuter'
  if (plan.manifest.wave === 'inbound') return 'inbound'
  if (plan.manifest.wave === 'line-haul') return 'lineHaul'
  if (plan.manifest.wave === 'last-mile') return 'lastMile'
  if (plan.manifest.wave === 'mutual-aid') return 'mutualAid'
  if (plan.manifest.wave === 'emergency') return 'emergency'
  if (plan.manifest.wave === 'civilian') return 'civilian'
  return 'support'
}

function proportionalStableIds(
  plansByDay: readonly VehiclePlan[][],
  role: 'lineHaul' | 'lastMile',
  services: readonly Service[],
  limit: number,
): string[] {
  const wave = role === 'lineHaul' ? 'line-haul' : 'last-mile'
  const groups = services.map((service) => {
    const candidates = [...new Set(plansByDay.flatMap((plans) => plans
      .filter((plan) => plan.manifest.wave === wave && plan.manifest.service === service && plan.active)
      .map((plan) => plan.stableId)))]
      .sort((left, right) => left.localeCompare(right, undefined, { numeric: true }))
    const totalUnits = plansByDay.reduce((total, plans) => total + plans
      .filter((plan) => plan.manifest.wave === wave && plan.manifest.service === service && plan.active)
      .reduce((dayTotal, plan) => dayTotal + plan.manifest.cargoUnits, 0), 0)
    return { service, candidates, totalUnits, seats: 0 }
  })

  for (let seat = 0; seat < limit; seat += 1) {
    const candidate = groups
      .filter((group) => group.seats < group.candidates.length && group.totalUnits > 1e-8)
      .sort((left, right) => (
        right.totalUnits / (right.seats + 1) - left.totalUnits / (left.seats + 1)
        || services.indexOf(left.service) - services.indexOf(right.service)
      ))[0]
    if (!candidate) break
    candidate.seats += 1
  }

  return groups.flatMap((group) => group.candidates.slice(0, group.seats))
}

function highestActivityStableIds(
  plansByDay: readonly VehiclePlan[][],
  role: VisibleVehicleRole,
  services: readonly Service[],
  limit: number,
): string[] {
  const scores = new Map<string, { activeDays: number; cargoUnits: number; serviceIndex: number }>()
  plansByDay.forEach((plans) => plans.forEach((plan) => {
    if (!plan.active || visibleVehicleRole(plan) !== role) return
    const existing = scores.get(plan.stableId) ?? {
      activeDays: 0,
      cargoUnits: 0,
      serviceIndex: plan.manifest.service === 'hub' ? -1 : services.indexOf(plan.manifest.service),
    }
    existing.activeDays += 1
    existing.cargoUnits += plan.manifest.cargoUnits
    scores.set(plan.stableId, existing)
  }))
  return [...scores.entries()]
    .sort(([leftId, left], [rightId, right]) => (
      right.cargoUnits - left.cargoUnits
      || right.activeDays - left.activeDays
      || left.serviceIndex - right.serviceIndex
      || leftId.localeCompare(rightId, undefined, { numeric: true })
    ))
    .slice(0, limit)
    .map(([stableId]) => stableId)
}

/**
 * Bounded deterministic scene sampling. Exact quantity-derived load equivalents stay
 * complete in the Toolbox; no selected plan is merged, inflated, or given a new mission.
 */
export function visibleVehicleStableIdsForResult(result: CompareResponse): string[] {
  const plansByDay = result.candidate.trajectory.map((_, dayIndex) => vehiclePlansForDay(result, dayIndex))
  const selected = [
    ...highestActivityStableIds(plansByDay, 'inbound', result.services, VISIBLE_VEHICLE_LIMITS.inbound),
    ...proportionalStableIds(plansByDay, 'lineHaul', result.services, VISIBLE_VEHICLE_LIMITS.lineHaul),
    ...proportionalStableIds(plansByDay, 'lastMile', result.services, VISIBLE_VEHICLE_LIMITS.lastMile),
    ...highestActivityStableIds(plansByDay, 'mutualAid', result.services, VISIBLE_VEHICLE_LIMITS.mutualAid),
    ...highestActivityStableIds(plansByDay, 'emergency', result.services, VISIBLE_VEHICLE_LIMITS.emergency),
    ...highestActivityStableIds(plansByDay, 'support', result.services, VISIBLE_VEHICLE_LIMITS.support),
    ...highestActivityStableIds(plansByDay, 'civilian', result.services, VISIBLE_VEHICLE_LIMITS.civilian),
    ...highestActivityStableIds(plansByDay, 'commuter', result.services, VISIBLE_VEHICLE_LIMITS.commuter),
  ]
  return [...new Set(selected)].slice(0, MAX_VISIBLE_ROAD_VEHICLES)
}

export function visibleVehiclePlansForDay(
  result: CompareResponse,
  dayIndex: number,
  stableIds: readonly string[] = visibleVehicleStableIdsForResult(result),
): VehiclePlan[] {
  const allowlist = new Set(stableIds)
  return vehiclePlansForDay(result, dayIndex).filter((plan) => allowlist.has(plan.stableId))
}

export function vehiclePlansForMode(
  plans: readonly VehiclePlan[],
  mode: VehicleFleetMode,
): VehiclePlan[] {
  // Assessment keeps ordinary missions rendered where they pulled over; only the
  // rapid-assessment wave advances until the response handoff.
  return [...plans]
}

export type ScheduledVehicleMission = {
  missionKey: string
  plan: VehiclePlan
  day: number
  dayIndex: number
  startOperationalDay: number
  endOperationalDay: number
  dispatchDelayOperationalDays: number
  initialProgress: number
  progressPerDay: number
}

export type VehicleMissionTimeline = {
  stableId: string
  dayCount: number
  dailyPlans: ReadonlyMap<number, VehiclePlan>
  missions: readonly ScheduledVehicleMission[]
  assessmentPauseDayIndices: readonly number[]
  pausesForAssessment: boolean
}

export type VehicleMissionSnapshot = {
  plan: VehiclePlan
  progress: number
  missionKey: string | null
  missionDay: number
  missionDayIndex: number
  activeMission: boolean
}

/** Weather belongs to the mission's returned engine day, never the day currently on screen. */
export function vehicleMissionWeatherMultiplier(result: CompareResponse, missionDayIndex: number): number {
  return result.candidate.trajectory[missionDayIndex]?.shock.type === 'weather' ? 0.58 : 1
}

/**
 * Maps the shared wall-clock cursor onto logistics time. Ordinary traffic pulls
 * over from shock impact through assessment, then resumes from the identical
 * position at response. Emergency assessment vehicles keep raw cursor time.
 */
export function vehicleOperationalAbsoluteDay(
  absoluteDay: number,
  assessmentPauseDayIndices: readonly number[],
): number {
  const boundedAbsoluteDay = Math.max(0, Number.isFinite(absoluteDay) ? absoluteDay : 0)
  return assessmentPauseDayIndices.reduce((operationalDay, dayIndex) => (
    operationalDay - clamp(
      boundedAbsoluteDay - dayIndex,
      0,
      SHOCK_RESPONSE_START_FRACTION,
    )
  ), boundedAbsoluteDay)
}

/**
 * Builds a deterministic, one-dispatch-per-returned-day queue for every stable vehicle slot.
 * The queue is immutable presentation data: mounting late, seeking, or replacing a compare
 * response cannot create another trip. A busy slot finishes its prior mission before taking
 * the next returned daily manifest.
 */
export function vehicleMissionTimelinesForResult(
  result: CompareResponse,
  mode: VehicleFleetMode = 'full',
): VehicleMissionTimeline[] {
  const visibleStableIds = visibleVehicleStableIdsForResult(result)
  const assessmentPauseDayIndices = result.candidate.trajectory
    .map((day, dayIndex) => (day.shock.type ? dayIndex : -1))
    .filter((dayIndex) => dayIndex >= 0)
  const timelines = new Map<string, {
    stableId: string
    dailyPlans: Map<number, VehiclePlan>
    missions: ScheduledVehicleMission[]
    pausesForAssessment: boolean
  }>()

  result.candidate.trajectory.forEach((day, dayIndex) => {
    const plans = vehiclePlansForMode(visibleVehiclePlansForDay(result, dayIndex, visibleStableIds), mode)
    const activeDispatchProgress = plans
      .filter((plan) => plan.active && plan.manifest.dispatchRank >= 0)
      .map((plan) => dispatchStartProgress(plan, result.seed, day.day))
    const leadingDispatchProgress = Math.max(0, ...activeDispatchProgress)
    plans.forEach((plan) => {
      const timeline = timelines.get(plan.stableId) ?? {
        stableId: plan.stableId,
        dailyPlans: new Map<number, VehiclePlan>(),
        missions: [],
        pausesForAssessment: plan.manifest.wave !== 'emergency',
      }
      timeline.dailyPlans.set(dayIndex, plan)
      timelines.set(plan.stableId, timeline)
      if (!plan.active) return

      const dispatchProgress = dispatchStartProgress(plan, result.seed, day.day)
      const progressPerDay = vehicleMissionProgressPerDay(
        plan,
        vehicleMissionWeatherMultiplier(result, dayIndex),
      )
      // The old renderer used dispatchProgress as the first visible cycle value.
      // That made a staged slot teleport into its route when the next queued
      // mission became active. Keep the same deterministic dispatch ordering,
      // but express the offset as a launch delay so every mission begins at the
      // physical staging point with zero geometric progress.
      const dispatchDelayOperationalDays = progressPerDay > 0 && plan.manifest.dispatchRank > 0
        ? Math.max(0, leadingDispatchProgress - dispatchProgress) / progressPerDay
        : 0
      const dayStartOperational = timeline.pausesForAssessment
        ? vehicleOperationalAbsoluteDay(dayIndex, assessmentPauseDayIndices)
        : dayIndex
      const preceding = timeline.missions[timeline.missions.length - 1]
      const startOperationalDay = Math.max(
        dayStartOperational + dispatchDelayOperationalDays,
        preceding?.endOperationalDay ?? dayStartOperational,
      )
      const endOperationalDay = progressPerDay > 0
        ? startOperationalDay + VEHICLE_MISSION_STAGE_PROGRESS / progressPerDay
        : Number.POSITIVE_INFINITY
      timeline.missions.push({
        missionKey: `${vehicleMissionDayKey(result, dayIndex)}:${plan.stableId}`,
        plan,
        day: day.day,
        dayIndex,
        startOperationalDay,
        endOperationalDay,
        dispatchDelayOperationalDays,
        initialProgress: 0,
        progressPerDay,
      })
    })
  })

  return [...timelines.values()].map((timeline) => ({
    ...timeline,
    dayCount: result.candidate.trajectory.length,
    assessmentPauseDayIndices,
  }))
}

/** Pure reconstruction of one vehicle slot at an arbitrary presentation cursor. */
export function vehicleMissionSnapshotAt(
  timeline: VehicleMissionTimeline,
  absoluteDay: number,
): VehicleMissionSnapshot {
  const boundedAbsoluteDay = clamp(
    Number.isFinite(absoluteDay) ? absoluteDay : 0,
    0,
    timeline.dayCount,
  )
  const operationalAbsoluteDay = timeline.pausesForAssessment
    ? vehicleOperationalAbsoluteDay(boundedAbsoluteDay, timeline.assessmentPauseDayIndices)
    : boundedAbsoluteDay
  const displayDayIndex = Math.min(
    timeline.dayCount - 1,
    Math.max(0, Math.floor(boundedAbsoluteDay)),
  )
  const mission = [...timeline.missions].reverse().find((candidate) => (
    operationalAbsoluteDay >= candidate.startOperationalDay
    && operationalAbsoluteDay < candidate.endOperationalDay
  ))

  if (mission) {
    return {
      plan: mission.plan,
      progress: advanceVehicleMissionProgress(
        mission.initialProgress,
        (operationalAbsoluteDay - mission.startOperationalDay) * mission.progressPerDay,
      ),
      missionKey: mission.missionKey,
      missionDay: mission.day,
      missionDayIndex: mission.dayIndex,
      activeMission: true,
    }
  }

  const fallbackPlan = timeline.dailyPlans.get(displayDayIndex)
    ?? [...timeline.dailyPlans.entries()]
      .sort(([left], [right]) => Math.abs(left - displayDayIndex) - Math.abs(right - displayDayIndex))[0]?.[1]
  if (!fallbackPlan) throw new Error(`Vehicle slot ${timeline.stableId} has no returned daily plan`)
  const completedToday = [...timeline.missions]
    .reverse()
    .find((candidate) => (
      candidate.dayIndex === displayDayIndex
      && operationalAbsoluteDay >= candidate.endOperationalDay
    ))

  return {
    plan: completedToday?.plan ?? fallbackPlan,
    progress: VEHICLE_MISSION_STAGE_PROGRESS,
    missionKey: completedToday?.missionKey ?? null,
    missionDay: completedToday?.day ?? 0,
    missionDayIndex: completedToday?.dayIndex ?? displayDayIndex,
    activeMission: false,
  }
}

export function vehicleAdvancesInMode(plan: VehiclePlan, mode: VehicleFleetMode): boolean {
  return mode === 'full' || plan.manifest.wave === 'emergency'
}

function samplePath(plan: VehiclePlan, progress: number, target: THREE.Vector3): THREE.Vector3 {
  if (plan.path.length <= 1) return target.set(...(plan.path[0] ?? [0, 0, 0]))
  let distance = clamp(progress) * plan.pathLength
  for (let index = 0; index < plan.segmentLengths.length; index += 1) {
    const length = plan.segmentLengths[index]
    if (distance <= length || index === plan.segmentLengths.length - 1) {
      const local = length <= 1e-8 ? 0 : distance / length
      const start = plan.path[index]
      const end = plan.path[index + 1]
      return target.set(
        THREE.MathUtils.lerp(start[0], end[0], local),
        THREE.MathUtils.lerp(start[1], end[1], local),
        THREE.MathUtils.lerp(start[2], end[2], local),
      )
    }
    distance -= length
  }
  return target.set(...plan.path[plan.path.length - 1])
}

function pathProgressAtWaypoint(plan: VehiclePlan, waypoint: WorldPoint): number {
  const waypointIndex = plan.path.findIndex((point) => (
    Math.hypot(point[0] - waypoint[0], point[2] - waypoint[2]) < 1e-6
  ))
  if (waypointIndex <= 0) return 0
  return plan.segmentLengths
    .slice(0, waypointIndex)
    .reduce((distance, length) => distance + length, 0) / plan.pathLength
}

/** Samples destination back to the mission-correct origin, entirely on the same road path. */
function sampleReturnToStage(plan: VehiclePlan, progress: number, target: THREE.Vector3): THREE.Vector3 {
  return samplePath(plan, 1 - clamp(progress), target)
}

function normalPositionForCycle(plan: VehiclePlan, progress: number, target: THREE.Vector3): THREE.Vector3 {
  const visitsFuel = plan.fuelStop !== null
  const state = vehicleCycleState(progress, visitsFuel)
  if (state === 'load') {
    return target.set(...plan.stagePosition)
  }
  if (state === 'fuel') {
    if (!plan.fuelStop) return target.set(...plan.path[0])
    const fuelPathProgress = pathProgressAtWaypoint(plan, plan.fuelStop)
    const travelProgress = clamp((progress - 0.10) / 0.045)
    return travelProgress < 1
      ? samplePath(plan, travelProgress * fuelPathProgress, target)
      : target.set(...plan.fuelStop)
  }
  if (visitsFuel && state === 'outbound') {
    const fuelPathProgress = pathProgressAtWaypoint(plan, plan.fuelStop!)
    const outboundProgress = clamp((progress - 0.21) / 0.27)
    return samplePath(plan, fuelPathProgress + outboundProgress * (1 - fuelPathProgress), target)
  }
  if (state === 'outbound') return samplePath(plan, (progress - 0.13) / 0.32, target)
  if (state === 'dock') return target.set(...plan.path[plan.path.length - 1])
  if (state === 'return') {
    const returnStart = visitsFuel ? 0.62 : 0.59
    return sampleReturnToStage(plan, (progress - returnStart) / (0.90 - returnStart), target)
  }
  return target.set(...plan.stagePosition)
}

function positionForCycle(plan: VehiclePlan, progress: number, target: THREE.Vector3): THREE.Vector3 {
  if (!plan.active) return target.set(...plan.stagePosition)
  if (!plan.broken || progress < 0.26) return normalPositionForCycle(plan, progress, target)

  // A deterministic breakdown waits in-lane for its truth-derived relief leg,
  // then retraces the same road to its mission-correct origin.
  const fuelProgress = plan.fuelStop ? pathProgressAtWaypoint(plan, plan.fuelStop) : 0
  const breakdownPathProgress = plan.fuelStop
    ? fuelProgress + clamp((0.26 - 0.21) / 0.27) * (1 - fuelProgress)
    : clamp((0.26 - 0.13) / 0.32)
  if (progress < 0.70) return samplePath(plan, breakdownPathProgress, target)
  const returnProgress = clamp((progress - 0.70) / 0.20)
  return samplePath(plan, breakdownPathProgress * (1 - returnProgress), target)
}

export function vehiclePositionForCycle(plan: VehiclePlan, progress: number): WorldPoint {
  const position = positionForCycle(plan, progress, new THREE.Vector3())
  return [position.x, position.y, position.z]
}

type VehiclePartScratch = {
  matrix: THREE.Matrix4
  position: THREE.Vector3
  offset: THREE.Vector3
  scale: THREE.Vector3
  quaternion: THREE.Quaternion
  localQuaternion: THREE.Quaternion
  euler: THREE.Euler
  color: THREE.Color
}

/** Writes a procedural accessory into a shared instanced mesh without allocating per frame. */
function writeVehiclePart(
  mesh: InstancedMesh,
  index: number,
  basePosition: THREE.Vector3,
  heading: THREE.Quaternion,
  visibility: number,
  offsetX: number,
  offsetY: number,
  offsetZ: number,
  width: number,
  height: number,
  length: number,
  colorValue: THREE.ColorRepresentation,
  scratch: VehiclePartScratch,
  rotationX = 0,
  rotationY = 0,
  rotationZ = 0,
): void {
  scratch.offset.set(offsetX, offsetY, offsetZ).applyQuaternion(heading)
  scratch.position.copy(basePosition).add(scratch.offset)
  scratch.scale.set(width * visibility, height * visibility, length * visibility)
  if (rotationX || rotationY || rotationZ) {
    scratch.euler.set(rotationX, rotationY, rotationZ)
    scratch.localQuaternion.setFromEuler(scratch.euler)
    scratch.quaternion.copy(heading).multiply(scratch.localQuaternion)
  } else {
    scratch.quaternion.copy(heading)
  }
  scratch.matrix.compose(scratch.position, scratch.quaternion, scratch.scale)
  mesh.setMatrixAt(index, scratch.matrix)
  mesh.setColorAt(index, scratch.color.set(colorValue))
}

export function VehicleFleet({
  result,
  dayIndex,
  mode = 'full',
  presentationAbsoluteDay,
  reducedMotion,
  onDockDwellChange,
  vehicleParts,
  castShadows,
}: {
  result: CompareResponse
  dayIndex: number
  mode?: VehicleFleetMode
  presentationAbsoluteDay: number
  reducedMotion: boolean
  onDockDwellChange?: (dwell: VehicleDockDwell) => void
  vehicleParts: RenderQualityProfile['vehicleParts']
  castShadows: boolean
}) {
  const missionTimelines = useMemo(
    () => vehicleMissionTimelinesForResult(result, mode),
    [mode, result],
  )
  const fleetSnapshots = useMemo(
    () => missionTimelines.map((timeline) => vehicleMissionSnapshotAt(timeline, presentationAbsoluteDay)),
    [missionTimelines, presentationAbsoluteDay],
  )
  const plans = useMemo(() => fleetSnapshots.map((snapshot) => snapshot.plan), [fleetSnapshots])
  const chassis = useRef<InstancedMesh>(null)
  const bodies = useRef<InstancedMesh>(null)
  const cabins = useRef<InstancedMesh>(null)
  const wheels = useRef<InstancedMesh>(null)
  const beacons = useRef<InstancedMesh>(null)
  const roofModules = useRef<InstancedMesh>(null)
  const detailModules = useRef<InstancedMesh>(null)
  const boomModules = useRef<InstancedMesh>(null)
  const bucketModules = useRef<InstancedMesh>(null)
  const palletModules = useRef<InstancedMesh>(null)
  const brushModules = useRef<InstancedMesh>(null)
  const tooltip = useRef<Group>(null)
  const positions = useRef<THREE.Vector3[]>([])
  const evidenceStates = useRef<Map<string, VehicleCycleState>>(new Map())
  const lastDockKey = useRef('')
  const [hovered, setHovered] = useState<number | null>(null)
  const [hoveredCycleState, setHoveredCycleState] = useState<VehicleCycleState>('stage')
  const matrix = useMemo(() => new THREE.Matrix4(), [])
  const position = useMemo(() => new THREE.Vector3(), [])
  const nextPosition = useMemo(() => new THREE.Vector3(), [])
  const partPosition = useMemo(() => new THREE.Vector3(), [])
  const partOffset = useMemo(() => new THREE.Vector3(), [])
  const quaternion = useMemo(() => new THREE.Quaternion(), [])
  const euler = useMemo(() => new THREE.Euler(), [])
  const scale = useMemo(() => new THREE.Vector3(), [])
  const color = useMemo(() => new THREE.Color(), [])
  const accessoryScratch = useMemo<VehiclePartScratch>(() => ({
    matrix: new THREE.Matrix4(),
    position: new THREE.Vector3(),
    offset: new THREE.Vector3(),
    scale: new THREE.Vector3(),
    quaternion: new THREE.Quaternion(),
    localQuaternion: new THREE.Quaternion(),
    euler: new THREE.Euler(),
    color: new THREE.Color(),
  }), [])
  useFrame(() => {
    const chassisMesh = chassis.current
    const bodyMesh = bodies.current
    const beaconMesh = beacons.current
    const roofMesh = roofModules.current
    const detailMesh = detailModules.current
    const boomMesh = boomModules.current
    const bucketMesh = bucketModules.current
    const palletMesh = palletModules.current
    const brushMesh = brushModules.current
    if (
      !chassisMesh
      || !bodyMesh
      || !beaconMesh
      || !roofMesh
      || !detailMesh
      || !boomMesh
      || !bucketMesh
      || !palletMesh
      || !brushMesh
    ) return
    if (positions.current.length !== plans.length) {
      positions.current = plans.map(() => new THREE.Vector3())
    }
    const evidenceHook = typeof window === 'undefined' ? undefined : window.__RELAY_EVIDENCE__
    const dwellingSnapshots: VehicleMissionSnapshot[] = []
    fleetSnapshots.forEach((snapshot, index) => {
      const plan = snapshot.plan
      // Mission position is reconstructed only from the returned run and the shared
      // absolute presentation cursor. React mount timing, seeks, and compare reruns
      // cannot advance, restart, or re-time a dispatch.
      const cycle = snapshot.progress
      const cycleState = vehicleCycleState(cycle, plan.fuelStop !== null)
      if (index === hovered && hoveredCycleState !== cycleState) setHoveredCycleState(cycleState)
      if (
        dwellingSnapshots.length === 0
        && plan.active
        && cycleState === 'dock'
        && (plan.manifest.wave === 'inbound' || plan.manifest.wave === 'line-haul' || plan.manifest.wave === 'mutual-aid')
      ) dwellingSnapshots.push(snapshot)
      positionForCycle(plan, cycle, position)
      positionForCycle(plan, Math.min(VEHICLE_MISSION_STAGE_PROGRESS, cycle + 0.004), nextPosition)
      positions.current[index].copy(position)
      if (evidenceHook && evidenceHook.vehicleId === plan.stableId && plan.active) {
        const evidenceResultId = result.result_id
        const evidenceDay = snapshot.missionDay || result.candidate.trajectory[dayIndex].day
        const evidenceKey = `${evidenceResultId}:${evidenceDay}:${plan.stableId}`
        if (evidenceStates.current.get(evidenceKey) !== cycleState) {
          evidenceStates.current.set(evidenceKey, cycleState)
          publishVehicleEvidenceTransition({
            stableId: plan.stableId,
            result_id: evidenceResultId,
            presentation_day: result.candidate.trajectory[dayIndex].day,
            day: evidenceDay,
            manifest: {
              id: `d${evidenceDay}-${plan.manifest.id}`,
              wave: plan.manifest.wave,
              fleet: plan.manifest.fleet,
              origin: plan.manifest.origin,
              destination: plan.manifest.destination,
              cargo_units: plan.manifest.cargoUnits,
              cargo_pallets: plan.manifest.cargoPallets,
              return_leg: plan.manifest.returnLeg,
              scheduled_stop: plan.manifest.scheduledStop ?? null,
            },
            cycle_state: cycleState,
            progress: Number(cycle.toFixed(6)),
            position: [position.x, position.y, position.z],
            timestamp_ms: performance.now(),
          }, evidenceHook)
        }
      }
      const heading = Math.atan2(nextPosition.x - position.x, nextPosition.z - position.z)
      euler.set(0, Number.isFinite(heading) ? heading : 0, 0)
      quaternion.setFromEuler(euler)

      const commuterIndex = plan.stableId.startsWith('commuter-')
        ? Number(plan.stableId.slice('commuter-'.length))
        : -1
      const commuterPulse = (Math.cos(presentationAbsoluteDay * 3.08 + result.seed * 0.0007) + 1) / 2
      const vehicleVisibility = commuterIndex >= 0 && plan.active
        ? (commuterIndex === 0 || commuterPulse > 0.22 + commuterIndex * 0.24 ? 1 : 0.015)
        : 1
      const silhouette = vehicleSilhouetteForFleet(plan.manifest.fleet)

      scale.set(0.82 * plan.scale[0] * vehicleVisibility, 0.18 * plan.scale[1] * vehicleVisibility, 1.18 * plan.scale[2] * vehicleVisibility)
      partPosition.copy(position)
      partPosition.y += 0.12
      matrix.compose(partPosition, quaternion, scale)
      chassisMesh.setMatrixAt(index, matrix)
      color.set(plan.broken ? '#77736b' : plan.color)
      chassisMesh.setColorAt(index, color)

      let bodyWidth = 0.72
      let bodyHeight = 0.54
      let bodyLength = 0.72
      let bodyY = 0.42
      let bodyZ = -0.18
      if (silhouette === 'ambulance') {
        bodyHeight = 0.68
        bodyLength = 0.82
        bodyY = 0.5
        bodyZ = -0.12
      } else if (silhouette === 'rapid-assessment') {
        bodyHeight = 0.48
        bodyLength = 0.68
        bodyY = 0.4
        bodyZ = -0.1
      } else if (silhouette === 'flatbed') {
        bodyHeight = 0.18
        bodyLength = 0.76
        bodyY = 0.27
      } else if (silhouette === 'refrigerated') {
        bodyWidth = 0.74
        bodyHeight = 0.86
        bodyLength = 0.82
        bodyY = 0.57
      } else if (silhouette === 'bucket-truck') {
        bodyHeight = 0.24
        bodyLength = 0.68
        bodyY = 0.29
        bodyZ = -0.2
      } else if (silhouette === 'bus') {
        bodyWidth = 0.76
        bodyHeight = 0.68
        bodyLength = 1.04
        bodyY = 0.5
        bodyZ = -0.04
      } else if (silhouette === 'road-sweeper') {
        bodyWidth = 0.68
        bodyHeight = 0.34
        bodyLength = 0.64
        bodyY = 0.34
        bodyZ = -0.12
      } else if (silhouette === 'loader') {
        bodyWidth = 0.66
        bodyHeight = 0.36
        bodyLength = 0.48
        bodyY = 0.38
        bodyZ = -0.08
      } else if (silhouette === 'dump-truck') {
        bodyHeight = 0.24
        bodyLength = 0.68
        bodyY = 0.3
        bodyZ = -0.2
      } else if (silhouette === 'rail') {
        bodyWidth = 0.78
        bodyHeight = 0.6
        bodyLength = 1.05
        bodyY = 0.48
        bodyZ = -0.08
      } else if (silhouette === 'car') {
        bodyWidth = 0.68
        bodyHeight = 0.32
        bodyLength = 0.56
        bodyY = 0.32
        bodyZ = -0.1
      } else if (silhouette === 'van') {
        bodyHeight = 0.52
        bodyLength = 0.72
        bodyY = 0.42
        bodyZ = -0.1
      }
      scale.set(
        bodyWidth * plan.scale[0] * vehicleVisibility,
        bodyHeight * plan.scale[1] * vehicleVisibility,
        bodyLength * plan.scale[2] * vehicleVisibility,
      )
      partOffset.set(0, bodyY * plan.scale[1], bodyZ * plan.scale[2]).applyQuaternion(quaternion)
      partPosition.copy(position).add(partOffset)
      matrix.compose(partPosition, quaternion, scale)
      bodyMesh.setMatrixAt(index, matrix)
      bodyMesh.setColorAt(index, color)

      if (cabins.current) {
        const cabinVisibility = silhouette === 'rail' ? 0.001 : vehicleVisibility
        const cabinHeight = silhouette === 'bus' ? 0.56 : silhouette === 'car' ? 0.36 : 0.48
        const cabinLength = silhouette === 'bus' ? 0.25 : silhouette === 'car' ? 0.34 : 0.38
        const cabinY = silhouette === 'bus' ? 0.47 : silhouette === 'car' ? 0.32 : 0.4
        const cabinZ = silhouette === 'bus' ? 0.62 : silhouette === 'car' ? 0.38 : 0.52
        scale.set(
          0.68 * plan.scale[0] * cabinVisibility,
          cabinHeight * plan.scale[1] * cabinVisibility,
          cabinLength * plan.scale[2] * cabinVisibility,
        )
        partOffset.set(0, cabinY * plan.scale[1], cabinZ * plan.scale[2]).applyQuaternion(quaternion)
        partPosition.copy(position).add(partOffset)
        matrix.compose(partPosition, quaternion, scale)
        cabins.current.setMatrixAt(index, matrix)
        cabins.current.setColorAt(index, color.set(plan.cabinColor))
      }

      if (wheels.current) {
        scale.set(0.78 * plan.scale[0] * vehicleVisibility, 0.13 * vehicleVisibility, 0.82 * plan.scale[2] * vehicleVisibility)
        partPosition.copy(position)
        partPosition.y += 0.04
        matrix.compose(partPosition, quaternion, scale)
        wheels.current.setMatrixAt(index, matrix)
      }

      // Start every accessory instance hidden; the typed cases below replace only
      // their own slots. This keeps one draw call per shared geometry while giving
      // each operational fleet a legible silhouette.
      const hidden = 0.001
      writeVehiclePart(roofMesh, index, position, quaternion, hidden, 0, 0.4, 0, 1, 1, 1, '#d8ddd7', accessoryScratch)
      writeVehiclePart(detailMesh, index, position, quaternion, hidden, 0, 0.4, 0, 1, 1, 1, '#46595b', accessoryScratch)
      writeVehiclePart(boomMesh, index, position, quaternion, hidden, 0, 0.4, 0, 1, 1, 1, '#a28752', accessoryScratch)
      writeVehiclePart(bucketMesh, index, position, quaternion, hidden, 0, 0.2, 0, 1, 1, 1, '#6c6a61', accessoryScratch)
      for (let palletIndex = 0; palletIndex < 3; palletIndex += 1) {
        writeVehiclePart(
          palletMesh,
          index * 3 + palletIndex,
          position,
          quaternion,
          hidden,
          0,
          0.3,
          0,
          1,
          1,
          1,
          '#96744d',
          accessoryScratch,
        )
      }
      for (let brushIndex = 0; brushIndex < 2; brushIndex += 1) {
        writeVehiclePart(
          brushMesh,
          index * 2 + brushIndex,
          position,
          quaternion,
          hidden,
          0,
          0.1,
          0,
          1,
          1,
          1,
          '#414744',
          accessoryScratch,
        )
      }

      if (silhouette === 'ambulance' || silhouette === 'rapid-assessment') {
        const rapid = silhouette === 'rapid-assessment'
        writeVehiclePart(
          roofMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          (rapid ? 0.67 : 0.88) * plan.scale[1],
          -0.08 * plan.scale[2],
          (rapid ? 0.56 : 0.6) * plan.scale[0],
          0.08 * plan.scale[1],
          (rapid ? 0.54 : 0.62) * plan.scale[2],
          rapid ? '#5d6967' : '#e2e3dd',
          accessoryScratch,
        )
        writeVehiclePart(
          detailMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          (rapid ? 0.76 : 0.98) * plan.scale[1],
          0.18 * plan.scale[2],
          0.56 * plan.scale[0],
          0.08 * plan.scale[1],
          0.12 * plan.scale[2],
          '#c99b56',
          accessoryScratch,
        )
      } else if (silhouette === 'flatbed') {
        writeVehiclePart(
          detailMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          0.36 * plan.scale[1],
          -0.58 * plan.scale[2],
          0.7 * plan.scale[0],
          0.16 * plan.scale[1],
          0.08 * plan.scale[2],
          '#59615d',
          accessoryScratch,
        )
        const cargoVisibility = plan.active && vehicleCarriesCargo(plan.manifest, cycleState)
          ? vehicleVisibility
          : hidden
        const palletOffsets = [
          [-0.2, -0.28],
          [0.2, -0.28],
          [0, -0.58],
        ] as const
        palletOffsets.forEach(([offsetX, offsetZ], palletIndex) => {
          writeVehiclePart(
            palletMesh,
            index * 3 + palletIndex,
            position,
            quaternion,
            cargoVisibility,
            offsetX * plan.scale[0],
            0.48 * plan.scale[1],
            offsetZ * plan.scale[2],
            0.28 * plan.scale[0],
            (0.28 + palletIndex * 0.04) * plan.scale[1],
            0.28 * plan.scale[2],
            palletIndex === 1 ? '#aa8659' : '#927149',
            accessoryScratch,
          )
        })
      } else if (silhouette === 'refrigerated') {
        writeVehiclePart(
          roofMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          1.04 * plan.scale[1],
          -0.22 * plan.scale[2],
          0.48 * plan.scale[0],
          0.08 * plan.scale[1],
          0.46 * plan.scale[2],
          '#eef0eb',
          accessoryScratch,
        )
        writeVehiclePart(
          detailMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          0.68 * plan.scale[1],
          0.27 * plan.scale[2],
          0.58 * plan.scale[0],
          0.38 * plan.scale[1],
          0.1 * plan.scale[2],
          '#73817f',
          accessoryScratch,
        )
      } else if (silhouette === 'bucket-truck') {
        writeVehiclePart(
          detailMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          0.33 * plan.scale[1],
          -0.2 * plan.scale[2],
          0.82 * plan.scale[0],
          0.1 * plan.scale[1],
          0.13 * plan.scale[2],
          '#595f59',
          accessoryScratch,
        )
        writeVehiclePart(
          boomMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          0.75 * plan.scale[1],
          -0.24 * plan.scale[2],
          0.11 * plan.scale[0],
          0.11 * plan.scale[1],
          0.92 * plan.scale[2],
          '#b18f51',
          accessoryScratch,
          0.5,
        )
        writeVehiclePart(
          bucketMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          1.08 * plan.scale[1],
          -0.62 * plan.scale[2],
          0.34 * plan.scale[0],
          0.26 * plan.scale[1],
          0.3 * plan.scale[2],
          '#a48650',
          accessoryScratch,
        )
      } else if (silhouette === 'bus') {
        writeVehiclePart(
          roofMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          0.87 * plan.scale[1],
          -0.08 * plan.scale[2],
          0.58 * plan.scale[0],
          0.07 * plan.scale[1],
          0.86 * plan.scale[2],
          '#deddd5',
          accessoryScratch,
        )
        writeVehiclePart(
          detailMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          0.59 * plan.scale[1],
          -0.08 * plan.scale[2],
          0.78 * plan.scale[0],
          0.22 * plan.scale[1],
          1.02 * plan.scale[2],
          '#3f5659',
          accessoryScratch,
        )
      } else if (silhouette === 'road-sweeper') {
        writeVehiclePart(
          roofMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          0.67 * plan.scale[1],
          0.18 * plan.scale[2],
          0.42 * plan.scale[0],
          0.08 * plan.scale[1],
          0.2 * plan.scale[2],
          '#b88b4b',
          accessoryScratch,
        )
        writeVehiclePart(
          detailMesh,
          index,
          position,
          quaternion,
          vehicleVisibility,
          0,
          0.42 * plan.scale[1],
          -0.3 * plan.scale[2],
          0.64 * plan.scale[0],
          0.42 * plan.scale[1],
          0.42 * plan.scale[2],
          '#65726d',
          accessoryScratch,
        )
        for (let brushIndex = 0; brushIndex < 2; brushIndex += 1) {
          writeVehiclePart(
            brushMesh,
            index * 2 + brushIndex,
            position,
            quaternion,
            vehicleVisibility,
            (brushIndex === 0 ? -0.43 : 0.43) * plan.scale[0],
            0.08,
            -0.16 * plan.scale[2],
            0.24 * plan.scale[0],
            0.08,
            0.24 * plan.scale[0],
            '#414744',
            accessoryScratch,
          )
        }
      } else if (silhouette === 'loader') {
        writeVehiclePart(roofMesh, index, position, quaternion, vehicleVisibility, 0, 0.72 * plan.scale[1], -0.2 * plan.scale[2], 0.52 * plan.scale[0], 0.08, 0.38 * plan.scale[2], '#d5cbb4', accessoryScratch)
        writeVehiclePart(detailMesh, index, position, quaternion, vehicleVisibility, 0, 0.42 * plan.scale[1], -0.38 * plan.scale[2], 0.68 * plan.scale[0], 0.4 * plan.scale[1], 0.34 * plan.scale[2], '#a18149', accessoryScratch)
        writeVehiclePart(boomMesh, index, position, quaternion, vehicleVisibility, 0, 0.35 * plan.scale[1], 0.36 * plan.scale[2], 0.12 * plan.scale[0], 0.12, 0.72 * plan.scale[2], '#a9874c', accessoryScratch, -0.2)
        writeVehiclePart(bucketMesh, index, position, quaternion, vehicleVisibility, 0, 0.16, 0.8 * plan.scale[2], 0.94 * plan.scale[0], 0.25, 0.34 * plan.scale[2], '#686860', accessoryScratch)
      } else if (silhouette === 'dump-truck') {
        writeVehiclePart(roofMesh, index, position, quaternion, vehicleVisibility, 0, 0.68 * plan.scale[1], 0.42 * plan.scale[2], 0.5 * plan.scale[0], 0.07, 0.28 * plan.scale[2], '#d5cbb4', accessoryScratch)
        writeVehiclePart(detailMesh, index, position, quaternion, vehicleVisibility, 0, 0.56 * plan.scale[1], -0.25 * plan.scale[2], 0.74 * plan.scale[0], 0.44 * plan.scale[1], 0.74 * plan.scale[2], '#927345', accessoryScratch, -0.18)
        writeVehiclePart(bucketMesh, index, position, quaternion, vehicleVisibility, 0, 0.48 * plan.scale[1], -0.62 * plan.scale[2], 0.78 * plan.scale[0], 0.34 * plan.scale[1], 0.08 * plan.scale[2], '#686860', accessoryScratch)
      } else if (silhouette === 'rail') {
        writeVehiclePart(roofMesh, index, position, quaternion, vehicleVisibility, 0, 0.84 * plan.scale[1], -0.08 * plan.scale[2], 0.66 * plan.scale[0], 0.08, 0.78 * plan.scale[2], '#a99d84', accessoryScratch)
        writeVehiclePart(detailMesh, index, position, quaternion, vehicleVisibility, 0, 0.5 * plan.scale[1], -0.08 * plan.scale[2], 0.82 * plan.scale[0], 0.16 * plan.scale[1], 1.08 * plan.scale[2], '#565e59', accessoryScratch)
      } else if (silhouette === 'van' && plan.manifest.fleet.includes('inspection')) {
        writeVehiclePart(roofMesh, index, position, quaternion, vehicleVisibility, 0, 0.7 * plan.scale[1], -0.06 * plan.scale[2], 0.56 * plan.scale[0], 0.07, 0.48 * plan.scale[2], '#5f6c68', accessoryScratch)
        writeVehiclePart(detailMesh, index, position, quaternion, vehicleVisibility, 0, 0.79 * plan.scale[1], 0.16 * plan.scale[2], 0.48 * plan.scale[0], 0.07, 0.1 * plan.scale[2], '#c49751', accessoryScratch)
      }

      const beaconPulse = plan.emergency && plan.active
        ? (reducedMotion ? 0.75 : 0.72 + Math.sin(cycle * Math.PI * 24) * 0.18)
        : 0.001
      scale.setScalar(beaconPulse * vehicleVisibility)
      partOffset.set(0, 0.78 * plan.scale[1], 0.28 * plan.scale[2]).applyQuaternion(quaternion)
      partPosition.copy(position).add(partOffset)
      matrix.compose(partPosition, quaternion, scale)
      beaconMesh.setMatrixAt(index, matrix)
    })
    const dwellingSnapshot = dwellingSnapshots[0]
    const dwellingPlan = dwellingSnapshot?.plan
    const dockKey = dwellingSnapshot
      ? `${dwellingSnapshot.missionKey ?? 'staged'}:dock`
      : 'none'
    if (lastDockKey.current !== dockKey) {
      lastDockKey.current = dockKey
      onDockDwellChange?.(dwellingPlan ? {
        id: dockKey,
        active: true,
        strength: clamp(dwellingPlan.manifest.cargoUnits / HEAVY_TRUCK_CAPACITY),
      } : { id: `${result.result_id}:${result.candidate.trajectory[dayIndex].day}:dock-clear`, active: false, strength: 0 })
    }
    for (const mesh of [
      chassis.current,
      bodies.current,
      cabins.current,
      wheels.current,
      beacons.current,
      roofModules.current,
      detailModules.current,
      boomModules.current,
      bucketModules.current,
      palletModules.current,
      brushModules.current,
    ]) {
      if (!mesh) continue
      mesh.instanceMatrix.needsUpdate = true
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    }
    if (hovered !== null && tooltip.current && positions.current[hovered]) {
      tooltip.current.position.copy(positions.current[hovered])
      tooltip.current.position.y += 1.45
    }
  })

  const hoverHandlers = {
    onPointerMove: (event: { stopPropagation: () => void; instanceId?: number }) => {
      event.stopPropagation()
      if (typeof event.instanceId === 'number') {
        setHovered(event.instanceId)
        const snapshot = fleetSnapshots[event.instanceId]
        if (snapshot) {
          setHoveredCycleState(vehicleCycleState(
            snapshot.progress,
            snapshot.plan.fuelStop !== null,
          ))
        }
      }
      document.body.style.cursor = 'help'
    },
    onPointerOut: () => {
      setHovered(null)
      document.body.style.cursor = ''
    },
  }
  const selectedPlan = hovered === null ? null : plans[hovered]

  return (
    <group name="purposeful-vehicle-economy">
      <instancedMesh ref={chassis} args={[undefined, undefined, plans.length]} castShadow={castShadows} {...hoverHandlers}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#ffffff" roughness={0.86} />
      </instancedMesh>
      <instancedMesh ref={bodies} args={[undefined, undefined, plans.length]} castShadow={castShadows}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#ffffff" roughness={0.82} />
      </instancedMesh>
      {vehicleParts >= 4 ? (
        <instancedMesh ref={cabins} args={[undefined, undefined, plans.length]} castShadow={castShadows}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="#ffffff" roughness={0.78} />
        </instancedMesh>
      ) : null}
      {vehicleParts >= 5 ? (
        <instancedMesh ref={wheels} args={[undefined, undefined, plans.length]} castShadow={castShadows}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="#303533" roughness={0.96} />
        </instancedMesh>
      ) : null}
      <instancedMesh ref={beacons} args={[undefined, undefined, plans.length]} castShadow={castShadows}>
        <cylinderGeometry args={[0.08, 0.09, 0.11, 10]} />
        <meshStandardMaterial color="#d0a057" emissive="#6e4e25" emissiveIntensity={0.25} roughness={0.72} />
      </instancedMesh>
      <instancedMesh ref={roofModules} args={[undefined, undefined, plans.length]} castShadow={castShadows}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#ffffff" roughness={0.82} />
      </instancedMesh>
      <instancedMesh ref={detailModules} args={[undefined, undefined, plans.length]} castShadow={castShadows}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#ffffff" roughness={0.78} />
      </instancedMesh>
      <instancedMesh ref={boomModules} args={[undefined, undefined, plans.length]} castShadow={castShadows}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#ffffff" roughness={0.74} />
      </instancedMesh>
      <instancedMesh ref={bucketModules} args={[undefined, undefined, plans.length]} castShadow={castShadows}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#ffffff" roughness={0.9} />
      </instancedMesh>
      <instancedMesh ref={palletModules} args={[undefined, undefined, plans.length * 3]} castShadow={castShadows}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#ffffff" roughness={0.96} />
      </instancedMesh>
      <instancedMesh ref={brushModules} args={[undefined, undefined, plans.length * 2]} castShadow={castShadows}>
        <cylinderGeometry args={[0.5, 0.5, 0.3, 10]} />
        <meshStandardMaterial color="#ffffff" roughness={1} />
      </instancedMesh>
      {selectedPlan ? (
        <group ref={tooltip}>
          <Html center className="scene-inspector-anchor" zIndexRange={[40, 0]}>
            <div className="scene-entity-card vehicle-card">
              <b>{selectedPlan.manifest.fleet}</b>
              <span>{selectedPlan.manifest.origin} → {selectedPlan.manifest.destination}</span>
              <small>{vehicleCargoCopy(selectedPlan.manifest, hoveredCycleState)}</small>
              {selectedPlan.manifest.scheduledStop ? <small>{selectedPlan.manifest.scheduledStop}</small> : null}
              <em>{selectedPlan.manifest.returnLeg}</em>
              {selectedPlan.broken ? <strong>Roadside hold · relief leg dispatched</strong> : null}
            </div>
          </Html>
        </group>
      ) : null}
    </group>
  )
}
