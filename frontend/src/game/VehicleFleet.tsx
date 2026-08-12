import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { Group, InstancedMesh } from 'three'
import type { CompareResponse, Service } from '../types'
import { DISTRICTS } from './model'
import {
  HEAVY_TRUCK_CAPACITY,
  LAST_MILE_CAPACITY,
  depotStatusesForDay,
  deterministicUnit,
  latestIncident,
  splitExactCargo,
  strongestShockService,
  type VehicleManifest,
} from './realism'
import {
  CITY_BUILDING_PLACEMENTS,
  CITY_DEPOTS,
  CITY_HUB,
  CITY_ROUTE_WAYPOINTS,
  type WorldPoint,
} from './worldLayout'
import type { RenderQualityProfile } from './renderQuality'

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
}

export type VehicleCycleState = 'load' | 'outbound' | 'dock' | 'return' | 'stage'

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

export function vehicleCycleState(progress: number): VehicleCycleState {
  const normalized = ((progress % 1) + 1) % 1
  if (normalized < 0.13) return 'load'
  if (normalized < 0.45) return 'outbound'
  if (normalized < 0.59) return 'dock'
  if (normalized < 0.90) return 'return'
  return 'stage'
}

function serviceFleet(service: Service): string {
  if (service === 'healthcare') return 'ambulance'
  if (service === 'housing') return 'brick flatbed'
  if (service === 'food') return 'refrigerated truck'
  if (service === 'public_services') return 'bucket truck'
  return 'recovery bus'
}

function scaleForFleet(fleet: string): readonly [number, number, number] {
  if (fleet.includes('bus')) return [0.72, 0.58, 1.25]
  if (fleet.includes('ambulance')) return [0.62, 0.62, 0.9]
  if (fleet.includes('flatbed')) return [0.72, 0.48, 1.08]
  if (fleet.includes('refrigerated')) return [0.72, 0.72, 1.08]
  if (fleet.includes('bucket')) return [0.68, 0.62, 1]
  if (fleet.includes('rail')) return [0.82, 0.64, 1.38]
  if (fleet.includes('car')) return [0.48, 0.38, 0.68]
  return [0.68, 0.6, 1]
}

function stagePosition(index: number, serviceIndex = 0): WorldPoint {
  return [
    CITY_HUB.staging[0] - 0.9 + (index % 6) * 0.7,
    0.34,
    CITY_HUB.staging[2] + Math.floor(index / 6) * 0.62 + serviceIndex * 0.08,
  ]
}

function depotFor(service: Service): (typeof CITY_DEPOTS)[number] {
  return CITY_DEPOTS.find((depot) => depot.service === service)!
}

function pathToBuilding(origin: WorldPoint, service: Service, index: number): readonly WorldPoint[] {
  const buildings = CITY_BUILDING_PLACEMENTS.filter((building) => building.service === service)
  const target = buildings[(index * 7 + 3) % buildings.length].position
  const district = DISTRICTS.find((item) => item.service === service)!
  return [
    origin,
    [district.center[0], 0.42, district.center[2]] as WorldPoint,
    [target[0], 0.42, target[2]] as WorldPoint,
  ]
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
    stagePosition: stagePosition(index, serviceIndex),
    speed,
    scale: scaleForFleet(manifest.fleet),
    color,
    cabinColor,
    emergency: manifest.wave === 'emergency',
    broken,
    relief,
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
  const incident = latestIncident(result, dayIndex, 2)
  const detour = transport < 0.42 || (incident?.type === 'aftershock' && incident.ageDays <= 1)
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
        ? [CITY_HUB.railIngress, [0, 0.42, -9] as WorldPoint, CITY_HUB.position]
        : [CITY_HUB.roadIngress, [-13, 0.42, -2] as WorldPoint, CITY_HUB.position],
      active: cargo > 0,
      index,
      speed: rail ? 0.34 : 0.30,
    }))
  })

  const allocationOrder = result.services
    .map((service, index) => ({ service, allocation: day.allocation[index] }))
    .sort((left, right) => right.allocation - left.allocation)

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
    const throughputMotion = day.logistics ? 0.45 + 0.55 * status.throughputSignal : 1
    const receivingService = day.logistics ? service : status.reroutedFrom ?? service
    const receivingDepot = depotFor(receivingService)
    const route = CITY_ROUTE_WAYPOINTS[receivingService]
    const linePath = detour ? route.detour : route.direct
    const breakdown = transport < 0.35
      && deterministicUnit(result.seed, day.day, 260 + serviceIndex) < 0.22

    lineLoads.forEach((cargo, index) => {
      const manifest: VehicleManifest = {
        id: `${service}-line-${index}`,
        service,
        fleet: 'line-haul truck',
        origin: 'central intake hub',
        destination: !day.logistics && status.reroutedFrom
          ? `${DISTRICTS.find((item) => item.service === receivingService)?.shortLabel} point of distribution · mutual aid for ${DISTRICTS.find((item) => item.service === service)?.shortLabel}`
          : `${DISTRICTS.find((item) => item.service === service)?.shortLabel} point of distribution`,
        cargoUnits: cargo,
        cargoPallets: cargo,
        returnLeg: (day.day + index + serviceIndex) % 7 === 0 ? 'fuel check, then hub staging' : 'returns empty to hub staging',
        wave: 'line-haul',
        dispatchRank,
      }
      plans.push(planFromManifest({
        stableId: `${service}-line-${index}`,
        manifest,
        path: linePath,
        active: cargo > 0,
        index,
        serviceIndex,
        speed: (0.27 - dispatchRank * 0.008) * throughputMotion,
        broken: breakdown && index === 0,
      }))
    })

    lastLoads.forEach((cargo, index) => {
      const manifest: VehicleManifest = {
        id: `${service}-last-${index}`,
        service,
        fleet: serviceFleet(service),
        origin: `${DISTRICTS.find((item) => item.service === receivingService)?.shortLabel} point of distribution`,
        destination: `${DISTRICTS.find((item) => item.service === service)?.shortLabel} recovery site ${(index * 7 + 3) % 36 + 1}`,
        cargoUnits: cargo,
        cargoPallets: cargo,
        returnLeg: 'returns empty to district depot',
        wave: 'last-mile',
        dispatchRank,
      }
      plans.push(planFromManifest({
        stableId: `${service}-last-${index}`,
        manifest,
        path: pathToBuilding(receivingDepot.position, service, index),
        active: cargo > 0 && (day.logistics !== undefined || status.damage !== 'rubble'),
        index,
        serviceIndex,
        speed: (service === 'healthcare' ? 0.34 : 0.25) * throughputMotion,
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
      path: pathToBuilding(depotFor(service).position, service, 31),
      active: level > 0.68 && lowAllocation,
      index: 30 + serviceIndex,
      serviceIndex,
      speed: 0.18,
    }))

    if (breakdown) {
      const reliefManifest: VehicleManifest = {
        ...maintenanceManifest,
        id: `${service}-relief`,
        fleet: 'relief truck',
        origin: 'central intake hub staging',
        destination: `${DISTRICTS.find((item) => item.service === service)?.shortLabel} delayed line-haul`,
        returnLeg: 'returns after completing the interrupted leg',
      }
      plans.push(planFromManifest({
        stableId: `${service}-relief`,
        manifest: reliefManifest,
        path: linePath,
        active: true,
        index: 40 + serviceIndex,
        serviceIndex,
        speed: 0.31,
        relief: true,
      }))
    }
  })

  if (day.logistics) {
    day.logistics.mutual_aid_transfers.forEach((transfer) => {
      const receiverIndex = result.services.indexOf(transfer.to_service)
      const donorIndex = result.services.indexOf(transfer.from_service)
      const netMatches = receiverIndex >= 0
        && donorIndex >= 0
        && Math.abs((day.logistics?.mutual_aid_net[receiverIndex] ?? 0) - transfer.units) <= 1e-6
        && Math.abs((day.logistics?.mutual_aid_net[donorIndex] ?? 0) + transfer.units) <= 1e-6
      if (!netMatches) return
      const maximumTransfer = Math.max(transfer.units, ...result.candidate.trajectory.flatMap((entry) => (
        entry.logistics?.mutual_aid_transfers
          .filter((event) => event.from_service === transfer.from_service && event.to_service === transfer.to_service)
          .map((event) => event.units) ?? []
      )))
      const transferThroughput = Math.min(
        day.logistics?.throughput_factor[donorIndex] ?? 0,
        day.logistics?.throughput_factor[receiverIndex] ?? 0,
      )
      const loads = activeLoads(
        transfer.units,
        HEAVY_TRUCK_CAPACITY,
        Math.max(1, Math.ceil(maximumTransfer / HEAVY_TRUCK_CAPACITY)),
      )
      loads.forEach((cargo, index) => {
        const manifest: VehicleManifest = {
          id: `${transfer.from_service}-${transfer.to_service}-mutual-aid-${index}`,
          service: transfer.to_service,
          fleet: 'mutual-aid line-haul truck',
          origin: `${DISTRICTS.find((item) => item.service === transfer.from_service)?.shortLabel} point of distribution`,
          destination: `${DISTRICTS.find((item) => item.service === transfer.to_service)?.shortLabel} point of distribution`,
          cargoUnits: cargo,
          cargoPallets: cargo,
          returnLeg: 'returns empty to donor depot',
          wave: 'mutual-aid',
          dispatchRank: result.services.length,
        }
        plans.push(planFromManifest({
          stableId: `${transfer.from_service}-${transfer.to_service}-mutual-aid-${index}`,
          manifest,
          path: [
            depotFor(transfer.from_service).position,
            [0, 0.42, 0] as WorldPoint,
            depotFor(transfer.to_service).position,
          ],
          active: cargo > 0,
          index: 90 + index,
          serviceIndex: receiverIndex,
          speed: 0.22 * (0.45 + 0.55 * transferThroughput),
        }))
      })
    })
  }

  const currentShock = day.shock.type ? day.shock : null
  const emergencyService = currentShock ? strongestShockService(currentShock, result.services) : null
  const emergencyCount = currentShock ? Math.min(4, 1 + Math.round(currentShock.severity * 8)) : 0
  Array.from({ length: 4 }, (_, index) => {
    const service = emergencyService ?? result.services[index % result.services.length]
    const route = CITY_ROUTE_WAYPOINTS[service]
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
      path: route.direct,
      active: index < emergencyCount,
      index: 50 + index,
      speed: 0.48,
    }))
  })

  DISTRICTS.forEach((district, districtIndex) => {
    const serviceIndex = result.services.indexOf(district.service)
    const level = day.services_end[serviceIndex]
    const prior = previous?.services_end[serviceIndex] ?? level
    const impacted = currentShock ? currentShock.impact[serviceIndex] * currentShock.severity > 0.08 : false
    const returning = !currentShock && level > prior + 0.002
    Array.from({ length: 2 }, (_, index) => {
      const edgeX = district.center[0] < 0 ? -26 : 26
      const localPath: readonly WorldPoint[] = impacted
        ? [[district.center[0], 0.33, district.center[2]], [edgeX, 0.33, -2]]
        : returning
          ? [[edgeX, 0.33, -2], [district.center[0], 0.33, district.center[2]]]
          : [
              [district.center[0] - 5, 0.33, district.center[2]],
              [district.center[0] + 5, 0.33, district.center[2]],
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
        active: index < Math.ceil(level * 2),
        index: 60 + districtIndex * 2 + index,
        serviceIndex,
        speed: impacted ? 0.27 : 0.16 + level * 0.06,
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
      destination: index === 2 ? 'Residential quarter pickup loop' : 'city commuter loop',
      cargoUnits: 0,
      cargoPallets: 0,
      returnLeg: 'completes the scheduled loop',
      wave: 'civilian',
      dispatchRank: 7,
    }
    plans.push(planFromManifest({
      stableId: `commuter-${index}`,
      manifest,
      path: [depotFor('transport').position, [17, 0.36, -11.5], [-17, 0.36, -11.5], depotFor('transport').position],
      active: index < 2 ? transport >= 0.45 : regularLoop,
      index: 80 + index,
      serviceIndex: transportIndex,
      speed: 0.18,
    }))
  })

  return plans
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

function positionForCycle(plan: VehiclePlan, progress: number, target: THREE.Vector3): THREE.Vector3 {
  if (!plan.active) return target.set(...plan.stagePosition)
  const state = vehicleCycleState(progress)
  if (plan.broken && progress >= 0.26 && progress < 0.9) {
    samplePath(plan, 0.42, target)
    target.x += 0.38
    return target
  }
  if (state === 'load') return target.set(...plan.path[0])
  if (state === 'outbound') return samplePath(plan, (progress - 0.13) / 0.32, target)
  if (state === 'dock') return target.set(...plan.path[plan.path.length - 1])
  if (state === 'return') return samplePath(plan, 1 - (progress - 0.59) / 0.31, target)
  return target.set(...plan.stagePosition)
}

function tooltipCargo(manifest: VehicleManifest): string {
  if (manifest.cargoUnits <= 0) return manifest.wave === 'emergency' ? 'Allocation-backed damage assessment' : 'No supply cargo'
  return `${manifest.cargoPallets.toFixed(1)} pallets = ${manifest.cargoUnits.toFixed(1)} supply units`
}

export function VehicleFleet({
  result,
  dayIndex,
  motionRate,
  reducedMotion,
  impactActive,
  vehicleParts,
  castShadows,
}: {
  result: CompareResponse
  dayIndex: number
  motionRate: number
  reducedMotion: boolean
  impactActive: boolean
  vehicleParts: RenderQualityProfile['vehicleParts']
  castShadows: boolean
}) {
  const plans = useMemo(() => vehiclePlansForDay(result, dayIndex), [dayIndex, result])
  const chassis = useRef<InstancedMesh>(null)
  const bodies = useRef<InstancedMesh>(null)
  const cabins = useRef<InstancedMesh>(null)
  const wheels = useRef<InstancedMesh>(null)
  const beacons = useRef<InstancedMesh>(null)
  const tooltip = useRef<Group>(null)
  const progress = useRef<number[]>([])
  const positions = useRef<THREE.Vector3[]>([])
  const [hovered, setHovered] = useState<number | null>(null)
  const matrix = useMemo(() => new THREE.Matrix4(), [])
  const position = useMemo(() => new THREE.Vector3(), [])
  const nextPosition = useMemo(() => new THREE.Vector3(), [])
  const partPosition = useMemo(() => new THREE.Vector3(), [])
  const partOffset = useMemo(() => new THREE.Vector3(), [])
  const quaternion = useMemo(() => new THREE.Quaternion(), [])
  const euler = useMemo(() => new THREE.Euler(), [])
  const scale = useMemo(() => new THREE.Vector3(), [])
  const color = useMemo(() => new THREE.Color(), [])
  const weatherSlow = result.candidate.trajectory[dayIndex].shock.type === 'weather' ? 0.58 : 1

  useFrame(({ clock }, delta) => {
    const chassisMesh = chassis.current
    const bodyMesh = bodies.current
    const beaconMesh = beacons.current
    if (!chassisMesh || !bodyMesh || !beaconMesh) return
    if (progress.current.length !== plans.length) {
      progress.current = plans.map((plan, index) => (
        (index * 0.071 + plan.manifest.dispatchRank * 0.043 + result.seed * 0.000001) % 1
      ))
      positions.current = plans.map(() => new THREE.Vector3())
    }
    plans.forEach((plan, index) => {
      if (!impactActive && plan.active) {
        const accessibilityRate = reducedMotion ? 0.28 : 1
        progress.current[index] = (progress.current[index] + Math.min(delta, 0.05) * motionRate * plan.speed * weatherSlow * accessibilityRate) % 1
      }
      const cycle = progress.current[index]
      positionForCycle(plan, cycle, position)
      positionForCycle(plan, (cycle + 0.004) % 1, nextPosition)
      if (impactActive && plan.active) position.x += 0.22
      positions.current[index].copy(position)
      const heading = Math.atan2(nextPosition.x - position.x, nextPosition.z - position.z)
      euler.set(0, Number.isFinite(heading) ? heading : 0, 0)
      quaternion.setFromEuler(euler)

      const commuterIndex = plan.stableId.startsWith('commuter-')
        ? Number(plan.stableId.slice('commuter-'.length))
        : -1
      const commuterPulse = (Math.cos(clock.elapsedTime * 0.44 + result.seed * 0.0007) + 1) / 2
      const vehicleVisibility = commuterIndex >= 0 && plan.active
        ? (commuterIndex === 0 || commuterPulse > 0.22 + commuterIndex * 0.24 ? 1 : 0.015)
        : 1

      scale.set(0.82 * plan.scale[0] * vehicleVisibility, 0.18 * plan.scale[1] * vehicleVisibility, 1.18 * plan.scale[2] * vehicleVisibility)
      partPosition.copy(position)
      partPosition.y += 0.12
      matrix.compose(partPosition, quaternion, scale)
      chassisMesh.setMatrixAt(index, matrix)
      color.set(plan.broken ? '#77736b' : plan.color)
      chassisMesh.setColorAt(index, color)

      scale.set(0.72 * plan.scale[0] * vehicleVisibility, 0.54 * plan.scale[1] * vehicleVisibility, 0.72 * plan.scale[2] * vehicleVisibility)
      partOffset.set(0, 0.42 * plan.scale[1], -0.18 * plan.scale[2]).applyQuaternion(quaternion)
      partPosition.copy(position).add(partOffset)
      matrix.compose(partPosition, quaternion, scale)
      bodyMesh.setMatrixAt(index, matrix)
      bodyMesh.setColorAt(index, color)

      if (cabins.current) {
        scale.set(0.68 * plan.scale[0] * vehicleVisibility, 0.48 * plan.scale[1] * vehicleVisibility, 0.38 * plan.scale[2] * vehicleVisibility)
        partOffset.set(0, 0.4 * plan.scale[1], 0.52 * plan.scale[2]).applyQuaternion(quaternion)
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

      const beaconPulse = plan.emergency && plan.active
        ? (reducedMotion ? 0.75 : 0.72 + Math.sin(cycle * Math.PI * 24) * 0.18)
        : 0.001
      scale.setScalar(beaconPulse * vehicleVisibility)
      partOffset.set(0, 0.78 * plan.scale[1], 0.28 * plan.scale[2]).applyQuaternion(quaternion)
      partPosition.copy(position).add(partOffset)
      matrix.compose(partPosition, quaternion, scale)
      beaconMesh.setMatrixAt(index, matrix)
    })
    for (const mesh of [chassis.current, bodies.current, cabins.current, wheels.current, beacons.current]) {
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
      if (typeof event.instanceId === 'number') setHovered(event.instanceId)
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
      {selectedPlan ? (
        <group ref={tooltip}>
          <Html center className="scene-inspector-anchor" zIndexRange={[40, 0]}>
            <div className="scene-entity-card vehicle-card">
              <b>{selectedPlan.manifest.fleet}</b>
              <span>{selectedPlan.manifest.origin} → {selectedPlan.manifest.destination}</span>
              <small>{tooltipCargo(selectedPlan.manifest)}</small>
              <em>{selectedPlan.manifest.returnLeg}</em>
              {selectedPlan.broken ? <strong>Roadside hold · relief leg dispatched</strong> : null}
            </div>
          </Html>
        </group>
      ) : null}
    </group>
  )
}
