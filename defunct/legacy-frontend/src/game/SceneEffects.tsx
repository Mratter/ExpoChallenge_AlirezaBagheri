import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { Group, InstancedMesh, Mesh } from 'three'
import type { CompareResponse, DayResult, Service } from '../types'
import { DISTRICT_BUILDING_OFFSETS, DISTRICTS, isBuildingRebuilding } from './model'
import { ARCHETYPE_SPECS } from './DenseCityBuildings'
import { CITY_BUILDING_PLACEMENTS } from './worldLayout'
import { hasCurrentDayRepairWork } from './realism'
import { presentationMotionPhase, repairPresentationMotionTime } from './presentationMotion'

export { hasCurrentDayRepairWork } from './realism'

export type WorldPosition = readonly [number, number, number]

export type ConvoyPlan = {
  service: Service
  allocation: number
  vehicleCount: number
  speed: number
  color: string
  start: WorldPosition
  control: WorldPosition
  end: WorldPosition
}

export type RepairPlan = {
  service: Service
  allocation: number
  realizedGain: number
  vehicleCount: number
  color: string
  position: WorldPosition
  buildingIndex: number
  tall: boolean
}

const CANONICAL_SERVICES: readonly Service[] = [
  'transport',
  'housing',
  'food',
  'healthcare',
  'public_services',
]

export type RepairPresentationCursor = {
  dayIndex: number
  progress: number
}

/**
 * Before RESPONSE, keep the prior returned day's work site at its exact end
 * pose. A first-day incident has no earlier crew to reveal.
 */
export function repairPresentationCursor(
  dayIndex: number,
  dayProgress: number,
  responseEnabled: boolean,
): RepairPresentationCursor | null {
  if (responseEnabled) {
    return { dayIndex, progress: clamp(dayProgress, 0, 1) }
  }
  return dayIndex > 0 ? { dayIndex: dayIndex - 1, progress: 1 } : null
}

export function repairActivityMotionTime(
  dayIndex: number,
  sharedRepairTime: number,
  responseEnabled: boolean,
): number {
  return responseEnabled ? sharedRepairTime : repairPresentationMotionTime(dayIndex)
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value))
}

function serviceOffset(services: readonly Service[], service: Service): number {
  const index = services.indexOf(service)
  return index >= 0 ? index : CANONICAL_SERVICES.indexOf(service)
}

/**
 * Converts the exact daily allocation into a monotonic vehicle plan. The denser fleet
 * keeps exact allocation share legible while the shared motion rate makes each trip a
 * sustained journey across the expanded plate.
 */
export function convoyPlansForDay(
  day: DayResult,
  services: readonly Service[] = CANONICAL_SERVICES,
  inactiveServices: readonly Service[] = [],
): ConvoyPlan[] {
  const budget = Math.max(day.available_budget, 1)
  return DISTRICTS.flatMap((district, districtIndex) => {
    if (inactiveServices.includes(district.service)) return []
    const index = serviceOffset(services, district.service)
    const allocation = Math.max(0, day.allocation[index] ?? 0)
    if (allocation <= 0.001) return []

    const share = clamp(allocation / budget, 0, 1)
    const vehicleCount = clamp(Math.ceil(share * 20), 2, 10)
    const speed = 0.105 + share * 0.48
    const target = new THREE.Vector2(district.center[0], district.center[2])
    const direction = target.clone().normalize()
    const perpendicular = new THREE.Vector2(-direction.y, direction.x)
    const lane = (districtIndex % 2 === 0 ? 1 : -1) * 0.28
    const start2 = direction.clone().multiplyScalar(2.05).addScaledVector(perpendicular, lane)
    const end2 = target.clone().addScaledVector(direction, -1.05).addScaledVector(perpendicular, lane)
    const control2 = start2.clone().lerp(end2, 0.52)
      .addScaledVector(perpendicular, (districtIndex - 2) * 0.18)

    return [{
      service: district.service,
      allocation,
      vehicleCount,
      speed,
      color: district.accent,
      start: [start2.x, 0.43, start2.y],
      control: [control2.x, 0.43, control2.y],
      end: [end2.x, 0.43, end2.y],
    }]
  })
}

/**
 * Returns only districts with a positive realized trajectory change. On the first day,
 * the same test is made against `services_after_shock`, so activity remains data-derived.
 */
export function repairPlansForDay(
  day: DayResult,
  previous: DayResult | undefined,
  services: readonly Service[] = CANONICAL_SERVICES,
  inactiveServices: readonly Service[] = [],
): RepairPlan[] {
  return DISTRICTS.flatMap((district) => {
    if (inactiveServices.includes(district.service)) return []
    const index = serviceOffset(services, district.service)
    const priorLevel = previous?.services_end[index] ?? day.services_after_shock[index]
    const realizedGain = (day.services_end[index] ?? priorLevel) - priorLevel
    if (!hasCurrentDayRepairWork(day, previous, district.service, services)) return []

    const allocation = Math.max(0, day.allocation[index] ?? 0)
    const rebuildingIndices = DISTRICT_BUILDING_OFFSETS
      .map((_, buildingIndex) => buildingIndex)
      .filter((buildingIndex) => isBuildingRebuilding(day, previous, district.service, buildingIndex))
    if (rebuildingIndices.length === 0) return []
    return rebuildingIndices.map((buildingIndex, cohortIndex) => {
      const [offsetX, offsetZ] = DISTRICT_BUILDING_OFFSETS[buildingIndex]
      // Cranes/lifts remain at every truth-derived site, but one parked repair
      // truck per improving service is enough to communicate the finite crew.
      const vehicleCount = cohortIndex === 0 ? 1 : 0
      return {
        service: district.service,
        allocation,
        realizedGain,
        vehicleCount,
        color: district.accent,
        position: [
          district.center[0] + offsetX,
          0.3,
          district.center[2] + offsetZ,
        ],
        buildingIndex,
        tall: ARCHETYPE_SPECS[
          CITY_BUILDING_PLACEMENTS.find((building) => (
            building.service === district.service && building.buildingIndex === buildingIndex
          ))?.archetype ?? 'rowhouse'
        ].size[1] >= 3,
      }
    })
  })
}

function quadraticPoint(
  start: WorldPosition,
  control: WorldPosition,
  end: WorldPosition,
  progress: number,
  target: THREE.Vector3,
): THREE.Vector3 {
  const inverse = 1 - progress
  return target.set(
    inverse * inverse * start[0] + 2 * inverse * progress * control[0] + progress * progress * end[0],
    inverse * inverse * start[1] + 2 * inverse * progress * control[1] + progress * progress * end[1],
    inverse * inverse * start[2] + 2 * inverse * progress * control[2] + progress * progress * end[2],
  )
}

function quadraticTangent(
  start: WorldPosition,
  control: WorldPosition,
  end: WorldPosition,
  progress: number,
  target: THREE.Vector3,
): THREE.Vector3 {
  return target.set(
    2 * (1 - progress) * (control[0] - start[0]) + 2 * progress * (end[0] - control[0]),
    0,
    2 * (1 - progress) * (control[2] - start[2]) + 2 * progress * (end[2] - control[2]),
  )
}

function TruckModel({ color, repair = false }: { color: string; repair?: boolean }) {
  return (
    <group scale={repair ? 0.43 : 0.48}>
      <mesh position={[0, 0.25, -0.18]} castShadow>
        <boxGeometry args={[1.08, 0.42, 0.82]} />
        <meshStandardMaterial color={color} roughness={0.78} />
      </mesh>
      <mesh position={[0, 0.38, 0.42]} castShadow>
        <boxGeometry args={[1.0, 0.62, 0.52]} />
        <meshStandardMaterial color={repair ? '#c59a48' : '#d7d1c2'} roughness={0.8} />
      </mesh>
      <mesh position={[0, 0.47, 0.692]}>
        <boxGeometry args={[0.68, 0.25, 0.025]} />
        <meshStandardMaterial color="#3c4b4d" roughness={0.58} />
      </mesh>
      {[-0.43, 0.43].flatMap((x) => [-0.34, 0.42].map((z) => (
        <mesh key={`${x}-${z}`} position={[x, 0.04, z]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.2, 0.2, 0.14, 12]} />
          <meshStandardMaterial color="#333735" roughness={0.94} />
        </mesh>
      )))}
      <mesh position={[0, 0.5, -0.18]} castShadow>
        <cylinderGeometry args={[0.12, 0.13, 0.1, 12]} />
        <meshStandardMaterial color={color} roughness={0.76} />
      </mesh>
    </group>
  )
}

function ConvoyVehicle({ route, index, day, reducedMotion, presentationTime }: {
  route: ConvoyPlan
  index: number
  day: number
  reducedMotion: boolean
  presentationTime: number
}) {
  const vehicle = useRef<Group>(null)
  const phaseOffset = (index / route.vehicleCount + (day * 0.173 + index * 0.071)) % 1
  const point = useMemo(() => new THREE.Vector3(), [])
  const tangent = useMemo(() => new THREE.Vector3(), [])

  useFrame(() => {
    if (!vehicle.current) return
    const progress = reducedMotion
      ? phaseOffset
      : presentationMotionPhase(presentationTime, route.speed, phaseOffset)
    quadraticPoint(route.start, route.control, route.end, progress, point)
    quadraticTangent(route.start, route.control, route.end, progress, tangent)
    point.y += Math.sin(progress * Math.PI) * 0.025
    vehicle.current.position.copy(point)
    vehicle.current.rotation.y = Math.atan2(tangent.x, tangent.z)
  })

  return (
    <group ref={vehicle}>
      <TruckModel color={route.color} />
    </group>
  )
}

export type AllocationConvoysProps = {
  /** The current candidate trajectory day; allocations are read directly from it. */
  day: DayResult
  /** Use `result.services` so array offsets remain tied to the response schema. */
  services?: readonly Service[]
  inactiveServices?: readonly Service[]
  reducedMotion?: boolean
  /** Shared deterministic cursor-derived time, in restrained motion seconds. */
  presentationTime?: number
}

/** Deterministic outbound traffic from the intake hub, proportional to exact daily allocations. */
export function AllocationConvoys({
  day,
  services = CANONICAL_SERVICES,
  inactiveServices = [],
  reducedMotion = false,
  presentationTime = 0,
}: AllocationConvoysProps) {
  const routes = useMemo(
    () => convoyPlansForDay(day, services, inactiveServices),
    [day, inactiveServices, services],
  )
  return (
    <group name="allocation-convoys">
      {routes.flatMap((route) => Array.from({ length: route.vehicleCount }, (_, index) => (
        <ConvoyVehicle
          key={`${route.service}-${index}`}
          route={route}
          index={index}
          day={day.day}
          reducedMotion={reducedMotion}
          presentationTime={presentationTime}
        />
      )))}
    </group>
  )
}

function RepairTruck({ plan, index }: {
  plan: RepairPlan
  index: number
}) {
  const side = index % 2 ? -1 : 1
  return (
    <group
      position={[
        plan.position[0] + side * (0.85 + index * 0.18),
        plan.position[1] + 0.1,
        plan.position[2] + 0.72 + Math.floor(index / 2) * 0.42,
      ]}
      rotation={[0, side > 0 ? -0.42 : 0.42, 0]}
    >
      <TruckModel color={plan.color} repair />
    </group>
  )
}

function RepairCrane({ plan, day, reducedMotion, presentationTime }: {
  plan: RepairPlan
  day: number
  reducedMotion: boolean
  presentationTime: number
}) {
  const boom = useRef<Group>(null)
  const hook = useRef<Mesh>(null)
  useFrame(() => {
    const phaseTime = reducedMotion ? day * 0.23 : presentationTime + day * 0.23
    if (boom.current) boom.current.rotation.y = Math.sin(phaseTime * 0.42) * 0.28
    if (hook.current) hook.current.position.y = -0.78 + Math.sin(phaseTime * 0.74) * 0.22
  })
  return (
    <group position={plan.position} scale={0.72}>
      <mesh position={[0, 1.45, 0]} castShadow>
        <boxGeometry args={[0.14, 2.9, 0.14]} />
        <meshStandardMaterial color="#b9873f" roughness={0.79} />
      </mesh>
      {[0.58, 1.18, 1.78, 2.38].map((height, index) => (
        <group key={height} position={[0, height, 0]} rotation={[0, index % 2 ? Math.PI / 4 : -Math.PI / 4, 0]}>
          <mesh><boxGeometry args={[0.7, 0.055, 0.055]} /><meshStandardMaterial color="#d0a451" roughness={0.78} /></mesh>
        </group>
      ))}
      <group ref={boom} position={[0, 2.82, 0]}>
        <mesh position={[0, 0, 0.82]} castShadow>
          <boxGeometry args={[0.13, 0.13, 1.78]} />
          <meshStandardMaterial color={plan.color} roughness={0.75} />
        </mesh>
        <mesh position={[0, -0.45, 1.62]} castShadow>
          <boxGeometry args={[0.035, 0.92, 0.035]} />
          <meshStandardMaterial color="#55544f" roughness={0.92} />
        </mesh>
        <mesh ref={hook} position={[0, -0.78, 1.62]} castShadow>
          <boxGeometry args={[0.2, 0.16, 0.2]} />
          <meshStandardMaterial color="#55544f" roughness={0.92} />
        </mesh>
      </group>
    </group>
  )
}

function RepairLift({ plan, day, reducedMotion, presentationTime }: {
  plan: RepairPlan
  day: number
  reducedMotion: boolean
  presentationTime: number
}) {
  const basket = useRef<Group>(null)
  useFrame(() => {
    if (!basket.current) return
    const pulse = reducedMotion ? 0.5 : (Math.sin(presentationTime * 0.34 + day) + 1) / 2
    basket.current.position.y = 0.95 + pulse * 0.72
  })
  return (
    <group position={plan.position} scale={0.72}>
      <mesh position={[0, 0.26, 0]} castShadow><boxGeometry args={[0.92, 0.5, 1.18]} /><meshStandardMaterial color="#656b65" roughness={0.88} /></mesh>
      <mesh position={[0.18, 0.94, 0]} rotation={[0.18, 0, -0.28]} castShadow><boxGeometry args={[0.12, 1.55, 0.12]} /><meshStandardMaterial color={plan.color} roughness={0.78} /></mesh>
      <group ref={basket} position={[-0.06, 1.3, 0]}>
        <mesh><boxGeometry args={[0.62, 0.16, 0.5]} /><meshStandardMaterial color={plan.color} roughness={0.8} /></mesh>
      </group>
    </group>
  )
}

export type RepairActivityProps = {
  /** The current candidate day and optional immediately preceding candidate day. */
  day: DayResult
  previous?: DayResult
  /** Use `result.services` to bind trajectory offsets to service names. */
  services?: readonly Service[]
  inactiveServices?: readonly Service[]
  reducedMotion?: boolean
  presentationProgress?: number
  /** Shared deterministic cursor-derived time, in restrained motion seconds. */
  presentationTime?: number
  /** Cursor-derived crossfade used for the brief day-boundary crew handoff. */
  visibility?: number
}

/** Repair trucks and cranes appear only where the real candidate trajectory improved. */
export function RepairActivity({
  day,
  previous,
  services = CANONICAL_SERVICES,
  inactiveServices = [],
  reducedMotion = false,
  presentationProgress = 1,
  presentationTime = 0,
  visibility = 1,
}: RepairActivityProps) {
  const plans = useMemo(
    () => repairPlansForDay(day, previous, services, inactiveServices),
    [day, inactiveServices, previous, services],
  )
  return (
    <group name="trajectory-derived-repairs">
      {plans.map((plan) => {
        const serviceOffset = services.indexOf(plan.service)
        const activation = 0.04 + (((plan.buildingIndex * 17 + serviceOffset * 11) % 37) / 37) * 0.5
        const planVisibility = Math.max(0, Math.min(1, (presentationProgress - activation) / 0.14))
          * Math.max(0, Math.min(1, visibility))
        return (
        <group key={`${plan.service}-${plan.buildingIndex}`} scale={[planVisibility, planVisibility, planVisibility]}>
          {plan.tall
            ? <RepairCrane plan={plan} day={day.day} reducedMotion={reducedMotion} presentationTime={presentationTime} />
            : <RepairLift plan={plan} day={day.day} reducedMotion={reducedMotion} presentationTime={presentationTime} />}
          {Array.from({ length: plan.vehicleCount }, (_, index) => (
            <RepairTruck
              key={index}
              plan={plan}
              index={index}
            />
          ))}
        </group>
        )
      })}
    </group>
  )
}

export type SceneEffectsProps = {
  /** Completed compare response currently driving the city and its playback index. */
  result: CompareResponse
  dayIndex: number
  inactiveServices?: readonly Service[]
  /** Previous returned-day city-condition exclusions for a frozen prior crew. */
  previousInactiveServices?: readonly Service[]
  /** False during TELEGRAPH, IMPACT, and transient ASSESSMENT presentation. */
  responseEnabled?: boolean
  reducedMotion?: boolean
  presentationProgress?: number
  /** Shared deterministic cursor-derived time, in restrained motion seconds. */
  presentationTime?: number
  /** Shared repair-only time with the day-boundary idle slices removed. */
  repairPresentationTime?: number
  /** Quintic handoff from the prior returned day's sites to the active day's sites. */
  shiftChangeBlend?: number
}

/** Concise CityScene integration: all persistent activity is derived from candidate data. */
export function SceneEffects({
  result,
  dayIndex,
  inactiveServices = [],
  previousInactiveServices = [],
  responseEnabled = true,
  reducedMotion = false,
  presentationProgress = 1,
  presentationTime = 0,
  repairPresentationTime = presentationTime,
  shiftChangeBlend = 1,
}: SceneEffectsProps) {
  const repairCursor = repairPresentationCursor(dayIndex, presentationProgress, responseEnabled)
  const activityMotionTime = repairActivityMotionTime(dayIndex, repairPresentationTime, responseEnabled)
  if (!repairCursor) return null
  const day = result.candidate.trajectory[repairCursor.dayIndex]
  const previous = result.candidate.trajectory[repairCursor.dayIndex - 1]
  const beforePrevious = result.candidate.trajectory[repairCursor.dayIndex - 2]
  if (!day) return null
  if (!responseEnabled) {
    return (
      <RepairActivity
        day={day}
        previous={previous}
        services={result.services}
        inactiveServices={previousInactiveServices}
        reducedMotion={reducedMotion}
        presentationProgress={repairCursor.progress}
        presentationTime={activityMotionTime}
        visibility={1}
      />
    )
  }
  return (
    <>
      {previous && shiftChangeBlend < 0.999 ? (
        <RepairActivity
          day={previous}
          previous={beforePrevious}
          services={result.services}
          inactiveServices={previousInactiveServices}
          reducedMotion={reducedMotion}
          presentationProgress={1}
          presentationTime={repairPresentationTime}
          visibility={1 - shiftChangeBlend}
        />
      ) : null}
      <RepairActivity
        day={day}
        previous={previous}
        services={result.services}
        inactiveServices={inactiveServices}
        reducedMotion={reducedMotion}
        presentationProgress={presentationProgress}
        presentationTime={repairPresentationTime}
        visibility={shiftChangeBlend}
      />
    </>
  )
}
