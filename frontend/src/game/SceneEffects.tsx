import { useFrame } from '@react-three/fiber'
import { createContext, useContext, useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import type { ReactNode, RefObject } from 'react'
import * as THREE from 'three'
import type { Group, InstancedMesh, Mesh, MeshBasicMaterial } from 'three'
import type { CompareResponse, DayResult, Service, ShockType } from '../types'
import { DISTRICT_BUILDING_OFFSETS, DISTRICTS, isBuildingRebuilding } from './model'
import { ARCHETYPE_SPECS } from './DenseCityBuildings'
import { CITY_BUILDING_PLACEMENTS } from './worldLayout'
import { CONTINUOUS_MOTION_BASE_RATE } from './pacing'

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

/** A unique `id` restarts the transient even if type and position are unchanged. */
export type SceneImpact = {
  id: string | number
  type: ShockType
  severity: number
  position: WorldPosition
}

const SceneMotionContext = createContext(CONTINUOUS_MOTION_BASE_RATE)

export function SceneMotionProvider({
  motionRate,
  children,
}: {
  motionRate: number
  children: ReactNode
}) {
  return (
    <SceneMotionContext.Provider value={Math.max(0, motionRate)}>
      {children}
    </SceneMotionContext.Provider>
  )
}

/** One shared gate keeps continuous city motion proportional to playback and pause state. */
export function sceneMotionStep(delta: number, reducedMotion: boolean, motionRate = 1): number {
  return reducedMotion ? 0 : Math.min(delta, 0.075) * Math.max(0, motionRate)
}

const CANONICAL_SERVICES: readonly Service[] = [
  'transport',
  'housing',
  'food',
  'healthcare',
  'public_services',
]

const IMPACT_COLORS: Record<ShockType, { wave: string; brick: string; smoke: string }> = {
  aftershock: { wave: '#d6aa82', brick: '#806a59', smoke: '#716d66' },
  supply: { wave: '#d3b06d', brick: '#9b7843', smoke: '#777168' },
  epidemic: { wave: '#9fb8aa', brick: '#718073', smoke: '#69736c' },
  utility: { wave: '#9ec4ce', brick: '#61757b', smoke: '#626e70' },
  weather: { wave: '#9eb4c4', brick: '#6b7780', smoke: '#626a70' },
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
  return DISTRICTS.flatMap((district, districtIndex) => {
    if (inactiveServices.includes(district.service)) return []
    const index = serviceOffset(services, district.service)
    const priorLevel = previous?.services_end[index] ?? day.services_after_shock[index]
    const realizedGain = (day.services_end[index] ?? priorLevel) - priorLevel
    if (realizedGain <= 0.002) return []

    const allocation = Math.max(0, day.allocation[index] ?? 0)
    const totalVehicleCount = clamp(Math.ceil(realizedGain / 0.01), 2, 8)
    const rebuildingIndices = DISTRICT_BUILDING_OFFSETS
      .map((_, buildingIndex) => buildingIndex)
      .filter((buildingIndex) => isBuildingRebuilding(day, previous, district.service, buildingIndex))
    if (rebuildingIndices.length === 0) return []
    return rebuildingIndices.map((buildingIndex, cohortIndex) => {
      const [offsetX, offsetZ] = DISTRICT_BUILDING_OFFSETS[buildingIndex]
      const vehicleCount = Math.max(
        1,
        Math.floor(totalVehicleCount / rebuildingIndices.length)
          + (cohortIndex < totalVehicleCount % rebuildingIndices.length ? 1 : 0),
      )
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

function ConvoyVehicle({ route, index, day, reducedMotion, motionRate }: {
  route: ConvoyPlan
  index: number
  day: number
  reducedMotion: boolean
  motionRate: number
}) {
  const vehicle = useRef<Group>(null)
  const progress = useRef((index / route.vehicleCount + (day * 0.173 + index * 0.071)) % 1)
  const point = useMemo(() => new THREE.Vector3(), [])
  const tangent = useMemo(() => new THREE.Vector3(), [])

  useFrame((_, delta) => {
    if (!vehicle.current) return
    progress.current = (progress.current + sceneMotionStep(delta, reducedMotion, motionRate) * route.speed) % 1
    quadraticPoint(route.start, route.control, route.end, progress.current, point)
    quadraticTangent(route.start, route.control, route.end, progress.current, tangent)
    point.y += Math.sin(progress.current * Math.PI) * 0.025
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
  motionRate?: number
}

/** Deterministic outbound traffic from the intake hub, proportional to exact daily allocations. */
export function AllocationConvoys({
  day,
  services = CANONICAL_SERVICES,
  inactiveServices = [],
  reducedMotion = false,
  motionRate = 1,
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
          motionRate={motionRate}
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

function RepairCrane({ plan, day, reducedMotion, motionRate }: {
  plan: RepairPlan
  day: number
  reducedMotion: boolean
  motionRate: number
}) {
  const boom = useRef<Group>(null)
  const hook = useRef<Mesh>(null)
  const elapsed = useRef(day * 0.23)
  useFrame((_, delta) => {
    elapsed.current += sceneMotionStep(delta, reducedMotion, motionRate)
    if (boom.current) boom.current.rotation.y = Math.sin(elapsed.current * 0.42) * 0.28
    if (hook.current) hook.current.position.y = -0.78 + Math.sin(elapsed.current * 0.74) * 0.22
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

function RepairLift({ plan, day, reducedMotion, motionRate }: {
  plan: RepairPlan
  day: number
  reducedMotion: boolean
  motionRate: number
}) {
  const basket = useRef<Group>(null)
  useFrame(({ clock }) => {
    if (!basket.current) return
    const pulse = reducedMotion ? 0.5 : (Math.sin(clock.elapsedTime * 0.34 * motionRate + day) + 1) / 2
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
  motionRate?: number
}

/** Repair trucks and cranes appear only where the real candidate trajectory improved. */
export function RepairActivity({
  day,
  previous,
  services = CANONICAL_SERVICES,
  inactiveServices = [],
  reducedMotion = false,
  motionRate = 1,
}: RepairActivityProps) {
  const plans = useMemo(
    () => repairPlansForDay(day, previous, services, inactiveServices),
    [day, inactiveServices, previous, services],
  )
  return (
    <group name="trajectory-derived-repairs">
      {plans.map((plan) => (
        <group key={`${plan.service}-${plan.buildingIndex}`}>
          {plan.tall
            ? <RepairCrane plan={plan} day={day.day} reducedMotion={reducedMotion} motionRate={motionRate} />
            : <RepairLift plan={plan} day={day.day} reducedMotion={reducedMotion} motionRate={motionRate} />}
          {Array.from({ length: plan.vehicleCount }, (_, index) => (
            <RepairTruck
              key={index}
              plan={plan}
              index={index}
            />
          ))}
        </group>
      ))}
    </group>
  )
}

function stringHash(value: string): number {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

function unitNoise(seed: number, index: number, salt: number): number {
  let value = (seed ^ Math.imul(index + 1, 0x9e3779b1) ^ Math.imul(salt + 1, 0x85ebca6b)) >>> 0
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d)
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b)
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295
}

type ImpactBurstProps = {
  impact: SceneImpact
  tremorTarget?: RefObject<Group | null>
  onComplete?: (impact: SceneImpact) => void
  reducedMotion?: boolean
}

function ImpactBurst({ impact, tremorTarget, onComplete, reducedMotion = false }: ImpactBurstProps) {
  const root = useRef<Group>(null)
  const wave = useRef<Mesh>(null)
  const waveMaterial = useRef<MeshBasicMaterial>(null)
  const bricks = useRef<InstancedMesh>(null)
  const smoke = useRef<InstancedMesh>(null)
  const elapsed = useRef(0)
  const finished = useRef(false)
  const baseline = useRef(new THREE.Vector3())
  const baselineCaptured = useRef(false)
  const onCompleteRef = useRef(onComplete)
  const normalizedSeverity = clamp(impact.severity / 0.4, 0.125, 1)
  const brickCount = 7 + Math.round(normalizedSeverity * 11)
  const smokeCount = 3 + Math.round(normalizedSeverity * 4)
  const colors = IMPACT_COLORS[impact.type]
  const seed = useMemo(() => stringHash(`${impact.id}:${impact.type}:${impact.severity.toFixed(4)}`), [impact])
  const matrix = useMemo(() => new THREE.Matrix4(), [])
  const position = useMemo(() => new THREE.Vector3(), [])
  const rotation = useMemo(() => new THREE.Quaternion(), [])
  const scale = useMemo(() => new THREE.Vector3(), [])
  const euler = useMemo(() => new THREE.Euler(), [])

  useEffect(() => {
    onCompleteRef.current = onComplete
  }, [onComplete])

  useLayoutEffect(() => () => {
    if (baselineCaptured.current && tremorTarget?.current) {
      tremorTarget.current.position.copy(baseline.current)
    }
  }, [tremorTarget])

  useFrame((_, delta) => {
    const step = Math.min(delta, 0.05)
    elapsed.current += step
    const time = elapsed.current
    const duration = 1.65 + normalizedSeverity * 0.3
    const progress = clamp(time / duration, 0, 1)

    if (root.current) root.current.visible = progress < 1
    if (wave.current) {
      const radius = reducedMotion ? 1.15 + normalizedSeverity * 0.4 : 0.55 + progress * (3.1 + normalizedSeverity * 2.2)
      wave.current.scale.setScalar(radius)
      wave.current.position.y = reducedMotion ? 0.06 : 0.05 + progress * 0.018
    }
    if (waveMaterial.current) {
      waveMaterial.current.opacity = Math.pow(1 - progress, 1.45) * (0.42 + normalizedSeverity * 0.2)
    }

    if (bricks.current) {
      for (let index = 0; index < brickCount; index += 1) {
        const angle = unitNoise(seed, index, 0) * Math.PI * 2
        const speed = (0.85 + unitNoise(seed, index, 1) * 1.5) * (0.55 + normalizedSeverity * 0.8)
        const lift = 1.0 + unitNoise(seed, index, 2) * 1.05
        const flightTime = Math.min(time, 1.18)
        const radialDistance = speed * flightTime * (1 - flightTime * 0.18)
        position.set(
          Math.cos(angle) * (reducedMotion ? 0.42 + unitNoise(seed, index, 1) * 0.95 : radialDistance),
          reducedMotion ? 0.12 + unitNoise(seed, index, 2) * 0.18 : Math.max(0.12, 0.15 + lift * flightTime - 2.7 * flightTime * flightTime),
          Math.sin(angle) * (reducedMotion ? 0.42 + unitNoise(seed, index, 1) * 0.95 : radialDistance),
        )
        euler.set(
          reducedMotion ? unitNoise(seed, index, 3) * 0.35 : time * (2.1 + unitNoise(seed, index, 3) * 3.2),
          reducedMotion ? angle : angle + time * (1.2 + unitNoise(seed, index, 4) * 2.4),
          reducedMotion ? unitNoise(seed, index, 5) * 0.3 : time * (1.4 + unitNoise(seed, index, 5) * 2.6),
        )
        rotation.setFromEuler(euler)
        const size = 0.62 + unitNoise(seed, index, 6) * 0.48
        scale.setScalar(size)
        matrix.compose(position, rotation, scale)
        bricks.current.setMatrixAt(index, matrix)
      }
      bricks.current.instanceMatrix.needsUpdate = true
    }

    if (smoke.current) {
      for (let index = 0; index < smokeCount; index += 1) {
        const delay = index * 0.07
        const smokeTime = reducedMotion ? 0.55 + index * 0.12 : Math.max(0, time - delay)
        const angle = unitNoise(seed, index, 7) * Math.PI * 2
        const drift = 0.22 + unitNoise(seed, index, 8) * 0.34
        position.set(
          Math.cos(angle) * smokeTime * drift,
          0.2 + smokeTime * (0.62 + unitNoise(seed, index, 9) * 0.3),
          Math.sin(angle) * smokeTime * drift,
        )
        rotation.identity()
        const puffScale = Math.max(0.01, Math.min(smokeTime * 2.8, 1))
          * (0.32 + normalizedSeverity * 0.25 + unitNoise(seed, index, 10) * 0.18)
        scale.setScalar(puffScale)
        matrix.compose(position, rotation, scale)
        smoke.current.setMatrixAt(index, matrix)
      }
      smoke.current.instanceMatrix.needsUpdate = true
      const material = smoke.current.material as THREE.MeshStandardMaterial
      material.opacity = Math.pow(1 - progress, 1.35) * 0.56
    }

    const target = tremorTarget?.current
    if (target) {
      if (!baselineCaptured.current) {
        baseline.current.copy(target.position)
        baselineCaptured.current = true
      }
      const tremorDuration = 0.24 + normalizedSeverity * 0.18
      if (!reducedMotion && time < tremorDuration) {
        const fade = Math.pow(1 - time / tremorDuration, 2)
        const amplitude = (0.018 + normalizedSeverity * 0.055) * fade
        target.position.set(
          baseline.current.x + Math.sin(time * 91 + seed * 0.001) * amplitude,
          baseline.current.y,
          baseline.current.z + Math.sin(time * 73 + seed * 0.0017) * amplitude * 0.72,
        )
      } else {
        target.position.copy(baseline.current)
      }
    }

    if (progress >= 1 && !finished.current) {
      finished.current = true
      onCompleteRef.current?.(impact)
    }
  })

  return (
    <group ref={root} position={[impact.position[0], impact.position[1], impact.position[2]]}>
      <mesh ref={wave} rotation={[-Math.PI / 2, 0, 0]} renderOrder={11}>
        <ringGeometry args={[0.82, 1, 48]} />
        <meshBasicMaterial
          ref={waveMaterial}
          color={colors.wave}
          transparent
          opacity={0.58}
          depthTest={false}
          depthWrite={false}
          side={THREE.DoubleSide}
          toneMapped={false}
        />
      </mesh>
      <instancedMesh ref={bricks} args={[undefined, undefined, brickCount]} castShadow frustumCulled={false}>
        <boxGeometry args={[0.42, 0.18, 0.24]} />
        <meshStandardMaterial color={colors.brick} roughness={0.91} />
      </instancedMesh>
      <instancedMesh ref={smoke} args={[undefined, undefined, smokeCount]} frustumCulled={false}>
        <dodecahedronGeometry args={[1, 0]} />
        <meshStandardMaterial
          color={colors.smoke}
          transparent
          opacity={0.42}
          depthWrite={false}
          roughness={1}
          flatShading
        />
      </instancedMesh>
    </group>
  )
}

export type ImpactVisualizationProps = {
  /** Current typed kick. Set to `null` after `onComplete`, and change `id` for each kick. */
  impact: SceneImpact | null
  /** Optional stationary world group to offset briefly, producing restrained scene tremor. */
  tremorTarget?: RefObject<Group | null>
  onComplete?: (impact: SceneImpact) => void
  reducedMotion?: boolean
}

/** Ground wave, deterministic brick scatter, smoke, and an optional short world tremor. */
export function ImpactVisualization({ impact, tremorTarget, onComplete, reducedMotion = false }: ImpactVisualizationProps) {
  if (!impact) return null
  return (
    <ImpactBurst
      key={impact.id}
      impact={impact}
      tremorTarget={tremorTarget}
      onComplete={onComplete}
      reducedMotion={reducedMotion}
    />
  )
}

export type SceneEffectsProps = {
  /** Completed compare response currently driving the city and its playback index. */
  result: CompareResponse
  dayIndex: number
  impact?: SceneImpact | null
  /** Pass a ref to the stationary city-world group if impact tremor is desired. */
  tremorTarget?: RefObject<Group | null>
  onImpactComplete?: (impact: SceneImpact) => void
  inactiveServices?: readonly Service[]
  reducedMotion?: boolean
  motionRateOverride?: number
}

/** Concise CityScene integration: all persistent activity is derived from candidate data. */
export function SceneEffects({
  result,
  dayIndex,
  inactiveServices = [],
  reducedMotion = false,
  motionRateOverride,
}: SceneEffectsProps) {
  const contextMotionRate = useContext(SceneMotionContext)
  const motionRate = motionRateOverride ?? contextMotionRate
  const day = result.candidate.trajectory[dayIndex]
  const previous = result.candidate.trajectory[dayIndex - 1]
  if (!day) return null
  return (
    <>
      <RepairActivity
        day={day}
        previous={previous}
        services={result.services}
        inactiveServices={inactiveServices}
        reducedMotion={reducedMotion}
        motionRate={motionRate}
      />
    </>
  )
}
