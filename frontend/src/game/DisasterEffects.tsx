import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { Group, InstancedMesh, Mesh, MeshBasicMaterial } from 'three'
import type { Service, ShockType } from '../types'
import { DISTRICT_BUILDING_OFFSETS, DISTRICTS } from './model'
import { CITY_AIM_RING_RADIUS } from './worldLayout'

export type DisasterWorldPosition = readonly [number, number, number]

/**
 * A view-only event assembled from a verified comparison response. `impact` must be the
 * returned shock vector, and `services` supplied to the planner must be response order.
 */
export type DisasterEffectEvent = {
  id: string | number
  type: ShockType
  severity: number
  /** Player-selected view focus; matches `CityImpactEvent.point` for direct composition. */
  point: DisasterWorldPosition
  service?: Service
  impact: readonly number[]
  /** Shared deterministic direction for front, rain, wind strokes, debris, and civic flag. */
  wind?: readonly [number, number]
}

export type DisasterEffectKind =
  | 'earthquake-crumble'
  | 'supply-interruption'
  | 'epidemic-haze'
  | 'utility-flicker'
  | 'weather-front'

export type DistrictEffectPlan = {
  service: Service
  center: DisasterWorldPosition
  accent: string
  coefficient: number
  /** Exact fractional shock pressure: returned severity × returned impact coefficient. */
  magnitude: number
  /** Readability normalization against the authored maximum severity of 0.40. */
  normalized: number
}

export type DisasterEffectPlan = {
  type: ShockType
  kind: DisasterEffectKind
  severity: number
  normalizedSeverity: number
  position: DisasterWorldPosition
  targetService?: Service
  districts: DistrictEffectPlan[]
  strongestService: Service
}

export const DISASTER_STRIKE_DURATION = 3.1

export const DISASTER_EFFECT_KINDS: Readonly<Record<ShockType, DisasterEffectKind>> = {
  aftershock: 'earthquake-crumble',
  supply: 'supply-interruption',
  epidemic: 'epidemic-haze',
  utility: 'utility-flicker',
  weather: 'weather-front',
}

const EFFECT_COLORS: Readonly<Record<ShockType, string>> = {
  aftershock: '#b28b6c',
  supply: '#b99758',
  epidemic: '#8aa79a',
  utility: '#7fa8b1',
  weather: '#809baa',
}

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, value))
}

export function disasterMagnitude(severity: number, coefficient: number): number {
  return clamp(severity, 0, 0.4) * clamp(coefficient)
}

/** Maps the returned impact vector by the returned service order; no canonical fallback. */
export function disasterEffectPlan(
  event: DisasterEffectEvent,
  services: readonly Service[],
): DisasterEffectPlan {
  const districts = DISTRICTS.map((district): DistrictEffectPlan => {
    const responseIndex = services.indexOf(district.service)
    const coefficient = responseIndex < 0 ? 0 : clamp(event.impact[responseIndex] ?? 0)
    const magnitude = disasterMagnitude(event.severity, coefficient)
    return {
      service: district.service,
      center: district.center,
      accent: district.accent,
      coefficient,
      magnitude,
      normalized: clamp(magnitude / 0.4),
    }
  })
  const strongest = districts.reduce((best, district) => (
    district.magnitude > best.magnitude ? district : best
  ))
  return {
    type: event.type,
    kind: DISASTER_EFFECT_KINDS[event.type],
    severity: clamp(event.severity, 0, 0.4),
    normalizedSeverity: clamp(event.severity / 0.4),
    position: event.point,
    targetService: event.service,
    districts,
    strongestService: strongest.service,
  }
}

function stringHash(value: string): number {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

/** Stable unit noise used for all procedural placement; deliberately no Math.random. */
export function disasterUnitNoise(seed: number, index: number, salt: number): number {
  let value = (seed ^ Math.imul(index + 1, 0x9e3779b1) ^ Math.imul(salt + 1, 0x85ebca6b)) >>> 0
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d)
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b)
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295
}

/**
 * Pixel projection offset for a short, decaying camera tremor. It is intentionally
 * nonzero only for the raw `aftershock` key and always zero under reduced motion.
 */
export function earthquakeCameraOffset(
  type: ShockType,
  severity: number,
  elapsed: number,
  seed: number,
  reducedMotion: boolean,
): readonly [number, number] {
  if (type !== 'aftershock' || reducedMotion || elapsed < 0) return [0, 0]
  const normalized = clamp(severity / 0.4)
  const duration = 0.48 + normalized * 0.24
  if (elapsed >= duration) return [0, 0]
  const fade = Math.pow(1 - elapsed / duration, 2)
  const amplitude = (2.5 + normalized * 9.5) * fade
  const phase = seed * 0.00013
  return [
    Math.sin(elapsed * 83 + phase) * amplitude,
    Math.sin(elapsed * 67 + phase * 1.71) * amplitude * 0.68,
  ]
}

/** A brief, low-amplitude telegraph pulse every few seconds before the boundary strike. */
export function earthquakeForeshockOffset(
  severity: number,
  elapsed: number,
  seed: number,
  reducedMotion: boolean,
): readonly [number, number] {
  if (reducedMotion || elapsed < 0) return [0, 0]
  const phase = elapsed % 2.6
  if (phase >= 0.16) return [0, 0]
  const normalized = clamp(severity / 0.4)
  const fade = Math.pow(1 - phase / 0.16, 2)
  const amplitude = (0.45 + normalized * 1.8) * fade
  const offset = seed * 0.00011
  return [
    Math.sin(phase * 116 + offset) * amplitude,
    Math.sin(phase * 89 + offset * 1.43) * amplitude * 0.55,
  ]
}

function frameStep(delta: number, playbackRate: number): number {
  return Math.min(delta, 0.075) * Math.max(0, playbackRate)
}

function useEffectElapsed(id: string | number, playbackRate: number) {
  const elapsed = useRef(0)
  useEffect(() => {
    elapsed.current = 0
  }, [id])
  useFrame((_, delta) => {
    elapsed.current += frameStep(delta, playbackRate)
  })
  return elapsed
}

function targetStrength(plan: DisasterEffectPlan, fallback: Service): number {
  const service = plan.targetService ?? fallback
  const target = plan.districts.find((district) => district.service === service)
  if (target) return target.normalized
  return serviceStrength(plan, fallback) || plan.normalizedSeverity
}

function serviceStrength(plan: DisasterEffectPlan, service: Service): number {
  return plan.districts.find((district) => district.service === service)?.normalized ?? 0
}

function FootprintRing({
  eventId,
  district,
  mode,
  reducedMotion,
  playbackRate,
}: {
  eventId: string | number
  district: DistrictEffectPlan
  mode: 'telegraph' | 'strike'
  reducedMotion: boolean
  playbackRate: number
}) {
  const mesh = useRef<Mesh>(null)
  const material = useRef<MeshBasicMaterial>(null)
  const elapsed = useEffectElapsed(`${eventId}:${mode}:${district.service}`, playbackRate)
  useFrame(() => {
    const strength = district.normalized
    if (mode === 'telegraph') {
      const pulse = reducedMotion ? 0.45 : (Math.sin(elapsed.current * 1.35) + 1) / 2
      mesh.current?.scale.setScalar(0.9 + strength * 0.34 + pulse * 0.1)
      if (material.current) material.current.opacity = 0.075 + strength * (0.18 + pulse * 0.1)
      return
    }
    const progress = clamp(elapsed.current / 1.75)
    mesh.current?.scale.setScalar(0.66 + progress * (0.8 + strength * 0.45))
    if (material.current) {
      material.current.opacity = Math.pow(1 - progress, 1.35) * (0.14 + strength * 0.42)
    }
  })
  if (district.magnitude <= 0) return null
  return (
    <mesh
      ref={mesh}
      position={[district.center[0], 0.36, district.center[2]]}
      rotation={[-Math.PI / 2, 0, 0]}
      renderOrder={14}
    >
      <ringGeometry args={[CITY_AIM_RING_RADIUS * 0.6, CITY_AIM_RING_RADIUS * 0.615, 64]} />
      <meshBasicMaterial
        ref={material}
        color={district.accent}
        transparent
        opacity={0}
        depthTest={false}
        depthWrite={false}
        toneMapped={false}
      />
    </mesh>
  )
}

function FootprintRings({
  eventId,
  plan,
  mode,
  reducedMotion,
  playbackRate,
}: {
  eventId: string | number
  plan: DisasterEffectPlan
  mode: 'telegraph' | 'strike'
  reducedMotion: boolean
  playbackRate: number
}) {
  return (
    <group name={`${mode}-typed-footprint`}>
      {plan.districts.map((district) => (
        <FootprintRing
          key={district.service}
          eventId={eventId}
          district={district}
          mode={mode}
          reducedMotion={reducedMotion}
          playbackRate={playbackRate}
        />
      ))}
    </group>
  )
}

function FaultTelegraph({ event, strength, reducedMotion, playbackRate }: {
  event: DisasterEffectEvent
  strength: number
  reducedMotion: boolean
  playbackRate: number
}) {
  const root = useRef<Group>(null)
  const elapsed = useEffectElapsed(event.id, playbackRate)
  useFrame(() => {
    if (!root.current) return
    const pulse = reducedMotion ? 0.4 : (Math.sin(elapsed.current * 1.1) + 1) / 2
    root.current.rotation.y = -0.22 + pulse * 0.03
    root.current.scale.setScalar(0.94 + pulse * 0.05)
  })
  return (
    <group ref={root} position={[event.point[0], 0.37, event.point[2]]}>
      {[-4.2, -2.1, 0.05, 2.2, 4.3].map((offset, index) => (
        <mesh key={offset} position={[offset, 0, Math.sin(index * 1.7) * 0.42]} rotation={[0, 0.42 + index * 0.14, 0]}>
          <boxGeometry args={[0.055, 0.025, 2.2 + (index % 2) * 0.8]} />
          <meshBasicMaterial color="#796a5e" transparent opacity={0.14 + strength * 0.28} depthWrite={false} />
        </mesh>
      ))}
    </group>
  )
}

function SupplyQueue({ event, strength, reducedMotion, playbackRate, strike = false }: {
  event: DisasterEffectEvent
  strength: number
  reducedMotion: boolean
  playbackRate: number
  strike?: boolean
}) {
  const gate = useRef<Group>(null)
  const elapsed = useEffectElapsed(event.id, playbackRate)
  useFrame(() => {
    if (!gate.current) return
    const pulse = reducedMotion ? 0.5 : (Math.sin(elapsed.current * (strike ? 4.1 : 1.25)) + 1) / 2
    gate.current.rotation.y = strike ? pulse * 0.045 : 0
  })
  const count = 3 + Math.round(strength * 4)
  return (
    <group position={[0, 0.34, 0]} name={strike ? 'supply-interruption' : 'supply-telegraph'}>
      <group ref={gate}>
        <mesh position={[0, 0.26, 2.15]}>
          <boxGeometry args={[2.8, 0.22, 0.16]} />
          <meshStandardMaterial color="#8c7148" roughness={0.9} />
        </mesh>
        {[-1.25, 1.25].map((x) => (
          <mesh key={x} position={[x, 0.52, 2.15]}>
            <boxGeometry args={[0.16, 1.05, 0.16]} />
            <meshStandardMaterial color="#665f54" roughness={0.94} />
          </mesh>
        ))}
      </group>
      {Array.from({ length: count }, (_, index) => (
        <group key={index} position={[-0.78 + (index % 3) * 0.78, 0, 2.65 + Math.floor(index / 3) * 0.55]}>
          <mesh position={[0, 0.18, 0]} castShadow>
            <boxGeometry args={[0.48, 0.36, 0.42]} />
            <meshStandardMaterial color={index % 2 ? '#9c7b4b' : '#b0925e'} roughness={0.94} />
          </mesh>
          <mesh position={[0, 0.4, 0]}>
            <boxGeometry args={[0.5, 0.045, 0.44]} />
            <meshStandardMaterial color="#625c52" roughness={0.9} />
          </mesh>
        </group>
      ))}
    </group>
  )
}

function HMarker({ opacity = 0.8 }: { opacity?: number }) {
  return (
    <group>
      {[-0.18, 0.18].map((x) => (
        <mesh key={x} position={[x, 0.38, 0]}>
          <boxGeometry args={[0.1, 0.72, 0.07]} />
          <meshStandardMaterial color="#ecebe4" transparent opacity={opacity} roughness={0.88} />
        </mesh>
      ))}
      <mesh position={[0, 0.38, 0]}>
        <boxGeometry args={[0.36, 0.1, 0.075]} />
        <meshStandardMaterial color="#ecebe4" transparent opacity={opacity} roughness={0.88} />
      </mesh>
    </group>
  )
}

function EpidemicResponse({ event, plan, reducedMotion, playbackRate, strike = false }: {
  event: DisasterEffectEvent
  plan: DisasterEffectPlan
  reducedMotion: boolean
  playbackRate: number
  strike?: boolean
}) {
  const root = useRef<Group>(null)
  const elapsed = useEffectElapsed(event.id, playbackRate)
  const healthcare = plan.districts.find((district) => district.service === 'healthcare')!
  const count = (strike ? 3 : 2) + Math.round(healthcare.normalized * 3)
  useFrame(() => {
    if (!root.current) return
    const pulse = reducedMotion ? 0.5 : (Math.sin(elapsed.current * (strike ? 2.4 : 0.85)) + 1) / 2
    root.current.position.z = 2.2 + pulse * 0.18
  })
  return (
    <group position={[healthcare.center[0], 0.35, healthcare.center[2]]}>
      <group ref={root} position={[0, 0, 2.2]}>
        {Array.from({ length: count }, (_, index) => {
          return (
            <group key={index} position={[(index - (count - 1) / 2) * 1.2, 0.25, index % 2 ? 0.55 : 0]}>
              <mesh position={[0, 0.25, 0]} castShadow>
                <boxGeometry args={[0.72, 0.46, 1.05]} />
                <meshStandardMaterial color="#d9ddd6" roughness={0.84} />
              </mesh>
              <mesh position={[0, 0.34, 0.46]}>
                <boxGeometry args={[0.48, 0.22, 0.04]} />
                <meshStandardMaterial color="#667f7b" roughness={0.72} />
              </mesh>
            </group>
          )
        })}
        <group position={[0, 0.2, -1.6]}>
          <mesh position={[0, 0.46, 0]} rotation={[0, 0, Math.PI / 4]} castShadow>
            <boxGeometry args={[1.0, 1.0, 1.5]} />
            <meshStandardMaterial color="#718b86" roughness={0.94} />
          </mesh>
          <group position={[0, 0.3, 0.78]}><HMarker opacity={0.74 + healthcare.normalized * 0.18} /></group>
        </group>
      </group>
      {strike ? (
        <group position={[0, 0.28, 2.25]}>
          <HMarker opacity={0.62 + healthcare.normalized * 0.28} />
        </group>
      ) : null}
    </group>
  )
}

function UtilityIndicators({ event, plan, reducedMotion, playbackRate, strike = false }: {
  event: DisasterEffectEvent
  plan: DisasterEffectPlan
  reducedMotion: boolean
  playbackRate: number
  strike?: boolean
}) {
  return (
    <group name={strike ? 'utility-strike' : 'utility-telegraph'}>
      {plan.districts.map((district, districtIndex) => (
        <UtilityIndicator
          key={district.service}
          eventId={event.id}
          district={district}
          districtIndex={districtIndex}
          reducedMotion={reducedMotion}
          playbackRate={playbackRate}
          strike={strike}
        />
      ))}
    </group>
  )
}

function UtilityIndicator({ eventId, district, districtIndex, reducedMotion, playbackRate, strike }: {
  eventId: string | number
  district: DistrictEffectPlan
  districtIndex: number
  reducedMotion: boolean
  playbackRate: number
  strike: boolean
}) {
  const material = useRef<MeshBasicMaterial>(null)
  const elapsed = useEffectElapsed(`${eventId}:utility:${district.service}`, playbackRate)
  useFrame(() => {
    if (!material.current) return
    const pulse = reducedMotion
      ? 0.58
      : (Math.sin(elapsed.current * (strike ? 13 : 2.2) + districtIndex * 1.7) + 1) / 2
    material.current.opacity = 0.035 + district.normalized * (strike ? 0.5 : 0.22) * pulse
  })
  return (
    <group position={[district.center[0] + 5.2, 0.38, district.center[2] - 4.4]}>
      <mesh position={[0, 0.54, 0]}>
        <boxGeometry args={[0.12, 1.08, 0.12]} />
        <meshStandardMaterial color="#5b6564" roughness={0.93} />
      </mesh>
      <mesh position={[0, 1.1, 0]} renderOrder={15}>
        <boxGeometry args={[0.34, 0.22, 0.09]} />
        <meshBasicMaterial
          ref={material}
          color="#d4e3df"
          transparent
          opacity={0}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
    </group>
  )
}

function WeatherClouds({ event, plan, reducedMotion, playbackRate, strike = false }: {
  event: DisasterEffectEvent
  plan: DisasterEffectPlan
  reducedMotion: boolean
  playbackRate: number
  strike?: boolean
}) {
  const root = useRef<Group>(null)
  const elapsed = useEffectElapsed(event.id, playbackRate)
  const strength = targetStrength(plan, 'transport')
  const [windX, windZ] = event.wind ?? [0.92, 0.38]
  const heading = Math.atan2(windX, windZ)
  useFrame(() => {
    if (!root.current) return
    const gather = reducedMotion ? 0.72 : 0.5 + (Math.sin(elapsed.current * (strike ? 0.7 : 0.28)) + 1) * 0.24
    const approach = strike ? 1 : reducedMotion ? 0.72 : Math.min(1, elapsed.current / 5.5)
    const remaining = (1 - approach) * 18
    root.current.position.set(
      event.point[0] - windX * remaining,
      (strike ? 5.2 : 6.0) - gather * 0.42,
      event.point[2] - windZ * remaining,
    )
    root.current.scale.setScalar(0.78 + gather * 0.28 + strength * 0.12)
    root.current.rotation.y = heading + (reducedMotion ? 0 : Math.sin(elapsed.current * 0.08) * 0.035)
  })
  return (
    <group ref={root} position={[event.point[0] - windX * 18, strike ? 5.2 : 6.0, event.point[2] - windZ * 18]}>
      <mesh position={[0, -0.62, 0]} rotation={[-Math.PI / 2, 0, 0]} renderOrder={15}>
        <circleGeometry args={[6.6 + strength * 1.4, 48]} />
        <meshBasicMaterial
          color={strike ? '#465a60' : '#596b6d'}
          transparent
          opacity={(strike ? 0.16 : 0.1) + strength * 0.12}
          depthTest={false}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
      {Array.from({ length: strike ? 9 : 7 }, (_, index) => {
        const angle = index * 2.399
        const radius = 1.2 + (index % 4) * 1.55
        return (
          <mesh
            key={index}
            position={[Math.cos(angle) * radius, (index % 3) * 0.2, Math.sin(angle) * radius * 0.72]}
            renderOrder={16}
          >
            <dodecahedronGeometry args={[1.5 + (index % 3) * 0.3, 0]} />
            <meshBasicMaterial
              color={strike ? '#52656b' : '#68777a'}
              transparent
              opacity={(strike ? 0.46 : 0.34) + strength * 0.22}
              depthTest={false}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
        )
      })}
    </group>
  )
}

function CrumbleField({ event, district, districtIndex, reducedMotion, playbackRate }: {
  event: DisasterEffectEvent
  district: DistrictEffectPlan
  districtIndex: number
  reducedMotion: boolean
  playbackRate: number
}) {
  const mesh = useRef<InstancedMesh>(null)
  const elapsed = useEffectElapsed(`${event.id}:crumble:${district.service}`, playbackRate)
  const count = district.magnitude > 0 ? 5 + Math.round(district.normalized * 16) : 0
  const seed = useMemo(
    () => stringHash(`${event.id}:${event.type}:${district.service}:${event.severity.toFixed(4)}`),
    [district.service, event.id, event.severity, event.type],
  )
  const matrix = useMemo(() => new THREE.Matrix4(), [])
  const position = useMemo(() => new THREE.Vector3(), [])
  const rotation = useMemo(() => new THREE.Quaternion(), [])
  const scale = useMemo(() => new THREE.Vector3(), [])
  const euler = useMemo(() => new THREE.Euler(), [])
  useFrame(() => {
    if (!mesh.current) return
    const time = reducedMotion ? 0.82 : Math.min(elapsed.current, 1.55)
    for (let index = 0; index < count; index += 1) {
      const buildingOffset = DISTRICT_BUILDING_OFFSETS[
        Math.floor(disasterUnitNoise(seed, index, 0) * DISTRICT_BUILDING_OFFSETS.length)
          % DISTRICT_BUILDING_OFFSETS.length
      ]
      const angle = disasterUnitNoise(seed, index, 1) * Math.PI * 2
      const launch = 0.42 + disasterUnitNoise(seed, index, 2) * 0.75
      const localTime = Math.max(0, time - index * 0.035)
      position.set(
        buildingOffset[0] + Math.cos(angle) * launch * localTime,
        Math.max(0.13, 1.15 + disasterUnitNoise(seed, index, 3) * 2.2 + localTime * 0.45 - localTime * localTime * 2.2),
        buildingOffset[1] + Math.sin(angle) * launch * localTime,
      )
      euler.set(localTime * 2.4, angle + localTime * 2.1, localTime * 1.8)
      rotation.setFromEuler(euler)
      scale.setScalar(0.66 + disasterUnitNoise(seed, index, 4) * 0.38)
      matrix.compose(position, rotation, scale)
      mesh.current.setMatrixAt(index, matrix)
    }
    mesh.current.instanceMatrix.needsUpdate = true
  })
  if (count === 0) return null
  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, count]}
      position={[district.center[0], 0, district.center[2]]}
      castShadow
      frustumCulled={false}
    >
      <boxGeometry args={[0.38, 0.18, 0.24]} />
      <meshStandardMaterial color={districtIndex % 2 ? '#77685c' : '#806e5f'} roughness={0.94} />
    </instancedMesh>
  )
}

function SupplyRoadInterruption({ event, plan, reducedMotion, playbackRate }: {
  event: DisasterEffectEvent
  plan: DisasterEffectPlan
  reducedMotion: boolean
  playbackRate: number
}) {
  return (
    <group name="supply-road-interruption">
      {plan.districts.map((district, index) => {
        const x = district.center[0] * 0.42
        const z = district.center[2] * 0.42
        const rotation = Math.atan2(district.center[0], district.center[2])
        return (
          <SupplyRoadBarrier
            key={district.service}
            eventId={event.id}
            district={district}
            index={index}
            position={[x, 0.39, z]}
            rotation={rotation}
            reducedMotion={reducedMotion}
            playbackRate={playbackRate}
          />
        )
      })}
      <SupplyQueue event={event} strength={serviceStrength(plan, 'food')} reducedMotion={reducedMotion} playbackRate={playbackRate} strike />
    </group>
  )
}

function SupplyRoadBarrier({ eventId, district, index, position, rotation, reducedMotion, playbackRate }: {
  eventId: string | number
  district: DistrictEffectPlan
  index: number
  position: DisasterWorldPosition
  rotation: number
  reducedMotion: boolean
  playbackRate: number
}) {
  const signal = useRef<MeshBasicMaterial>(null)
  const elapsed = useEffectElapsed(`${eventId}:supply-road:${district.service}`, playbackRate)
  const opacity = 0.16 + district.normalized * 0.58
  useFrame(() => {
    if (!signal.current) return
    const pulse = reducedMotion ? 0.45 : 0.18 + (Math.sin(elapsed.current * 5 + index * 1.3) + 1) * 0.25
    signal.current.opacity = opacity * pulse
  })
  return (
    <group position={[position[0], position[1], position[2]]} rotation={[0, rotation, 0]}>
      <mesh position={[0, 0.18, 0]} castShadow>
        <boxGeometry args={[1.15 + district.normalized * 0.75, 0.22, 0.18]} />
        <meshStandardMaterial color={index % 2 ? '#8d724b' : '#a18451'} transparent opacity={opacity} roughness={0.9} />
      </mesh>
      {[-0.42, 0.42].map((offset) => (
        <mesh key={offset} position={[offset, 0, 0]} rotation={[0, 0, -0.18]}>
          <boxGeometry args={[0.11, 0.44, 0.11]} />
          <meshStandardMaterial color="#625e55" roughness={0.94} />
        </mesh>
      ))}
      <mesh position={[0, 0.48, -0.03]} renderOrder={15}>
        <sphereGeometry args={[0.09, 12, 8]} />
        <meshBasicMaterial
          ref={signal}
          color="#d1b475"
          transparent
          opacity={0}
          toneMapped={false}
        />
      </mesh>
    </group>
  )
}

function RainField({ event, district, districtIndex, reducedMotion, playbackRate }: {
  event: DisasterEffectEvent
  district: DistrictEffectPlan
  districtIndex: number
  reducedMotion: boolean
  playbackRate: number
}) {
  const mesh = useRef<InstancedMesh>(null)
  const elapsed = useEffectElapsed(`${event.id}:rain:${district.service}`, playbackRate)
  const count = district.magnitude > 0 ? 10 + Math.round(district.normalized * 22) : 0
  const seed = useMemo(
    () => stringHash(`${event.id}:rain:${district.service}:${event.severity.toFixed(4)}`),
    [district.service, event.id, event.severity],
  )
  const matrix = useMemo(() => new THREE.Matrix4(), [])
  const position = useMemo(() => new THREE.Vector3(), [])
  const rotation = useMemo(() => {
    const [windX, windZ] = event.wind ?? [0.92, 0.38]
    return new THREE.Quaternion().setFromEuler(new THREE.Euler(windZ * 0.22, Math.atan2(windX, windZ), -windX * 0.22))
  }, [event.wind])
  const scale = useMemo(() => new THREE.Vector3(1, 1, 1), [])
  useFrame(() => {
    if (!mesh.current) return
    for (let index = 0; index < count; index += 1) {
      const phase = reducedMotion
        ? disasterUnitNoise(seed, index, 0)
        : (disasterUnitNoise(seed, index, 0) + elapsed.current * (0.45 + district.normalized * 0.42)) % 1
      position.set(
        district.center[0] + (disasterUnitNoise(seed, index, 1) - 0.5) * 14.5,
        0.52 + (1 - phase) * 5.4,
        district.center[2] + (disasterUnitNoise(seed, index, 2) - 0.5) * 13.8,
      )
      matrix.compose(position, rotation, scale)
      mesh.current.setMatrixAt(index, matrix)
    }
    mesh.current.instanceMatrix.needsUpdate = true
  })
  if (count === 0) return null
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, count]} frustumCulled={false} renderOrder={13}>
      <boxGeometry args={[0.035, 0.58, 0.035]} />
      <meshBasicMaterial
        color={districtIndex % 2 ? '#b4c7cb' : '#a9c0c7'}
        transparent
        opacity={0.15 + district.normalized * 0.32}
        depthWrite={false}
        toneMapped={false}
      />
    </instancedMesh>
  )
}

function WindStrokes({ event, plan, reducedMotion, playbackRate }: {
  event: DisasterEffectEvent
  plan: DisasterEffectPlan
  reducedMotion: boolean
  playbackRate: number
}) {
  const root = useRef<Group>(null)
  const elapsed = useEffectElapsed(`${event.id}:wind`, playbackRate)
  const [windX, windZ] = event.wind ?? [0.92, 0.38]
  useFrame(() => {
    if (!root.current || reducedMotion) return
    root.current.position.x = Math.sin(elapsed.current * 1.4) * 0.28
  })
  return (
    <group ref={root} rotation={[0, Math.atan2(windX, windZ), 0]}>
      {plan.districts.flatMap((district, districtIndex) => (
        Array.from({ length: 3 + Math.round(district.normalized * 4) }, (_, index) => (
          <mesh
            key={`${district.service}-${index}`}
            position={[
              district.center[0] - 5.2 + index * 1.7,
              2.0 + (index % 3) * 0.75,
              district.center[2] + ((index + districtIndex) % 2 ? -4.2 : 3.8),
            ]}
            rotation={[0, -0.34, 0.12]}
          >
            <boxGeometry args={[2.2 + district.normalized * 1.5, 0.025, 0.045]} />
            <meshBasicMaterial color="#d0dcdb" transparent opacity={0.08 + district.normalized * 0.18} depthWrite={false} />
          </mesh>
        ))
      ))}
    </group>
  )
}

function CameraImpactShake({ event, reducedMotion, playbackRate }: {
  event: DisasterEffectEvent
  reducedMotion: boolean
  playbackRate: number
}) {
  const { camera, size } = useThree()
  const elapsed = useEffectElapsed(`${event.id}:camera`, playbackRate)
  const seed = useMemo(() => stringHash(`${event.id}:${event.type}:${event.severity.toFixed(4)}`), [event.id, event.severity, event.type])
  useEffect(() => () => {
    if (camera instanceof THREE.PerspectiveCamera) camera.clearViewOffset()
  }, [camera])
  useFrame(() => {
    if (!(camera instanceof THREE.PerspectiveCamera)) return
    if (playbackRate <= 0) {
      camera.clearViewOffset()
      return
    }
    const [x, y] = earthquakeCameraOffset(event.type, event.severity, elapsed.current, seed, reducedMotion)
    if (x === 0 && y === 0) {
      camera.clearViewOffset()
      return
    }
    camera.setViewOffset(size.width, size.height, x, y, size.width, size.height)
  })
  return null
}

function CameraForeshock({ event, reducedMotion, playbackRate }: {
  event: DisasterEffectEvent
  reducedMotion: boolean
  playbackRate: number
}) {
  const { camera, size } = useThree()
  const elapsed = useEffectElapsed(`${event.id}:foreshock`, playbackRate)
  const seed = useMemo(() => stringHash(`${event.id}:foreshock:${event.severity.toFixed(4)}`), [event.id, event.severity])
  useEffect(() => () => {
    if (camera instanceof THREE.PerspectiveCamera) camera.clearViewOffset()
  }, [camera])
  useFrame(() => {
    if (!(camera instanceof THREE.PerspectiveCamera)) return
    const [x, y] = earthquakeForeshockOffset(event.severity, elapsed.current, seed, reducedMotion)
    if (x === 0 && y === 0) camera.clearViewOffset()
    else camera.setViewOffset(size.width, size.height, x, y, size.width, size.height)
  })
  return null
}

function StrikeLifecycle({ eventId, playbackRate, onComplete }: {
  eventId: string | number
  playbackRate: number
  onComplete?: () => void
}) {
  const elapsed = useEffectElapsed(`${eventId}:lifecycle`, playbackRate)
  const completed = useRef(false)
  const completeRef = useRef(onComplete)
  useEffect(() => {
    completed.current = false
  }, [eventId])
  useEffect(() => {
    completeRef.current = onComplete
  }, [onComplete])
  useFrame(() => {
    if (completed.current || elapsed.current < DISASTER_STRIKE_DURATION) return
    completed.current = true
    completeRef.current?.()
  })
  return null
}

export function DisasterTelegraph({
  event,
  services,
  reducedMotion = false,
  playbackRate = 1,
}: {
  event: DisasterEffectEvent
  services: readonly Service[]
  reducedMotion?: boolean
  playbackRate?: number
}) {
  const plan = useMemo(() => disasterEffectPlan(event, services), [event, services])
  const strength = targetStrength(plan, plan.strongestService)
  return (
    <group name={`disaster-telegraph-${plan.kind}`}>
      <FootprintRings eventId={event.id} plan={plan} mode="telegraph" reducedMotion={reducedMotion} playbackRate={playbackRate} />
      {event.type === 'aftershock' ? (
        <>
          <CameraForeshock event={event} reducedMotion={reducedMotion} playbackRate={playbackRate} />
          <FaultTelegraph event={event} strength={strength} reducedMotion={reducedMotion} playbackRate={playbackRate} />
        </>
      ) : null}
      {event.type === 'supply' ? <SupplyQueue event={event} strength={serviceStrength(plan, 'food')} reducedMotion={reducedMotion} playbackRate={playbackRate} /> : null}
      {event.type === 'epidemic' ? <EpidemicResponse event={event} plan={plan} reducedMotion={reducedMotion} playbackRate={playbackRate} /> : null}
      {event.type === 'utility' ? <UtilityIndicators event={event} plan={plan} reducedMotion={reducedMotion} playbackRate={playbackRate} /> : null}
      {event.type === 'weather' ? <WeatherClouds event={event} plan={plan} reducedMotion={reducedMotion} playbackRate={playbackRate} /> : null}
    </group>
  )
}

export function DisasterStrike({
  event,
  services,
  reducedMotion = false,
  playbackRate = 1,
  onComplete,
}: {
  event: DisasterEffectEvent
  services: readonly Service[]
  reducedMotion?: boolean
  playbackRate?: number
  onComplete?: () => void
}) {
  const plan = useMemo(() => disasterEffectPlan(event, services), [event, services])
  return (
    <group name={`disaster-strike-${plan.kind}`}>
      <StrikeLifecycle eventId={event.id} playbackRate={playbackRate} onComplete={onComplete} />
      {event.type === 'aftershock' ? (
        <CameraImpactShake event={event} reducedMotion={reducedMotion} playbackRate={playbackRate} />
      ) : null}
      <FootprintRings eventId={event.id} plan={plan} mode="strike" reducedMotion={reducedMotion} playbackRate={playbackRate} />
      {event.type === 'aftershock' ? plan.districts.map((district, index) => (
        <CrumbleField
          key={district.service}
          event={event}
          district={district}
          districtIndex={index}
          reducedMotion={reducedMotion}
          playbackRate={playbackRate}
        />
      )) : null}
      {event.type === 'supply' ? <SupplyRoadInterruption event={event} plan={plan} reducedMotion={reducedMotion} playbackRate={playbackRate} /> : null}
      {event.type === 'epidemic' ? <EpidemicResponse event={event} plan={plan} reducedMotion={reducedMotion} playbackRate={playbackRate} strike /> : null}
      {event.type === 'utility' ? <UtilityIndicators event={event} plan={plan} reducedMotion={reducedMotion} playbackRate={playbackRate} strike /> : null}
      {event.type === 'weather' ? (
        <>
          <WeatherClouds event={event} plan={plan} reducedMotion={reducedMotion} playbackRate={playbackRate} strike />
          {plan.districts.map((district, index) => (
            <RainField
              key={district.service}
              event={event}
              district={district}
              districtIndex={index}
              reducedMotion={reducedMotion}
              playbackRate={playbackRate}
            />
          ))}
          <WindStrokes event={event} plan={plan} reducedMotion={reducedMotion} playbackRate={playbackRate} />
        </>
      ) : null}
    </group>
  )
}

/** Convenience composition for CityScene integration. */
export function DisasterPresentation({
  telegraph,
  strike,
  services,
  reducedMotion = false,
  playbackRate = 1,
  onStrikeComplete,
}: {
  telegraph?: DisasterEffectEvent | null
  strike?: DisasterEffectEvent | null
  services: readonly Service[]
  reducedMotion?: boolean
  playbackRate?: number
  onStrikeComplete?: () => void
}) {
  return (
    <>
      {telegraph ? (
        <DisasterTelegraph
          event={telegraph}
          services={services}
          reducedMotion={reducedMotion}
          playbackRate={playbackRate}
        />
      ) : null}
      {strike ? (
        <DisasterStrike
          event={strike}
          services={services}
          reducedMotion={reducedMotion}
          playbackRate={playbackRate}
          onComplete={onStrikeComplete}
        />
      ) : null}
    </>
  )
}

/** Public palette hook for adjacent restrained overlays such as convoy interruption. */
export function disasterEffectColor(type: ShockType): string {
  return EFFECT_COLORS[type]
}
