import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { Group, InstancedMesh, MeshStandardMaterial } from 'three'
import type { CompareResponse } from '../types'
import { latestIncident, recoveryArcForService } from './realism'
import {
  CITY_BRIDGE,
  CITY_RAIL_TIES,
  CITY_ROAD_SEGMENTS,
  type WorldPoint,
} from './worldLayout'

export type RoadCondition = 'open' | 'cracked' | 'restricted' | 'closed'

export type RoadNetworkState = {
  condition: RoadCondition
  transportLevel: number
  closedSegments: number
  debrisSegments: number
  detourActive: boolean
  railAvailable: boolean
  signalPower: number
  freshPatches: number
  quakeAge: number | null
  weatherAge: number | null
}

export function roadNetworkState(result: CompareResponse, dayIndex: number): RoadNetworkState {
  const day = result.candidate.trajectory[dayIndex]
  const previous = result.candidate.trajectory[dayIndex - 1]
  const transportIndex = result.services.indexOf('transport')
  const civicIndex = result.services.indexOf('public_services')
  const transportLevel = day.services_end[transportIndex]
  const incident = latestIncident(result, dayIndex, 4)
  const quakeAge = incident?.type === 'aftershock' ? incident.ageDays : null
  const weatherAge = incident?.type === 'weather' ? incident.ageDays : null
  const quakeLoad = incident?.type === 'aftershock'
    ? incident.shock.severity * (incident.shock.impact[transportIndex] ?? 0)
    : 0
  let condition: RoadCondition = 'open'
  if (transportLevel < 0.24 || quakeLoad > 0.22 && quakeAge === 0) condition = 'closed'
  else if (transportLevel < 0.42 || quakeLoad > 0.13 && (quakeAge ?? 9) <= 1) condition = 'restricted'
  else if (transportLevel < 0.68 || quakeAge !== null && quakeAge <= 2) condition = 'cracked'
  const closedSegments = condition === 'closed' ? 4 : condition === 'restricted' ? 2 : 0
  const recoveryArc = recoveryArcForService(result, dayIndex, 'transport')
  const debrisSegments = quakeAge === null
    ? 0
    : Math.max(0, Math.round((1 - recoveryArc.progress) * 6) - quakeAge)
  const recovered = previous ? transportLevel - previous.services_end[transportIndex] : 0
  return {
    condition,
    transportLevel,
    closedSegments,
    debrisSegments,
    detourActive: condition === 'closed' || condition === 'restricted',
    railAvailable: transportLevel >= 0.55,
    signalPower: day.services_end[civicIndex],
    freshPatches: recovered > 0.003 ? Math.min(5, Math.ceil(recovered / 0.008)) : 0,
    quakeAge,
    weatherAge,
  }
}

function RailSpur({ available }: { available: boolean }) {
  const ties = useRef<InstancedMesh>(null)
  useLayoutEffect(() => {
    if (!ties.current) return
    const matrix = new THREE.Matrix4()
    CITY_RAIL_TIES.forEach((tie, index) => {
      matrix.compose(
        new THREE.Vector3(...tie.position),
        new THREE.Quaternion().setFromEuler(new THREE.Euler(0, tie.rotation, 0)),
        new THREE.Vector3(...tie.scale),
      )
      ties.current!.setMatrixAt(index, matrix)
    })
    ties.current.instanceMatrix.needsUpdate = true
    ties.current.computeBoundingSphere()
  }, [])
  return (
    <group name="regional-rail-spur">
      <instancedMesh ref={ties} args={[undefined, undefined, CITY_RAIL_TIES.length]} receiveShadow>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#6a5544" roughness={0.98} />
      </instancedMesh>
      {[-0.52, 0.52].map((x) => (
        <mesh key={x} position={[x, 0.39, -12.4]} receiveShadow>
          <boxGeometry args={[0.09, 0.08, 23.0]} />
          <meshStandardMaterial color={available ? '#6d7471' : '#817b72'} metalness={0.22} roughness={0.72} />
        </mesh>
      ))}
      <Html position={[1.35, 0.8, -21.8]} className="world-label-anchor" zIndexRange={[12, 0]}>
        <div className="world-label compact"><b>REGIONAL SPUR</b><span>{available ? 'freight routing active' : 'freight shifted to road'}</span></div>
      </Html>
    </group>
  )
}

function RoadMarks({ state }: { state: RoadNetworkState }) {
  const cracks = useRef<InstancedMesh>(null)
  const crackCount = state.condition === 'open' ? 0 : state.condition === 'cracked' ? 12 : state.condition === 'restricted' ? 22 : 30
  useLayoutEffect(() => {
    if (!cracks.current) return
    const matrix = new THREE.Matrix4()
    for (let index = 0; index < crackCount; index += 1) {
      const segment = CITY_ROAD_SEGMENTS[(index * 7 + 3) % CITY_ROAD_SEGMENTS.length]
      const offset = ((index % 5) - 2) * 0.42
      const position = new THREE.Vector3(
        segment.position[0] + (segment.rotation === 0 ? offset * 2.4 : offset),
        0.31,
        segment.position[2] + (segment.rotation === 0 ? offset : offset * 2.4),
      )
      const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, segment.rotation + 0.35 + (index % 3) * 0.3, 0))
      matrix.compose(position, rotation, new THREE.Vector3(0.72 + (index % 4) * 0.18, 0.018, 0.045))
      cracks.current!.setMatrixAt(index, matrix)
    }
    cracks.current.instanceMatrix.needsUpdate = true
    cracks.current.computeBoundingSphere()
  }, [crackCount])
  return (
    <instancedMesh ref={cracks} args={[undefined, undefined, Math.max(1, crackCount)]} visible={crackCount > 0}>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial color="#343b39" transparent opacity={0.55} />
    </instancedMesh>
  )
}

function Barrier({ position, rotation = 0 }: { position: WorldPoint; rotation?: number }) {
  return (
    <group position={position} rotation={[0, rotation, 0]}>
      <mesh position={[0, 0.42, 0]} castShadow><boxGeometry args={[2.1, 0.22, 0.16]} /><meshStandardMaterial color="#c18e46" roughness={0.88} /></mesh>
      {[-0.85, 0.85].map((x) => <mesh key={x} position={[x, 0.22, 0]}><boxGeometry args={[0.12, 0.46, 0.13]} /><meshStandardMaterial color="#555c58" /></mesh>)}
      <mesh position={[0, 0.78, 0]}><boxGeometry args={[0.78, 0.32, 0.08]} /><meshStandardMaterial color="#d4c8a7" roughness={0.9} /></mesh>
    </group>
  )
}

function BridgeChokepoint({ state }: { state: RoadNetworkState }) {
  const closed = state.condition === 'closed' || state.quakeAge === 0 && state.transportLevel < 0.5
  return (
    <group name="overpass-chokepoint">
      <mesh position={[CITY_BRIDGE.position[0], 0.17, CITY_BRIDGE.position[2]]} receiveShadow>
        <boxGeometry args={[CITY_BRIDGE.size[0] + 1.8, 0.25, 4.4]} />
        <meshStandardMaterial color="#546b70" roughness={0.92} />
      </mesh>
      <mesh position={CITY_BRIDGE.position} receiveShadow castShadow>
        <boxGeometry args={CITY_BRIDGE.size} />
        <meshStandardMaterial color={closed ? '#666966' : '#777c78'} roughness={0.94} />
      </mesh>
      {[-3.1, 3.1].map((x) => (
        <group key={x}>
          <mesh position={[CITY_BRIDGE.position[0] + x, 0.72, -3.05]} castShadow><boxGeometry args={[0.35, 0.98, 0.35]} /><meshStandardMaterial color="#666c68" /></mesh>
          <mesh position={[CITY_BRIDGE.position[0] + x, 0.72, -0.95]} castShadow><boxGeometry args={[0.35, 0.98, 0.35]} /><meshStandardMaterial color="#666c68" /></mesh>
        </group>
      ))}
      {closed ? (
        <>
          <Barrier position={[7.5, 0.35, -2]} rotation={Math.PI / 2} />
          {[0, 1, 2].map((index) => (
            <mesh key={index} position={[4.0 - index * 1.25, 0.42, -2.48]} castShadow>
              <boxGeometry args={[0.7, 0.38, 1.1]} />
              <meshStandardMaterial color={index % 2 ? '#b9b1a2' : '#6b7a78'} roughness={0.84} />
            </mesh>
          ))}
        </>
      ) : null}
    </group>
  )
}

function DebrisClearance({ state, reducedMotion, motionRate }: {
  state: RoadNetworkState
  reducedMotion: boolean
  motionRate: number
}) {
  const loader = useRef<Group>(null)
  const elapsed = useRef(0)
  useFrame((_, delta) => {
    if (!loader.current || state.debrisSegments <= 0) return
    elapsed.current += Math.min(delta, 0.05) * motionRate * (reducedMotion ? 0.25 : 1)
    const progress = (elapsed.current * 0.12) % 1
    loader.current.position.x = -3 + progress * 6
    loader.current.rotation.y = progress > 0.82 ? Math.PI : 0
  })
  if (state.debrisSegments <= 0) return null
  return (
    <group position={[0, 0.3, -2]} name="arterial-first-debris-clearance">
      {Array.from({ length: state.debrisSegments * 3 }, (_, index) => (
        <mesh key={index} position={[-3.8 + (index % 7) * 1.05, 0.14 + (index % 2) * 0.12, (index % 3 - 1) * 0.45]} rotation={[index * 0.15, index * 0.8, 0]} castShadow>
          <boxGeometry args={[0.52, 0.25, 0.36]} />
          <meshStandardMaterial color={index % 3 ? '#766d63' : '#8a7e70'} roughness={0.96} />
        </mesh>
      ))}
      <group ref={loader} position={[-3, 0.1, 1.1]}>
        <mesh position={[0, 0.35, 0]} castShadow><boxGeometry args={[1.15, 0.62, 1.45]} /><meshStandardMaterial color="#c19a4b" roughness={0.82} /></mesh>
        <mesh position={[0, 0.23, 1.05]} rotation={[0.18, 0, 0]} castShadow><boxGeometry args={[1.28, 0.18, 0.85]} /><meshStandardMaterial color="#8f7141" roughness={0.9} /></mesh>
      </group>
      <group position={[3.25, 0.08, 1.18]} rotation={[0, Math.PI, 0]}>
        <mesh position={[0, 0.34, 0]} castShadow><boxGeometry args={[1.08, 0.58, 1.55]} /><meshStandardMaterial color="#766d63" roughness={0.9} /></mesh>
        <mesh position={[0, 0.62, -0.36]} castShadow><boxGeometry args={[0.96, 0.46, 0.64]} /><meshStandardMaterial color="#b18b4b" roughness={0.85} /></mesh>
        {[-0.5, 0.5].flatMap((x) => [-0.48, 0.48].map((z) => (
          <mesh key={`${x}-${z}`} position={[x, 0.12, z]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.17, 0.17, 0.1, 10]} /><meshStandardMaterial color="#303633" roughness={0.94} /></mesh>
        )))}
      </group>
    </group>
  )
}

function FreshPatches({ count }: { count: number }) {
  return (
    <group name="fresh-asphalt-repair-patches">
      {Array.from({ length: count }, (_, index) => (
        <mesh key={index} position={[-8 + index * 4.2, 0.32, -2 + (index % 2 ? 0.35 : -0.35)]} rotation={[0, 0.12 * (index - 2), 0]} receiveShadow>
          <boxGeometry args={[2.4, 0.018, 0.58]} />
          <meshStandardMaterial color="#3f4845" roughness={0.88} />
        </mesh>
      ))}
    </group>
  )
}

function TrafficSignals({ power, utilityFailure, reducedMotion }: {
  power: number
  utilityFailure: boolean
  reducedMotion: boolean
}) {
  const lamps = useRef<Group>(null)
  useFrame(({ clock }) => {
    if (!lamps.current) return
    const enabled = power >= 0.45 && !utilityFailure
    lamps.current.children.forEach((child, index) => {
      const material = (child as THREE.Mesh).material as MeshStandardMaterial
      if (!material) return
      const flicker = utilityFailure && !reducedMotion ? Math.sin(clock.elapsedTime * 11 + index) > 0.25 : false
      material.emissiveIntensity = enabled ? 0.55 : flicker ? 0.18 : 0
      material.opacity = enabled || flicker ? 1 : 0.42
    })
  })
  const positions: WorldPoint[] = [[-5, 0.3, -2], [5, 0.3, -2], [0, 0.3, -7], [0, 0.3, 5]]
  return (
    <group name="intersection-signals">
      {positions.map((position, index) => (
        <group key={index} position={position}>
          <mesh position={[0, 1.1, 0]} castShadow><boxGeometry args={[0.1, 2.2, 0.1]} /><meshStandardMaterial color="#4f5855" /></mesh>
          <group ref={index === 0 ? lamps : undefined}>
            <mesh position={[0, 2.12, 0]}><boxGeometry args={[0.28, 0.42, 0.2]} /><meshStandardMaterial color="#54615d" emissive="#d5b26a" transparent /></mesh>
          </group>
          {(power < 0.45 || utilityFailure) ? (
            <group position={[0.72, 0.25, 0]}>
              <mesh position={[0, 0.34, 0]}><boxGeometry args={[0.84, 0.45, 0.08]} /><meshStandardMaterial color="#d2c4a3" /></mesh>
              <mesh position={[0, 0.08, 0]}><boxGeometry args={[0.08, 0.6, 0.08]} /><meshStandardMaterial color="#555a55" /></mesh>
            </group>
          ) : null}
        </group>
      ))}
    </group>
  )
}

function LightPolesAndMainBreak({ state, severity, reducedMotion, motionRate }: {
  state: RoadNetworkState
  severity: number
  reducedMotion: boolean
  motionRate: number
}) {
  const spray = useRef<Group>(null)
  useFrame(({ clock }) => {
    if (!spray.current) return
    const pulse = reducedMotion ? 0.6 : 0.55 + Math.sin(clock.elapsedTime * 3.5 * motionRate) * 0.2
    spray.current.scale.y = pulse
  })
  const toppled = state.quakeAge === 0 ? Math.min(4, Math.ceil(severity * 10)) : 0
  const capped = state.quakeAge === null || state.quakeAge > 0
  return (
    <group name="roadside-utilities">
      {Array.from({ length: 10 }, (_, index) => {
        const x = -22 + index * 4.8
        const fallen = index < toppled
        return (
          <group key={index} position={[x, 0.28, -4.05]} rotation={[0, 0, fallen ? Math.PI / 2.3 : 0]}>
            <mesh position={[0, 1.35, 0]} castShadow><boxGeometry args={[0.11, 2.7, 0.11]} /><meshStandardMaterial color="#555f5c" roughness={0.9} /></mesh>
            <mesh position={[0.32, 2.62, 0]}><boxGeometry args={[0.72, 0.08, 0.08]} /><meshStandardMaterial color="#555f5c" /></mesh>
          </group>
        )
      })}
      {!capped ? (
        <group ref={spray} position={[-16.5, 0.35, -9.4]}>
          {[-0.28, 0, 0.28].map((x) => (
            <mesh key={x} position={[x, 0.8, 0]} rotation={[0, 0, x * 0.8]}>
              <cylinderGeometry args={[0.025, 0.06, 1.7, 8]} />
              <meshStandardMaterial color="#91b8bc" transparent opacity={0.52} roughness={0.35} />
            </mesh>
          ))}
        </group>
      ) : null}
    </group>
  )
}

export function InfrastructureScene({
  result,
  dayIndex,
  reducedMotion,
  motionRate,
}: {
  result: CompareResponse
  dayIndex: number
  reducedMotion: boolean
  motionRate: number
}) {
  const state = useMemo(() => roadNetworkState(result, dayIndex), [dayIndex, result])
  const day = result.candidate.trajectory[dayIndex]
  const utilityFailure = day.shock.type === 'utility' || (
    latestIncident(result, dayIndex, 2)?.type === 'utility'
    && state.signalPower < 0.58
  )
  const severity = state.quakeAge === null ? 0 : latestIncident(result, dayIndex, 4)?.shock.severity ?? 0
  return (
    <group name="state-derived-infrastructure">
      <RailSpur available={state.railAvailable} />
      <RoadMarks state={state} />
      <BridgeChokepoint state={state} />
      {state.closedSegments > 0 ? (
        <>
          <Barrier position={[-6, 0.35, -2]} rotation={Math.PI / 2} />
          {state.closedSegments > 2 ? <Barrier position={[0, 0.35, 5]} /> : null}
        </>
      ) : null}
      <DebrisClearance state={state} reducedMotion={reducedMotion} motionRate={motionRate} />
      <FreshPatches count={state.freshPatches} />
      <TrafficSignals power={state.signalPower} utilityFailure={utilityFailure} reducedMotion={reducedMotion} />
      <LightPolesAndMainBreak state={state} severity={severity} reducedMotion={reducedMotion} motionRate={motionRate} />
    </group>
  )
}
