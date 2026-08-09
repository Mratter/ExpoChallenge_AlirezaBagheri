import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { Group, Mesh, MeshStandardMaterial } from 'three'
import type { CompareResponse, Service, ShockType } from '../types'
import { DISTRICTS, SERVICE_LABELS } from './model'
import {
  disasterWind,
  intensityBand,
  latestIncident,
  recoveryArcForService,
  strongestShockService,
} from './realism'
import {
  CITY_CIVIC_FLAG,
  CITY_DEPOTS,
  CITY_HOUSING_PARK,
  CITY_SUBSTATION,
  CITY_WATER_TOWER,
  type WorldPoint,
} from './worldLayout'

function serviceLevel(result: CompareResponse, dayIndex: number, service: Service): number {
  const index = result.services.indexOf(service)
  return result.candidate.trajectory[dayIndex].services_end[index] ?? 0
}

function targetCenter(result: CompareResponse, dayIndex: number): WorldPoint {
  const incident = latestIncident(result, dayIndex, 4)
  if (!incident) return [0, 0.3, 0]
  const service = strongestShockService(incident.shock, result.services)
  return DISTRICTS.find((district) => district.service === service)?.center ?? [0, 0.3, 0]
}

function Tent({ position, color = '#607f8e', scale = 1 }: {
  position: WorldPoint
  color?: string
  scale?: number
}) {
  return (
    <group position={position} scale={scale}>
      <mesh position={[0, 0.38, 0]} rotation={[0, 0, Math.PI / 4]} castShadow>
        <boxGeometry args={[0.85, 0.85, 1.35]} />
        <meshStandardMaterial color={color} roughness={0.94} />
      </mesh>
      <mesh position={[0, 0.18, 0.68]}><boxGeometry args={[0.5, 0.5, 0.04]} /><meshStandardMaterial color="#4b5d61" roughness={0.92} /></mesh>
    </group>
  )
}

function GeneratorTrailer({ position, rotation = 0 }: { position: WorldPoint; rotation?: number }) {
  return (
    <group position={position} rotation={[0, rotation, 0]}>
      <mesh position={[0, 0.42, 0]} castShadow><boxGeometry args={[1.55, 0.78, 0.82]} /><meshStandardMaterial color="#c4b893" roughness={0.86} /></mesh>
      <mesh position={[0, 0.43, 0.43]}><boxGeometry args={[0.65, 0.28, 0.04]} /><meshStandardMaterial color="#505b59" /></mesh>
      {[-0.5, 0.5].map((x) => <mesh key={x} position={[x, 0.08, 0]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.18, 0.18, 0.12, 12]} /><meshStandardMaterial color="#343836" /></mesh>)}
    </group>
  )
}

function EarthquakeAftermath({ result, dayIndex, reducedMotion, motionRate }: {
  result: CompareResponse
  dayIndex: number
  reducedMotion: boolean
  motionRate: number
}) {
  const incident = latestIncident(result, dayIndex, 4)
  if (!incident || incident.type !== 'aftershock') return null
  const center = targetCenter(result, dayIndex)
  const service = strongestShockService(incident.shock, result.services)
  const arc = recoveryArcForService(result, dayIndex, service)
  const opacity = Math.max(0.05, (1 - arc.progress) * 0.62) * (1 - incident.ageDays * 0.13)
  const aftershock = incident.ageDays > 0 && incident.ageDays <= 2
  return (
    <group name="earthquake-aftermath">
      <group position={[center[0], 0.34, center[2]]}>
        {Array.from({ length: 13 }, (_, index) => {
          const angle = (index / 13) * Math.PI * 2 + incident.shock.severity
          const radius = 1.2 + (index % 5) * 1.25
          return (
            <mesh key={index} position={[Math.cos(angle) * radius, 0, Math.sin(angle) * radius]} rotation={[0, -angle + 0.35, 0]}>
              <boxGeometry args={[1.5 + (index % 3) * 0.6, 0.025, 0.055]} />
              <meshBasicMaterial color="#584f47" transparent opacity={opacity} />
            </mesh>
          )
        })}
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.55, 0.72, 36]} />
          <meshBasicMaterial color="#8e705c" transparent opacity={0.48} />
        </mesh>
        {aftershock ? <AftershockPulse severity={incident.shock.severity} reducedMotion={reducedMotion} motionRate={motionRate} /> : null}
      </group>
      <ExclusionZone center={center} severity={incident.shock.severity} />
      <DustHaze center={center} strength={(1 - arc.progress) * incident.shock.severity} reducedMotion={reducedMotion} motionRate={motionRate} />
      {incident.ageDays === 0 ? <BirdStartle center={center} reducedMotion={reducedMotion} motionRate={motionRate} /> : null}
    </group>
  )
}

function AftershockPulse({ severity, reducedMotion, motionRate }: {
  severity: number
  reducedMotion: boolean
  motionRate: number
}) {
  const ring = useRef<Mesh>(null)
  useFrame(({ clock }) => {
    if (!ring.current) return
    const pulse = reducedMotion ? 0.72 : 0.68 + Math.sin(clock.elapsedTime * 1.4 * motionRate) * 0.08
    ring.current.scale.setScalar(pulse + severity)
  })
  return (
    <mesh ref={ring} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
      <ringGeometry args={[2.5, 2.58, 48]} />
      <meshBasicMaterial color="#a1836c" transparent opacity={0.16 + severity * 0.25} />
    </mesh>
  )
}

function ExclusionZone({ center, severity }: { center: WorldPoint; severity: number }) {
  const count = Math.max(2, Math.round(severity * 11))
  return (
    <group position={[center[0] + 2.4, 0.28, center[2] + 1.7]}>
      {Array.from({ length: count }, (_, index) => (
        <group key={index} position={[(index - (count - 1) / 2) * 0.58, 0, Math.sin(index) * 0.1]}>
          <mesh position={[0, 0.36, 0]}><boxGeometry args={[0.48, 0.1, 0.08]} /><meshStandardMaterial color="#bd8b46" /></mesh>
          <mesh position={[0, 0.16, 0]}><boxGeometry args={[0.05, 0.4, 0.06]} /><meshStandardMaterial color="#545a56" /></mesh>
        </group>
      ))}
    </group>
  )
}

function DustHaze({ center, strength, reducedMotion, motionRate }: {
  center: WorldPoint
  strength: number
  reducedMotion: boolean
  motionRate: number
}) {
  const haze = useRef<Group>(null)
  useFrame(({ clock }) => {
    if (!haze.current) return
    haze.current.children.forEach((child, index) => {
      const phase = reducedMotion ? index / Math.max(1, haze.current!.children.length) : (clock.elapsedTime * 0.055 * motionRate + index * 0.17) % 1
      child.position.y = 0.6 + phase * 2.2
      child.scale.setScalar(0.5 + phase * 0.62)
    })
  })
  if (strength <= 0.015) return null
  return (
    <group ref={haze} position={[center[0], 0.25, center[2]]}>
      {Array.from({ length: 6 }, (_, index) => (
        <mesh key={index} position={[Math.sin(index * 2.4) * 2.2, 0.6, Math.cos(index * 2.4) * 1.8]}>
          <dodecahedronGeometry args={[0.82, 0]} />
          <meshStandardMaterial color="#807b72" transparent opacity={0.06 + strength * 0.24} depthWrite={false} flatShading />
        </mesh>
      ))}
    </group>
  )
}

function BirdStartle({ center, reducedMotion, motionRate }: {
  center: WorldPoint
  reducedMotion: boolean
  motionRate: number
}) {
  const birds = useRef<Group>(null)
  useFrame(({ clock }) => {
    if (!birds.current) return
    const travel = reducedMotion ? 0.35 : Math.min(1, (clock.elapsedTime * 0.08 * motionRate) % 1)
    birds.current.position.set(center[0] + travel * 12, 5 + travel * 3, center[2] - travel * 8)
    birds.current.visible = travel < 0.9
  })
  return (
    <group ref={birds} position={[center[0], 5, center[2]]}>
      {Array.from({ length: 5 }, (_, index) => (
        <group key={index} position={[index * 0.48, Math.sin(index) * 0.3, index * 0.22]} rotation={[0.1, 0.5, 0]}>
          <mesh rotation={[0, 0, 0.45]}><boxGeometry args={[0.34, 0.025, 0.08]} /><meshBasicMaterial color="#4f5653" /></mesh>
          <mesh rotation={[0, 0, -0.45]}><boxGeometry args={[0.34, 0.025, 0.08]} /><meshBasicMaterial color="#4f5653" /></mesh>
        </group>
      ))}
    </group>
  )
}

function WeatherAftermath({ result, dayIndex, reducedMotion, motionRate }: {
  result: CompareResponse
  dayIndex: number
  reducedMotion: boolean
  motionRate: number
}) {
  const incident = latestIncident(result, dayIndex, 4)
  if (!incident || incident.type !== 'weather') return null
  const center = targetCenter(result, dayIndex)
  const strength = Math.max(0, incident.shock.severity * (1 - incident.ageDays / 5))
  const [windX, windZ] = disasterWind(result, incident.shock.day)
  const poolCount = Math.max(0, Math.round(strength * 18) - incident.ageDays)
  const debrisCount = Math.max(0, Math.round(strength * 14) - incident.ageDays * 2)
  return (
    <group name="weather-recovery-residue">
      {Array.from({ length: poolCount }, (_, index) => {
        const angle = index * 2.399
        const radius = 1.2 + (index % 5) * 1.05
        return (
          <mesh key={index} position={[center[0] + Math.cos(angle) * radius, 0.325, center[2] + Math.sin(angle) * radius]} rotation={[-Math.PI / 2, 0, angle]}>
            <circleGeometry args={[0.46 + (index % 3) * 0.18, 20]} />
            <meshStandardMaterial color="#6f9095" transparent opacity={0.24 + strength * 0.25} roughness={0.3} />
          </mesh>
        )
      })}
      {Array.from({ length: debrisCount }, (_, index) => (
        <group key={index} position={[center[0] - 3 + (index % 6) * 1.05, 0.42, center[2] + 2.2 + Math.floor(index / 6) * 0.6]} rotation={[0, Math.atan2(windX, windZ), 0]}>
          <mesh rotation={[0, 0, 0.55]}><cylinderGeometry args={[0.07, 0.1, 0.9, 7]} /><meshStandardMaterial color="#6b553f" roughness={1} /></mesh>
          <mesh position={[0.25, 0.12, 0]}><dodecahedronGeometry args={[0.28, 0]} /><meshStandardMaterial color="#526e50" roughness={1} /></mesh>
        </group>
      ))}
      <SandbagLine position={[center[0] - windX * 3.4, 0.35, center[2] - windZ * 3.4]} rotation={Math.atan2(windX, windZ)} count={5 + Math.round(strength * 9)} />
      <WetRoadSheen strength={strength} />
      <BranchCollector center={center} reducedMotion={reducedMotion} motionRate={motionRate} active={incident.ageDays > 0 && debrisCount > 0} />
    </group>
  )
}

function WetRoadSheen({ strength }: { strength: number }) {
  return (
    <group>
      <mesh position={[0, 0.322, -2]} receiveShadow><boxGeometry args={[42, 0.015, 1.15]} /><meshStandardMaterial color="#73898a" transparent opacity={strength * 0.22} roughness={0.28} /></mesh>
      <mesh position={[0, 0.323, 0.5]} rotation={[0, Math.PI / 2, 0]} receiveShadow><boxGeometry args={[38, 0.015, 1.15]} /><meshStandardMaterial color="#73898a" transparent opacity={strength * 0.18} roughness={0.28} /></mesh>
    </group>
  )
}

function SandbagLine({ position, rotation, count }: { position: WorldPoint; rotation: number; count: number }) {
  return (
    <group position={position} rotation={[0, rotation, 0]}>
      {Array.from({ length: Math.min(10, count) }, (_, index) => (
        <mesh key={index} position={[(index - Math.min(10, count) / 2) * 0.42, 0, index % 2 ? 0.16 : 0]} castShadow>
          <capsuleGeometry args={[0.13, 0.25, 3, 6]} />
          <meshStandardMaterial color="#aa9670" roughness={0.98} />
        </mesh>
      ))}
    </group>
  )
}

function BranchCollector({ center, active, reducedMotion, motionRate }: {
  center: WorldPoint
  active: boolean
  reducedMotion: boolean
  motionRate: number
}) {
  const truck = useRef<Group>(null)
  useFrame(({ clock }) => {
    if (!truck.current || !active) return
    const progress = reducedMotion ? 0.55 : (clock.elapsedTime * 0.06 * motionRate) % 1
    truck.current.position.x = center[0] - 3 + progress * 6
  })
  if (!active) return null
  return (
    <group ref={truck} position={[center[0] - 3, 0.38, center[2] + 3.3]}>
      <mesh position={[0, 0.34, 0]} castShadow><boxGeometry args={[1.1, 0.65, 1.6]} /><meshStandardMaterial color="#6c8068" roughness={0.86} /></mesh>
      <mesh position={[0, 0.5, 0.65]} castShadow><boxGeometry args={[1, 0.62, 0.45]} /><meshStandardMaterial color="#c5bfaf" roughness={0.83} /></mesh>
    </group>
  )
}

function EpidemicLogistics({ result, dayIndex }: { result: CompareResponse; dayIndex: number }) {
  const incident = latestIncident(result, dayIndex, 3)
  if (!incident || incident.type !== 'epidemic') return null
  const healthcare = serviceLevel(result, dayIndex, 'healthcare')
  const pressure = Math.max(0, incident.shock.severity * (incident.shock.impact[result.services.indexOf('healthcare')] ?? 0))
  const campus = DISTRICTS.find((district) => district.service === 'healthcare')!.center
  const tentCount = healthcare < 0.55 ? Math.max(1, Math.ceil((0.55 - healthcare) * 10)) : 0
  return (
    <group name="epidemic-response-logistics">
      {Array.from({ length: tentCount }, (_, index) => (
        <Tent key={index} position={[campus[0] - 4 + index * 1.55, 0.3, campus[2] + 4.8]} color="#78928d" scale={0.82} />
      ))}
      <GeneratorTrailer position={[campus[0] + 4.2, 0.3, campus[2] + 3.8]} />
      {['food', 'transport'].map((service, index) => {
        const district = DISTRICTS.find((item) => item.service === service as Service)!
        return (
          <group key={service} position={[district.center[0] + 3.9, 0.3, district.center[2] - 4.4]}>
            <mesh position={[0, 0.65, 0]}><boxGeometry args={[1.1, 0.62, 0.08]} /><meshStandardMaterial color="#c8c1a9" roughness={0.9} /></mesh>
            <mesh position={[0, 0.2, 0]}><boxGeometry args={[0.08, 0.9, 0.08]} /><meshStandardMaterial color="#525b57" /></mesh>
            <mesh position={[0, 0.67, 0.05]}><ringGeometry args={[0.16, 0.21, 20]} /><meshBasicMaterial color="#6b8179" /></mesh>
            <Html position={[0, 1.2, 0]} center className="world-label-anchor" zIndexRange={[12, 0]}>
              <div className="world-label compact"><b>ADVISORY</b><span>{intensityBand('epidemic', incident.shock.severity)} · {pressure.toFixed(2)}</span></div>
            </Html>
          </group>
        )
      })}
    </group>
  )
}

function SupplyLogistics({ result, dayIndex }: { result: CompareResponse; dayIndex: number }) {
  const incident = latestIncident(result, dayIndex, 3)
  if (!incident || incident.type !== 'supply') return null
  const foodDepot = CITY_DEPOTS.find((depot) => depot.service === 'food')!
  const food = serviceLevel(result, dayIndex, 'food')
  const barrierCount = Math.max(3, Math.round((1 - food) * 9))
  return (
    <group name="supply-disruption-logistics" position={foodDepot.position} rotation={[0, foodDepot.rotation, 0]}>
      {Array.from({ length: barrierCount }, (_, index) => (
        <group key={index} position={[-2.7 + index * 0.68, 0.2, 3.1 + (index % 2) * 0.38]}>
          <mesh position={[0, 0.35, 0]}><boxGeometry args={[0.55, 0.12, 0.1]} /><meshStandardMaterial color="#ac7c40" /></mesh>
          <mesh position={[0, 0.14, 0]}><boxGeometry args={[0.06, 0.42, 0.07]} /><meshStandardMaterial color="#535954" /></mesh>
        </group>
      ))}
      <Html position={[0, 3.9, 0]} center className="world-label-anchor" zIndexRange={[13, 0]}>
        <div className="world-label compact"><b>CRITICAL GOODS FIRST</b><span>{intensityBand('supply', incident.shock.severity)} · {incident.shock.severity.toFixed(2)}</span></div>
      </Html>
    </group>
  )
}

function UtilityContinuity({ result, dayIndex, reducedMotion, motionRate }: {
  result: CompareResponse
  dayIndex: number
  reducedMotion: boolean
  motionRate: number
}) {
  const incident = latestIncident(result, dayIndex, 4)
  const civic = serviceLevel(result, dayIndex, 'public_services')
  const active = incident?.type === 'utility' || civic < 0.52
  const arc = recoveryArcForService(result, dayIndex, 'public_services')
  return (
    <group name="utility-continuity-assets">
      <Substation active={active} progress={arc.progress} reducedMotion={reducedMotion} motionRate={motionRate} />
      <WaterTower level={civic} />
      {active ? (
        <>
          <GeneratorTrailer position={[12.2, 0.3, -7.6]} rotation={Math.PI / 2} />
          {CITY_DEPOTS.filter((_, index) => index % 2 === 0).map((depot, index) => (
            <GeneratorTrailer key={depot.service} position={[depot.position[0] + 2.6, 0.3, depot.position[2] - 1.8]} rotation={index * Math.PI / 2} />
          ))}
        </>
      ) : null}
    </group>
  )
}

function Substation({ active, progress, reducedMotion, motionRate }: {
  active: boolean
  progress: number
  reducedMotion: boolean
  motionRate: number
}) {
  const indicators = useRef<Group>(null)
  useFrame(({ clock }) => {
    if (!indicators.current) return
    indicators.current.children.forEach((child, index) => {
      const material = (child as Mesh).material as MeshStandardMaterial
      const stageReady = progress >= [0.28, 0.58, 0.82][index]
      const flicker = active && !stageReady && !reducedMotion ? Math.sin(clock.elapsedTime * (7 + index) * motionRate) > 0.35 : false
      material.emissiveIntensity = stageReady ? 0.5 : flicker ? 0.18 : 0
    })
  })
  return (
    <group position={CITY_SUBSTATION}>
      <mesh position={[0, 0.1, 0]} receiveShadow><boxGeometry args={[4.8, 0.16, 4.2]} /><meshStandardMaterial color="#777e78" roughness={0.98} /></mesh>
      {[-1.15, 0, 1.15].map((x) => (
        <group key={x} position={[x, 0.55, 0]}>
          <mesh castShadow><boxGeometry args={[0.58, 1.1, 0.72]} /><meshStandardMaterial color="#626f6d" roughness={0.82} /></mesh>
          <mesh position={[0, 0.72, 0]}><cylinderGeometry args={[0.2, 0.2, 0.32, 10]} /><meshStandardMaterial color="#aaa38f" roughness={0.7} /></mesh>
        </group>
      ))}
      <group ref={indicators} position={[0, 0.35, 1.5]}>
        {['power', 'water', 'signals'].map((label, index) => (
          <mesh key={label} position={[(index - 1) * 0.72, 0, 0]}>
            <boxGeometry args={[0.42, 0.22, 0.08]} />
            <meshStandardMaterial color={index === 0 ? '#d6b96e' : index === 1 ? '#7ca0a4' : '#809a78'} emissive={index === 0 ? '#d6b96e' : index === 1 ? '#7ca0a4' : '#809a78'} />
          </mesh>
        ))}
      </group>
      <Html position={[0, 2.0, 0]} center className="world-label-anchor" zIndexRange={[12, 0]}>
        <div className="world-label compact"><b>SUBSTATION</b><span>power → water → signals · {Math.round(progress * 100)}%</span></div>
      </Html>
    </group>
  )
}

function WaterTower({ level }: { level: number }) {
  return (
    <group position={CITY_WATER_TOWER}>
      {[-0.55, 0.55].flatMap((x) => [-0.55, 0.55].map((z) => (
        <mesh key={`${x}-${z}`} position={[x, 1.45, z]} rotation={[x * 0.08, 0, z * 0.08]} castShadow>
          <boxGeometry args={[0.1, 2.9, 0.1]} /><meshStandardMaterial color="#656e69" />
        </mesh>
      )))}
      <mesh position={[0, 3.1, 0]} castShadow><cylinderGeometry args={[1.05, 0.78, 1.5, 14]} /><meshStandardMaterial color="#8e9a96" roughness={0.82} /></mesh>
      <mesh position={[0, 3.1, 0.8]}><boxGeometry args={[0.18, 1.1 * level, 0.08]} /><meshStandardMaterial color="#7da1a5" roughness={0.38} /></mesh>
    </group>
  )
}

function WellbeingSignals({ result, dayIndex, fallen, reducedMotion, motionRate }: {
  result: CompareResponse
  dayIndex: number
  fallen: boolean
  reducedMotion: boolean
  motionRate: number
}) {
  const day = result.candidate.trajectory[dayIndex]
  const housing = serviceLevel(result, dayIndex, 'housing')
  const food = serviceLevel(result, dayIndex, 'food')
  const shelterCount = housing < 0.52 ? Math.min(8, Math.ceil((0.52 - housing) * 18)) : 0
  const parkedCars = Math.max(1, Math.round(housing * 8))
  const parkProps = day.resilience >= 0.68 ? Math.min(6, Math.round((day.resilience - 0.6) * 15)) : 0
  return (
    <group name="aggregate-wellbeing-signals">
      {Array.from({ length: shelterCount }, (_, index) => (
        <Tent key={index} position={[CITY_HOUSING_PARK[0] - 2.4 + (index % 4) * 1.45, 0.3, CITY_HOUSING_PARK[2] + 2.5 + Math.floor(index / 4) * 1.4]} color="#6a8794" scale={0.74} />
      ))}
      {Array.from({ length: parkedCars }, (_, index) => (
        <group key={index} position={[-23.8 + (index % 4) * 1.2, 0.36, -16.5 + Math.floor(index / 4) * 1.05]} rotation={[0, Math.PI / 2, 0]}>
          <mesh castShadow><boxGeometry args={[0.62, 0.3, 1.0]} /><meshStandardMaterial color={index % 2 ? '#a19a8c' : '#6f807e'} roughness={0.83} /></mesh>
        </group>
      ))}
      {Array.from({ length: parkProps }, (_, index) => (
        <group key={index} position={[CITY_HOUSING_PARK[0] - 2.5 + index, 0.32, CITY_HOUSING_PARK[2] - 2.4 + (index % 2) * 0.8]}>
          <mesh position={[0, 0.28, 0]}><boxGeometry args={[0.92, 0.1, 0.3]} /><meshStandardMaterial color="#8c6d4e" /></mesh>
          <mesh position={[-0.35, 0.1, 0]}><boxGeometry args={[0.08, 0.4, 0.08]} /><meshStandardMaterial color="#555a54" /></mesh>
          <mesh position={[0.35, 0.1, 0]}><boxGeometry args={[0.08, 0.4, 0.08]} /><meshStandardMaterial color="#555a54" /></mesh>
        </group>
      ))}
      {food < 0.46 ? <FieldKitchen position={[-12.2, 0.3, 4.8]} /> : null}
      <CivicFlag result={result} dayIndex={dayIndex} fallen={fallen} reducedMotion={reducedMotion} motionRate={motionRate} />
    </group>
  )
}

function FieldKitchen({ position }: { position: WorldPoint }) {
  return (
    <group position={position}>
      <Tent position={[0, 0, 0]} color="#a48254" scale={0.9} />
      {[-1.1, 1.1].map((x) => <mesh key={x} position={[x, 0.38, 0]} castShadow><boxGeometry args={[0.72, 0.65, 0.72]} /><meshStandardMaterial color="#7b8078" roughness={0.9} /></mesh>)}
      <Html position={[0, 1.7, 0]} center className="world-label-anchor" zIndexRange={[12, 0]}><div className="world-label compact"><b>FIELD KITCHEN</b><span>food deficit response</span></div></Html>
    </group>
  )
}

function CivicFlag({ result, dayIndex, fallen, reducedMotion, motionRate }: {
  result: CompareResponse
  dayIndex: number
  fallen: boolean
  reducedMotion: boolean
  motionRate: number
}) {
  const flag = useRef<Group>(null)
  const incident = latestIncident(result, dayIndex, 2)
  const wind = incident?.type === 'weather' ? disasterWind(result, incident.shock.day) : [0.9, 0.25] as const
  const raised = serviceLevel(result, dayIndex, 'public_services') >= 0.6
  useFrame(({ clock }) => {
    if (!flag.current) return
    flag.current.rotation.y = Math.atan2(wind[0], wind[1])
    flag.current.rotation.z = reducedMotion ? 0 : Math.sin(clock.elapsedTime * 1.4 * motionRate) * 0.04
  })
  const height = fallen ? 2.4 : raised ? 4.35 : 3.1
  return (
    <group position={CITY_CIVIC_FLAG}>
      <mesh position={[0, 2.4, 0]} castShadow><cylinderGeometry args={[0.045, 0.06, 4.8, 10]} /><meshStandardMaterial color="#5d6763" /></mesh>
      <group ref={flag} position={[0, height, 0]}>
        <mesh position={[0.65, 0, 0]} castShadow><boxGeometry args={[1.3, 0.65, 0.035]} /><meshStandardMaterial color={fallen ? '#6e716d' : '#71866a'} roughness={0.88} /></mesh>
      </group>
    </group>
  )
}

export function RecoveryPhenomenology({
  result,
  dayIndex,
  fallen,
  reducedMotion,
  motionRate,
}: {
  result: CompareResponse
  dayIndex: number
  fallen: boolean
  reducedMotion: boolean
  motionRate: number
}) {
  const incident = latestIncident(result, dayIndex, 4)
  return (
    <group name="typed-persistent-phenomenology">
      <EarthquakeAftermath result={result} dayIndex={dayIndex} reducedMotion={reducedMotion} motionRate={motionRate} />
      <WeatherAftermath result={result} dayIndex={dayIndex} reducedMotion={reducedMotion} motionRate={motionRate} />
      <EpidemicLogistics result={result} dayIndex={dayIndex} />
      <SupplyLogistics result={result} dayIndex={dayIndex} />
      <UtilityContinuity result={result} dayIndex={dayIndex} reducedMotion={reducedMotion} motionRate={motionRate} />
      <WellbeingSignals result={result} dayIndex={dayIndex} fallen={fallen} reducedMotion={reducedMotion} motionRate={motionRate} />
      {incident ? (
        <Html position={[targetCenter(result, dayIndex)[0], 6.7, targetCenter(result, dayIndex)[2]]} center className="world-label-anchor incident-world-label" zIndexRange={[14, 0]}>
          <div className="world-label"><b>{incident.type === 'aftershock' ? 'EARTHQUAKE' : incident.type.toUpperCase()}</b><span>{intensityBand(incident.type as ShockType, incident.shock.severity)} · raw {incident.shock.severity.toFixed(2)} · {incident.ageDays === 0 ? 'active day' : `recovery day +${incident.ageDays}`}</span></div>
        </Html>
      ) : null}
    </group>
  )
}
