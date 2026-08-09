import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { Group, InstancedMesh } from 'three'
import type { DayResult, Service } from '../types'
import { DISTRICTS, type DamageState } from './model'
import { depotStatusesForDay, type DepotStatus } from './realism'
import { CITY_DEPOTS, CITY_HUB } from './worldLayout'

const DEPOT_COLORS: Readonly<Record<Service, string>> = {
  transport: '#5a8290',
  housing: '#bd6b52',
  food: '#d49a3d',
  healthcare: '#d9ded7',
  public_services: '#71866a',
}

function PalletField({ units, sparse = false }: { units: number; sparse?: boolean }) {
  const mesh = useRef<InstancedMesh>(null)
  const visibleCount = Math.max(0, Math.min(18, Math.ceil(units / (sparse ? 8 : 5))))
  useLayoutEffect(() => {
    if (!mesh.current) return
    const matrix = new THREE.Matrix4()
    const color = new THREE.Color()
    for (let index = 0; index < visibleCount; index += 1) {
      const column = index % 6
      const row = Math.floor(index / 6)
      matrix.makeTranslation((column - 2.5) * 0.36, 0.14 + row * 0.17, (index % 2) * 0.34)
      mesh.current.setMatrixAt(index, matrix)
      color.set(index % 3 === 0 ? '#a97745' : index % 3 === 1 ? '#b98b54' : '#846849')
      mesh.current.setColorAt(index, color)
    }
    mesh.current.instanceMatrix.needsUpdate = true
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true
    mesh.current.computeBoundingSphere()
  }, [visibleCount])
  if (visibleCount === 0) return null
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, visibleCount]} castShadow>
      <boxGeometry args={[0.31, 0.18, 0.3]} />
      <meshStandardMaterial color="#ffffff" roughness={0.92} />
    </instancedMesh>
  )
}

function DamageDressing({ damage }: { damage: DamageState }) {
  if (damage === 'intact') return null
  return (
    <group>
      {damage === 'slight' ? (
        <group position={[0.85, 1.25, 1.48]} rotation={[0, 0, 0.5]}>
          <mesh><boxGeometry args={[0.045, 0.82, 0.05]} /><meshStandardMaterial color="#4e4943" /></mesh>
          <mesh position={[0.18, -0.32, 0]} rotation={[0, 0, -1]}><boxGeometry args={[0.045, 0.48, 0.05]} /><meshStandardMaterial color="#4e4943" /></mesh>
        </group>
      ) : null}
      {damage === 'moderate' ? (
        <>
          <mesh position={[0.55, 2.26, -0.18]} rotation={[0, 0, -0.05]} castShadow>
            <boxGeometry args={[2.7, 0.08, 2.25]} />
            <meshStandardMaterial color="#557c93" roughness={0.92} />
          </mesh>
          {[-0.72, 0, 0.72].map((x) => (
            <mesh key={x} position={[x, 0.82, 1.49]}>
              <boxGeometry args={[0.52, 0.78, 0.07]} />
              <meshStandardMaterial color="#6c6152" roughness={0.98} />
            </mesh>
          ))}
        </>
      ) : null}
      {damage === 'rubble' ? (
        <group>
          {Array.from({ length: 14 }, (_, index) => {
            const angle = index * 2.399
            const radius = 0.35 + (index % 5) * 0.32
            return (
              <mesh key={index} position={[Math.cos(angle) * radius, 0.16 + (index % 3) * 0.14, Math.sin(angle) * radius]} rotation={[index * 0.2, angle, index * 0.13]} castShadow>
                <boxGeometry args={[0.55, 0.24, 0.34]} />
                <meshStandardMaterial color={index % 4 ? '#887d70' : '#5d7180'} roughness={0.96} />
              </mesh>
            )
          })}
          <mesh position={[0.2, 0.48, -0.1]} rotation={[0.2, 0.35, 0.08]} castShadow>
            <boxGeometry args={[2.7, 0.1, 1.9]} />
            <meshStandardMaterial color="#557c93" roughness={0.94} />
          </mesh>
        </group>
      ) : null}
    </group>
  )
}

function Forklift({ active, reducedMotion, motionRate }: {
  active: boolean
  reducedMotion: boolean
  motionRate: number
}) {
  const vehicle = useRef<Group>(null)
  const elapsed = useRef(0)
  useFrame((_, delta) => {
    if (!vehicle.current) return
    if (!active || reducedMotion) return
    elapsed.current += Math.min(delta, 0.05) * motionRate
    const progress = (elapsed.current * 0.11) % 1
    const dwell = progress < 0.2 || progress > 0.82
    const travel = clampForklift((progress - 0.2) / 0.62)
    vehicle.current.position.x = -1.15 + travel * 2.3
    vehicle.current.rotation.y = progress > 0.82 ? Math.PI : 0
    if (dwell) vehicle.current.position.x = progress < 0.2 ? -1.15 : 1.15
  })
  return (
    <group ref={vehicle} position={[-1.15, 0.18, 1.95]} scale={0.58}>
      <mesh position={[0, 0.28, 0]} castShadow><boxGeometry args={[0.72, 0.48, 0.82]} /><meshStandardMaterial color="#b9873f" roughness={0.82} /></mesh>
      <mesh position={[0, 0.72, -0.18]} castShadow><boxGeometry args={[0.62, 0.08, 0.72]} /><meshStandardMaterial color="#574f44" /></mesh>
      {[-0.3, 0.3].map((x) => <mesh key={x} position={[x, 0.42, 0.28]}><boxGeometry args={[0.06, 0.9, 0.06]} /><meshStandardMaterial color="#4e514e" /></mesh>)}
      <mesh position={[0, 0.08, 0.76]}><boxGeometry args={[0.9, 0.06, 0.08]} /><meshStandardMaterial color="#454947" /></mesh>
    </group>
  )
}

function clampForklift(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function QueueBarriers({ count }: { count: number }) {
  return (
    <group position={[0, 0.23, 2.62]}>
      {Array.from({ length: Math.min(5, count) }, (_, index) => (
        <group key={index} position={[-1.65 + index * 0.78, 0, index % 2 ? 0.32 : 0]}>
          <mesh position={[0, 0.3, 0]} castShadow><boxGeometry args={[0.68, 0.12, 0.11]} /><meshStandardMaterial color="#bd8d45" /></mesh>
          {[-0.27, 0.27].map((x) => <mesh key={x} position={[x, 0.14, 0]}><boxGeometry args={[0.07, 0.3, 0.08]} /><meshStandardMaterial color="#565a55" /></mesh>)}
        </group>
      ))}
    </group>
  )
}

function Depot({ status, selected, onSelect, reducedMotion, motionRate }: {
  status: DepotStatus
  selected: boolean
  onSelect?: (service: Service) => void
  reducedMotion: boolean
  motionRate: number
}) {
  const layout = CITY_DEPOTS.find((item) => item.service === status.service)!
  const [hovered, setHovered] = useState(false)
  const standing = status.damage !== 'rubble'
  const bodyHeight = status.damage === 'moderate' ? 1.45 : 2.05
  const placardColor = status.placard === 'GREEN' ? '#5d8265' : status.placard === 'YELLOW' ? '#c19b4c' : '#9c5f55'
  return (
    <group
      position={layout.position}
      rotation={[0, layout.rotation, 0]}
      onClick={(event) => { event.stopPropagation(); onSelect?.(status.service) }}
      onPointerOver={(event) => { event.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer' }}
      onPointerOut={() => { setHovered(false); document.body.style.cursor = '' }}
    >
      <mesh position={[0, -0.06, 0]} receiveShadow>
        <boxGeometry args={[4.5, 0.16, 4.2]} />
        <meshStandardMaterial color={selected ? '#d5c9ae' : '#8b9088'} roughness={0.98} />
      </mesh>
      {standing ? (
        <>
          <mesh position={[0, bodyHeight / 2, 0]} castShadow receiveShadow>
            <boxGeometry args={[3.55, bodyHeight, 2.9]} />
            <meshStandardMaterial color={status.damage === 'moderate' ? '#777d77' : DEPOT_COLORS[status.service]} roughness={0.87} />
          </mesh>
          <mesh position={[0, bodyHeight + 0.13, 0]} castShadow>
            <boxGeometry args={[3.85, 0.24, 3.15]} />
            <meshStandardMaterial color="#59615d" roughness={0.9} />
          </mesh>
          {[-1.05, 0, 1.05].map((x) => (
            <mesh key={x} position={[x, 0.72, 1.48]}>
              <boxGeometry args={[0.76, 1.12, 0.08]} />
              <meshStandardMaterial color={status.damage === 'moderate' ? '#5e5c57' : '#3f4a4a'} roughness={0.68} />
            </mesh>
          ))}
          <group position={[-1.5, bodyHeight + 0.32, -1.12]}>
            <PalletField
              units={status.palletUnits}
              sparse={status.damage !== 'intact' || status.palletUnits < status.stockCapacity * 0.2}
            />
          </group>
        </>
      ) : null}
      <DamageDressing damage={status.damage} />
      <Forklift
        active={standing
          && status.throughputSignal > 0.42
          && (!status.recorded || status.recorded.landedUnits + status.recorded.repairDispatchUnits > 1e-7)}
        reducedMotion={reducedMotion}
        motionRate={motionRate}
      />
      <QueueBarriers count={status.dockQueue} />
      <mesh position={[1.55, 0.72, 1.54]}>
        <boxGeometry args={[0.45, 0.62, 0.08]} />
        <meshStandardMaterial color={placardColor} roughness={0.9} />
      </mesh>
      {(hovered || selected) ? (
        <Html position={[0, 3.35, 0]} center className="scene-inspector-anchor" zIndexRange={[35, 0]}>
          <div className="scene-entity-card" data-placard={status.placard.toLowerCase()}>
            <b>{DISTRICTS.find((item) => item.service === status.service)?.shortLabel} point of distribution</b>
            <span>{status.placard} placard · {status.damage} condition</span>
            <small>{status.engineTruth
              ? `${status.palletUnits.toFixed(1)} / ${status.stockCapacity.toFixed(0)} units in recorded stock`
              : `${status.palletUnits.toFixed(1)} pallets = ${status.allocationUnits.toFixed(1)} assigned units`}</small>
            {status.recorded ? <small>{status.recorded.landedUnits.toFixed(1)} units landed · {status.recorded.repairSupplyUnits.toFixed(1)} effective repair supply</small> : null}
            {status.recorded ? <small>{status.pendingUnits.toFixed(1)} units queued · {Math.round(status.throughputSignal * 100)}% effective throughput</small> : null}
            {status.recorded && status.recorded.damagePenalty > 1e-7 ? <small>Damage penalty {status.recorded.damagePenalty.toFixed(2)} · {status.recorded.damageDaysRemaining} days remaining</small> : null}
            {status.spoilageUnits > 0.0001 ? <small>{status.spoilageUnits.toFixed(2)} food units expired in the recorded ledger</small> : null}
            {status.reroutedFrom ? <em>Mutual aid route from {DISTRICTS.find((item) => item.service === status.reroutedFrom)?.shortLabel}</em> : null}
          </div>
        </Html>
      ) : null}
    </group>
  )
}

function FuelPoint() {
  return (
    <group position={CITY_HUB.fuelPoint}>
      <mesh position={[0, 0.42, 0]} castShadow><cylinderGeometry args={[0.65, 0.65, 0.84, 14]} /><meshStandardMaterial color="#6d756f" roughness={0.9} /></mesh>
      <mesh position={[0, 1.05, 0]} castShadow><cylinderGeometry args={[0.48, 0.65, 0.3, 14]} /><meshStandardMaterial color="#59635f" roughness={0.9} /></mesh>
      <mesh position={[1.0, 0.54, 0]} castShadow><boxGeometry args={[0.34, 1.08, 0.42]} /><meshStandardMaterial color="#c4b287" roughness={0.86} /></mesh>
      <mesh position={[1.0, 1.16, 0]}><boxGeometry args={[0.55, 0.22, 0.08]} /><meshStandardMaterial color="#404946" /></mesh>
    </group>
  )
}

function StagedVehicles() {
  return (
    <group position={CITY_HUB.staging}>
      {Array.from({ length: 4 }, (_, index) => (
        <group key={index} position={[(index % 2) * 1.05, 0.16, Math.floor(index / 2) * 0.92]} rotation={[0, Math.PI / 2, 0]}>
          <mesh position={[0, 0.22, 0]} castShadow><boxGeometry args={[0.72, 0.38, 1.05]} /><meshStandardMaterial color="#c9c3b5" roughness={0.84} /></mesh>
          <mesh position={[0, 0.34, 0.42]} castShadow><boxGeometry args={[0.66, 0.42, 0.35]} /><meshStandardMaterial color="#5d706f" roughness={0.78} /></mesh>
        </group>
      ))}
    </group>
  )
}

function IntakeHub({ day }: { day: DayResult }) {
  return (
    <group position={CITY_HUB.position}>
      <mesh position={[0, -0.07, 0]} receiveShadow>
        <boxGeometry args={[8.7, 0.2, 6.5]} />
        <meshStandardMaterial color="#727b74" roughness={0.96} />
      </mesh>
      <mesh position={[0, 1.35, 0]} castShadow receiveShadow>
        <boxGeometry args={[5.6, 2.7, 4.0]} />
        <meshStandardMaterial color="#b7ac98" roughness={0.88} />
      </mesh>
      <mesh position={[0, 2.82, 0]} castShadow>
        <boxGeometry args={[6.05, 0.25, 4.45]} />
        <meshStandardMaterial color="#58615d" roughness={0.91} />
      </mesh>
      {[-1.8, -0.6, 0.6, 1.8].map((x) => (
        <group key={x} position={[x, 0.78, -2.05]}>
          <mesh><boxGeometry args={[0.82, 1.15, 0.1]} /><meshStandardMaterial color="#3c4747" roughness={0.7} /></mesh>
          <mesh position={[0, -0.68, -0.34]} receiveShadow><boxGeometry args={[1.05, 0.18, 0.78]} /><meshStandardMaterial color="#838981" roughness={0.97} /></mesh>
        </group>
      ))}
      <group position={[-2.15, 0.18, 2.4]}><PalletField units={day.available_budget} /></group>
      <FuelPoint />
      <StagedVehicles />
      <Html position={[0, 3.35, 0]} center className="world-label-anchor" zIndexRange={[18, 0]}>
        <div className="world-label"><b>INTAKE HUB</b><span>{day.available_budget.toFixed(1)} supply units received</span></div>
      </Html>
    </group>
  )
}

export function DepotNetwork({
  day,
  services,
  selectedService,
  onSelectService,
  reducedMotion,
  motionRate,
}: {
  day: DayResult
  services: readonly Service[]
  selectedService: Service | null
  onSelectService?: (service: Service) => void
  reducedMotion: boolean
  motionRate: number
}) {
  const statuses = useMemo(() => depotStatusesForDay(day, services), [day, services])
  return (
    <group name="distributed-logistics-network">
      <IntakeHub day={day} />
      {statuses.map((status) => (
        <Depot
          key={status.service}
          status={status}
          selected={selectedService === status.service}
          onSelect={onSelectService}
          reducedMotion={reducedMotion}
          motionRate={motionRate}
        />
      ))}
    </group>
  )
}
