import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { Group, InstancedMesh } from 'three'
import type { DayResult, Service } from '../types'
import { DISTRICTS, type DamageState } from './model'
import { presentationMotionPhase } from './presentationMotion'
import { depotStatusesForDay, type DepotStatus } from './realism'
import { MAX_VISIBLE_ROAD_VEHICLES } from './VehicleFleet'
import { CITY_DEPOTS, CITY_HUB } from './worldLayout'

const DEPOT_COLORS: Readonly<Record<Service, string>> = {
  transport: '#5a8290',
  housing: '#bd6b52',
  food: '#d49a3d',
  healthcare: '#d9ded7',
  public_services: '#71866a',
}

export type PalletFieldPlan = {
  instanceCount: number
  finalPalletFraction: number
}

export const IN_WORLD_INTERPOLATION_DISCLOSURE = 'Visual interpolation between returned day states · exact daily values in Analyst Toolbox'

export type DepotPresentationCopy = {
  condition: string
  stock: string
  flow: string | null
  queue: string | null
  damage: string | null
  spoilage: string | null
  reroute: string | null
  mutualAid: string | null
}

/**
 * Copy for the animated in-world card. DepotStatus may be built from the
 * presentation sampler's visual DayResult, so none of these midpoint values
 * are described as recorded observations. The exact returned rows remain in
 * Analyst Toolbox.
 */
export function depotPresentationCopy(status: DepotStatus): DepotPresentationCopy {
  const hasOperations = Boolean(status.recorded)
  return {
    condition: status.engineTruth
      ? `${status.damage} condition · visual interpolation`
      : 'Depot condition unavailable',
    stock: status.engineTruth
      ? `${(status.palletUnits ?? 0).toFixed(1)} / ${(status.stockCapacity ?? 0).toFixed(0)} supply units · visual stock interpolation`
      : `${status.allocationUnits.toFixed(1)} assigned units · stock unavailable`,
    flow: hasOperations
      ? `Visual interpolation: ${status.recorded!.landedUnits.toFixed(1)} units landed · ${status.recorded!.repairSupplyUnits.toFixed(1)} effective repair supply`
      : null,
    queue: hasOperations
      ? `Visual interpolation: ${(status.dockQueueUnits ?? 0).toFixed(1)} units constrained at docks · ${(status.pendingUnits ?? 0).toFixed(1)} scheduled/held next day · ${Math.round((status.throughputSignal ?? 0) * 100)}% effective throughput`
      : null,
    damage: hasOperations && status.recorded!.damagePenalty > 1e-7
      ? `Visual interpolation: damage penalty ${status.recorded!.damagePenalty.toFixed(2)} · ${status.recorded!.damageDaysRemaining} days remaining`
      : null,
    spoilage: hasOperations && (status.spoilageUnits ?? 0) > 0.0001
      ? `Visual interpolation: ${(status.spoilageUnits ?? 0).toFixed(2)} food units expired in storage`
      : null,
    reroute: status.reroutedFrom
      ? `Nearest-healthy presentation route from ${DISTRICTS.find((item) => item.service === status.reroutedFrom)?.shortLabel} · triggered by local depot rubble in the visual interpolation`
      : null,
    mutualAid: status.mutualAidFrom
      ? `Visual interpolation: mutual aid from ${DISTRICTS.find((item) => item.service === status.mutualAidFrom)?.shortLabel}`
      : null,
  }
}

export function intakeHubPresentationCopy(availableBudget: number): string {
  return `${availableBudget.toFixed(1)} daily supply units · visual budget interpolation · exact daily value in Analyst Toolbox`
}

/**
 * One rendered pallet-equivalent represents exactly one returned supply unit.
 * Daily supply is validated by the API, so there is deliberately no visual cap:
 * the accepted 500-unit maximum renders 500 instances in one draw call. A
 * fractional final unit is a proportionally narrow final pallet, never a whole
 * invented pallet.
 */
export function palletFieldPlan(units: number): PalletFieldPlan {
  const normalizedUnits = Number.isFinite(units) ? Math.max(0, units) : 0
  const instanceCount = Math.ceil(normalizedUnits)
  return {
    instanceCount,
    finalPalletFraction: instanceCount === 0 ? 0 : normalizedUnits - (instanceCount - 1),
  }
}

function PalletField({ units, sparse = false, capacity = 500 }: { units: number; sparse?: boolean; capacity?: number }) {
  const mesh = useRef<InstancedMesh>(null)
  const instanceCapacity = Math.max(1, Math.ceil(capacity))
  const plan = palletFieldPlan(units)
  const visibleCount = Math.min(plan.instanceCount, instanceCapacity)
  const finalPalletFraction = plan.finalPalletFraction
  useLayoutEffect(() => {
    if (!mesh.current) return
    const matrix = new THREE.Matrix4()
    const quaternion = new THREE.Quaternion()
    const position = new THREE.Vector3()
    const scale = new THREE.Vector3()
    const color = new THREE.Color()
    const columns = sparse ? 8 : 10
    const rows = 5
    const spacing = sparse ? 0.29 : 0.24
    for (let index = 0; index < visibleCount; index += 1) {
      const column = index % columns
      const row = Math.floor(index / columns) % rows
      const layer = Math.floor(index / (columns * rows))
      const fractional = index === visibleCount - 1 ? finalPalletFraction : 1
      position.set(
        (column - (columns - 1) / 2) * spacing,
        0.05 + layer * 0.075,
        (row - (rows - 1) / 2) * spacing,
      )
      scale.set(fractional, 1, 1)
      matrix.compose(position, quaternion, scale)
      mesh.current.setMatrixAt(index, matrix)
      color.set(index % 3 === 0 ? '#a97745' : index % 3 === 1 ? '#b98b54' : '#846849')
      mesh.current.setColorAt(index, color)
    }
    mesh.current.count = visibleCount
    mesh.current.instanceMatrix.needsUpdate = true
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true
    mesh.current.computeBoundingSphere()
  }, [finalPalletFraction, sparse, visibleCount])
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, instanceCapacity]} castShadow visible={visibleCount > 0}>
      <boxGeometry args={[0.2, 0.055, 0.18]} />
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

function Forklift({ active, reducedMotion, presentationTime }: {
  active: boolean
  reducedMotion: boolean
  presentationTime: number
}) {
  const vehicle = useRef<Group>(null)
  useFrame(() => {
    if (!vehicle.current) return
    if (!active || reducedMotion) return
    const progress = presentationMotionPhase(presentationTime, 0.11)
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

function Depot({ status, selected, onSelect, reducedMotion, presentationTime, operationsVisible }: {
  status: DepotStatus
  selected: boolean
  onSelect?: (service: Service) => void
  reducedMotion: boolean
  presentationTime: number
  operationsVisible: boolean
}) {
  const layout = CITY_DEPOTS.find((item) => item.service === status.service)!
  const [hovered, setHovered] = useState(false)
  const standing = status.damage !== 'rubble'
  const bodyHeight = status.damage === 'moderate' ? 1.45 : 2.05
  const placardColor = status.placard === 'GREEN' ? '#5d8265' : status.placard === 'YELLOW' ? '#c19b4c' : '#9c5f55'
  const copy = depotPresentationCopy(status)
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
            <meshStandardMaterial color={status.damage === null ? '#777c78' : status.damage === 'moderate' ? '#777d77' : DEPOT_COLORS[status.service]} roughness={0.87} />
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
          {status.recorded && operationsVisible ? <group position={[-1.5, bodyHeight + 0.32, -1.12]}>
            <PalletField
              units={status.palletUnits ?? 0}
              capacity={status.stockCapacity ?? 500}
              sparse={status.damage !== 'intact' || (status.palletUnits ?? 0) < (status.stockCapacity ?? 0) * 0.2}
            />
          </group> : null}
        </>
      ) : null}
      {status.damage ? <DamageDressing damage={status.damage} /> : null}
      <Forklift
        active={Boolean(operationsVisible
          && status.recorded
          && standing
          && (status.throughputSignal ?? 0) > 0.42
          && status.recorded.landedUnits + status.recorded.repairDispatchUnits > 1e-7)}
        reducedMotion={reducedMotion}
        presentationTime={presentationTime}
      />
      <QueueBarriers count={operationsVisible ? status.dockQueue ?? 0 : 0} />
      {operationsVisible && status.placard ? <mesh position={[1.55, 0.72, 1.54]}>
        <boxGeometry args={[0.45, 0.62, 0.08]} />
        <meshStandardMaterial color={placardColor} roughness={0.9} />
      </mesh> : null}
      {(hovered || selected) ? (
        <Html position={[0, 3.35, 0]} center className="scene-inspector-anchor" zIndexRange={[35, 0]}>
          <div className="scene-entity-card" data-placard={status.placard?.toLowerCase() ?? 'unavailable'}>
            <b>{DISTRICTS.find((item) => item.service === status.service)?.shortLabel} point of distribution</b>
            <span>{copy.condition}</span>
            {status.engineTruth ? <small>{IN_WORLD_INTERPOLATION_DISCLOSURE}</small> : null}
            {!operationsVisible ? <small>Damage assessment in progress · end-of-day stock, dispatch, queue, and repair ledger withheld until RESPONSE</small> : null}
            {operationsVisible ? <small>{copy.stock}</small> : null}
            {operationsVisible && copy.flow ? <small>{copy.flow}</small> : null}
            {operationsVisible && copy.queue ? <small>{copy.queue}</small> : null}
            {operationsVisible && copy.damage ? <small>{copy.damage}</small> : null}
            {operationsVisible && copy.spoilage ? <small>{copy.spoilage}</small> : null}
            {operationsVisible && copy.reroute ? <em>{copy.reroute}</em> : null}
            {operationsVisible && copy.mutualAid ? <em>{copy.mutualAid}</em> : null}
            {!status.recorded ? <small>{status.source}</small> : null}
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
      <Html position={[0, 3.35, 0]} center className="world-label-anchor" zIndexRange={[18, 0]}>
        <div className="world-label"><b>INTAKE HUB</b><span>{intakeHubPresentationCopy(day.available_budget)} · one rendered pallet per unit · road view limited to {MAX_VISIBLE_ROAD_VEHICLES} mission slots; complete manifest in Toolbox</span></div>
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
  presentationTime,
  operationsVisible = true,
}: {
  day: DayResult
  services: readonly Service[]
  selectedService: Service | null
  onSelectService?: (service: Service) => void
  reducedMotion: boolean
  presentationTime: number
  operationsVisible?: boolean
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
          presentationTime={presentationTime}
          operationsVisible={operationsVisible}
        />
      ))}
    </group>
  )
}
