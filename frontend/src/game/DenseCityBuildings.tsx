import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { InstancedMesh } from 'three'
import type { Service } from '../types'
import type { DamageState, DistrictDefinition } from './model'
import type { Placard, RecoveryStage } from './realism'
import type { CityBuildingArchetype, CityBuildingPlacement } from './worldLayout'

export type ArchetypeSpec = {
  size: readonly [number, number, number]
  capScale: readonly [number, number, number]
  capOffset: readonly [number, number, number]
}

export const ARCHETYPE_SPECS: Readonly<Record<CityBuildingArchetype, ArchetypeSpec>> = {
  apartment: { size: [1.7, 3.2, 1.55], capScale: [1.3, 0.35, 1.2], capOffset: [0, 3.38, 0] },
  rowhouse: { size: [2.05, 1.65, 1.5], capScale: [1.55, 0.35, 1.05], capOffset: [0.18, 1.84, 0] },
  office: { size: [1.7, 3.8, 1.65], capScale: [1.25, 0.45, 1.2], capOffset: [-0.14, 4.02, 0.08] },
  hospital: { size: [2.55, 1.65, 1.85], capScale: [1.65, 0.32, 1.25], capOffset: [0, 1.86, 0] },
  market: { size: [2.45, 1.35, 1.85], capScale: [2.05, 0.3, 1.35], capOffset: [0, 1.57, 0] },
  warehouse: { size: [2.55, 1.5, 1.75], capScale: [1.85, 0.38, 1.15], capOffset: [-0.18, 1.75, 0] },
  transit: { size: [2.8, 1.25, 1.55], capScale: [2.3, 0.22, 1.05], capOffset: [0, 1.48, 0] },
  civic: { size: [2.35, 2.25, 1.75], capScale: [1.65, 0.36, 1.2], capOffset: [0, 2.48, 0] },
}

/** Three landmarks per district retain the richer authored prefab treatment. */
export const DETAILED_BUILDING_INDICES = new Set([0, 17, 35])

export function isDetailedBuilding(buildingIndex: number): boolean {
  return DETAILED_BUILDING_INDICES.has(buildingIndex)
}

type VisualBuilding = {
  placement: CityBuildingPlacement
  state: DamageState
  rebuilding: boolean
  placard?: Placard
  repairProgress?: number
  recoveryStage?: RecoveryStage
  rebuiltVariant?: boolean
  freshness?: number
}

type BoxInstance = {
  position: readonly [number, number, number]
  rotation: readonly [number, number, number]
  scale: readonly [number, number, number]
  color: string
}

function mixBodyColor(body: string, state: DamageState, freshness = 0): string {
  const color = new THREE.Color(body)
  if (state === 'moderate') color.lerp(new THREE.Color('#77736b'), 0.42)
  if (state === 'slight') color.lerp(new THREE.Color('#8d8579'), 0.16)
  if (state === 'intact' && freshness > 0) color.lerp(new THREE.Color('#dfd4c1'), freshness * 0.12)
  return `#${color.getHexString()}`
}

function transformInstance(
  placement: CityBuildingPlacement,
  localPosition: readonly [number, number, number],
  localScale: readonly [number, number, number],
  color: string,
  localRotation: readonly [number, number, number] = [0, 0, 0],
): BoxInstance {
  const rootQuaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, placement.rotation, 0))
  const localQuaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(...localRotation))
  const local = new THREE.Vector3(...localPosition).multiplyScalar(placement.scale).applyQuaternion(rootQuaternion)
  return {
    position: [
      placement.position[0] + local.x,
      placement.position[1] + local.y,
      placement.position[2] + local.z,
    ],
    rotation: new THREE.Euler().setFromQuaternion(rootQuaternion.multiply(localQuaternion)).toArray().slice(0, 3) as [number, number, number],
    scale: [localScale[0] * placement.scale, localScale[1] * placement.scale, localScale[2] * placement.scale],
    color,
  }
}

function InstanceBatch({
  instances,
  geometry = 'box',
  roughness = 0.84,
  castShadow = true,
  receiveShadow = false,
}: {
  instances: readonly BoxInstance[]
  geometry?: 'box' | 'stud'
  roughness?: number
  castShadow?: boolean
  receiveShadow?: boolean
}) {
  const mesh = useRef<InstancedMesh>(null)
  const capacity = Math.max(1, instances.length)
  const instanceColors = useMemo(() => new Float32Array(capacity * 3), [capacity])
  useLayoutEffect(() => {
    if (!mesh.current) return
    const matrix = new THREE.Matrix4()
    const position = new THREE.Vector3()
    const quaternion = new THREE.Quaternion()
    const scale = new THREE.Vector3()
    const euler = new THREE.Euler()
    instances.forEach((instance, index) => {
      position.set(...instance.position)
      euler.set(...instance.rotation)
      quaternion.setFromEuler(euler)
      scale.set(...instance.scale)
      matrix.compose(position, quaternion, scale)
      mesh.current!.setMatrixAt(index, matrix)
      mesh.current!.setColorAt(index, new THREE.Color(instance.color))
    })
    mesh.current.count = instances.length
    mesh.current.instanceMatrix.needsUpdate = true
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true
    mesh.current.computeBoundingSphere()
  }, [instances])

  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, capacity]}
      castShadow={castShadow}
      receiveShadow={receiveShadow}
    >
      <instancedBufferAttribute attach="instanceColor" args={[instanceColors, 3]} />
      {geometry === 'stud'
        ? <cylinderGeometry args={[0.15, 0.16, 0.12, 10]} />
        : <boxGeometry args={[1, 1, 1]} />}
      <meshStandardMaterial color="#ffffff" roughness={roughness} />
    </instancedMesh>
  )
}

function rubbleInstances(building: VisualBuilding, color: string): BoxInstance[] {
  const amount = building.state === 'rubble' ? 8 : building.state === 'moderate' ? 3 : 0
  return Array.from({ length: amount }, (_, index) => {
    const angle = index * 2.399 + building.placement.buildingIndex * 0.41
    const radius = 0.28 + (index % 4) * 0.27
    return transformInstance(
      building.placement,
      [Math.cos(angle) * radius, 0.1 + (index % 3) * 0.1, Math.sin(angle) * radius],
      [0.5, 0.22, 0.28],
      index % 4 === 0 ? '#716a62' : color,
      [(index % 3) * 0.22, angle, (index % 4) * 0.17],
    )
  })
}

function scaffoldInstances(building: VisualBuilding): BoxInstance[] {
  if (!building.rebuilding) return []
  const height = Math.max(1.25, ARCHETYPE_SPECS[building.placement.archetype].size[1] * 0.82)
  const stage = building.recoveryStage ?? 'active-work'
  const frames = [
    ...([-0.88, 0.88] as const).flatMap((x) => ([-0.72, 0.72] as const).map((z) => (
      transformInstance(building.placement, [x, height / 2, z], [0.07, height, 0.07], '#aa7a35')
    ))),
    transformInstance(building.placement, [0, height * 0.42, 0.72], [1.82, 0.06, 0.06], '#c19248'),
    transformInstance(building.placement, [0.88, height * 0.67, 0], [0.06, 0.06, 1.52], '#c19248'),
  ]
  if (stage === 'frame') return frames.slice(0, 4)
  return frames
}

export function DenseDistrictInfill({
  buildings,
  district,
  castShadows = true,
}: {
  buildings: readonly VisualBuilding[]
  district: DistrictDefinition
  castShadows?: boolean
}) {
  const batches = useMemo(() => {
    const foundations: BoxInstance[] = []
    const bodies: BoxInstance[] = []
    const caps: BoxInstance[] = []
    const windows: BoxInstance[] = []
    const studs: BoxInstance[] = []
    const rubble: BoxInstance[] = []
    const scaffolds: BoxInstance[] = []
    const tarps: BoxInstance[] = []
    const boards: BoxInstance[] = []
    const placards: BoxInstance[] = []
    const fences: BoxInstance[] = []
    const wraps: BoxInstance[] = []

    buildings.forEach((building) => {
      const { placement, state } = building
      const spec = ARCHETYPE_SPECS[placement.archetype]
      const [width, height, depth] = spec.size
      const bodyColor = mixBodyColor(district.body, state, building.freshness)
      const shellScaleY = state === 'moderate' ? 0.67 : 1
      const lean = state === 'slight' ? 0.025 : 0

      foundations.push(transformInstance(placement, [0, -0.04, 0], [2.9, 0.18, 2.35], '#8f968d'))
      rubble.push(...rubbleInstances(building, bodyColor))
      scaffolds.push(...scaffoldInstances(building))
      const placardColor = building.placard === 'GREEN'
        ? '#5d8265'
        : building.placard === 'YELLOW'
          ? '#c19b4c'
          : '#9c5f55'
      placards.push(transformInstance(placement, [width * 0.35, 0.62, depth / 2 + 0.06], [0.22, 0.32, 0.04], placardColor))
      if (state === 'moderate') {
        tarps.push(transformInstance(placement, [0.15, height * 0.68, 0], [width * 0.88, 0.06, depth * 0.9], '#557c93', [0, 0, -0.06]))
        boards.push(transformInstance(placement, [0, Math.min(0.82, height * 0.45), depth / 2 + 0.07], [width * 0.68, 0.48, 0.05], '#6c6152'))
      }
      if (building.rebuilding) {
        ;([-1, 1] as const).forEach((side) => {
          fences.push(transformInstance(placement, [side * 1.25, 0.42, 0], [0.06, 0.78, 2.25], '#8e784f'))
          fences.push(transformInstance(placement, [0, 0.42, side * 1.05], [2.55, 0.78, 0.06], '#8e784f'))
        })
        if (building.recoveryStage === 'wrap' || building.recoveryStage === 'active-work') {
          wraps.push(transformInstance(placement, [0, height * 0.52, depth / 2 + 0.11], [width * 1.08, height * 0.92, 0.035], '#d8d2c3'))
        }
      }
      if (state === 'rubble') return

      bodies.push(transformInstance(
        placement,
        [0, height * shellScaleY / 2, 0],
        [width, height * shellScaleY, depth],
        bodyColor,
        [0, 0, lean],
      ))
      caps.push(transformInstance(
        placement,
        [spec.capOffset[0], spec.capOffset[1] * shellScaleY, spec.capOffset[2]],
        [spec.capScale[0] * (building.rebuiltVariant ? 0.86 : 1), spec.capScale[1] * shellScaleY, spec.capScale[2] * (building.rebuiltVariant ? 1.12 : 1)],
        district.accent,
        [0, 0, lean],
      ))
      windows.push(transformInstance(
        placement,
        [0, Math.min(height * 0.55, 1.55) * shellScaleY, depth / 2 + 0.026],
        [width * 0.7, 0.26 * shellScaleY, 0.045],
        '#34454a',
        [0, 0, lean],
      ))
      if (height > 2.6) {
        windows.push(transformInstance(
          placement,
          [0, height * 0.76 * shellScaleY, depth / 2 + 0.026],
          [width * 0.7, 0.24 * shellScaleY, 0.045],
          '#34454a',
          [0, 0, lean],
        ))
      }
      const studY = (spec.capOffset[1] + spec.capScale[1] / 2 + 0.06) * shellScaleY
      const spread = Math.min(0.42, spec.capScale[0] * 0.18)
      studs.push(transformInstance(placement, [-spread, studY, 0], [1, 1, 1], district.accent, [0, 0, lean]))
      studs.push(transformInstance(placement, [spread, studY, 0], [1, 1, 1], district.accent, [0, 0, lean]))
    })

    return { foundations, bodies, caps, windows, studs, rubble, scaffolds, tarps, boards, placards, fences, wraps }
  }, [buildings, district.accent, district.body])

  return (
    <group name={`dense-${district.service satisfies Service}`}>
      <InstanceBatch instances={batches.foundations} roughness={0.95} receiveShadow castShadow={false} />
      <InstanceBatch instances={batches.bodies} castShadow={castShadows} />
      <InstanceBatch instances={batches.caps} roughness={0.76} castShadow={castShadows} />
      <InstanceBatch instances={batches.windows} roughness={0.55} castShadow={false} />
      <InstanceBatch instances={batches.studs} geometry="stud" roughness={0.78} castShadow={castShadows} />
      <InstanceBatch instances={batches.rubble} roughness={0.94} castShadow={castShadows} />
      <InstanceBatch instances={batches.scaffolds} roughness={0.82} castShadow={castShadows} />
      <InstanceBatch instances={batches.tarps} roughness={0.92} castShadow={castShadows} />
      <InstanceBatch instances={batches.boards} roughness={0.96} castShadow={castShadows} />
      <InstanceBatch instances={batches.placards} roughness={0.86} castShadow={false} />
      <InstanceBatch instances={batches.fences} roughness={0.9} castShadow={castShadows} />
      <InstanceBatch instances={batches.wraps} roughness={0.76} castShadow={false} />
    </group>
  )
}
