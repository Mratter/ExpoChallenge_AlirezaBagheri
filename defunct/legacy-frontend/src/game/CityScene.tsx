import { Html, Line, OrbitControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { Group, InstancedMesh, Mesh } from 'three'
import type { CompareResponse, Service } from '../types'
import type { PresentationSample } from './presentation'
import {
  DISTRICTS,
  damageStateFor,
  relayOrbNarration,
  serviceIndex,
  type BuildingArchetype,
  type CityImpactEvent,
  type DamageState,
  type DistrictDefinition,
} from './model'
import { repairPresentationCursor, SceneEffects } from './SceneEffects'
import { ARCHETYPE_SPECS, DenseDistrictInfill } from './DenseCityBuildings'
import { DisasterPresentation } from './DisasterEffects'
import { DepotNetwork } from './DepotNetwork'
import { InfrastructureScene } from './InfrastructureScene'
import { RecoveryPhenomenology } from './RecoveryPhenomenology'
import { VehicleFleet, type VehicleDockDwell, type VehicleFleetMode } from './VehicleFleet'
import {
  presentationMotionPhase,
  presentationMotionTime,
  repairPresentationMotionTime,
  repairShiftChangeBlend,
} from './presentationMotion'
import {
  buildingPlacard,
  buildingRepairStartedAt,
  latestIncidentOfType,
  recoveryArcForService,
  repairFreshnessForBuildingAt,
  repairProgressForBuildingAt,
  repairStageForBuildingAt,
  strongestShockService,
  type IncidentPhase,
  type Placard,
  type RecoveryStage,
} from './realism'
import { usesDetailedBuilding, type RenderQualityProfile } from './renderQuality'
import { CRITICAL_FLOOR } from './stakes'
import {
  CITY_CAMERA_TARGET_Y,
  cityCameraContainedDistance,
  cityCameraFraming,
  cityCameraRequiredDistance,
  cityCameraVerticalFov,
} from './cameraFraming'
import {
  CITY_AIM_RING_RADIUS,
  CITY_BUILDING_PLACEMENTS,
  CITY_CAMERA_LAYOUT,
  CITY_DEPOTS,
  CITY_DISTRICT_PAD_SIZE,
  CITY_LANE_MARKERS,
  CITY_PLATE,
  CITY_ROAD_SEGMENTS,
  CITY_TABLETOP,
  CITY_TREE_PLACEMENTS,
} from './worldLayout'

const CAMERA_TARGET: [number, number, number] = [0, CITY_CAMERA_TARGET_Y, 0]
const MIN_POLAR_ANGLE = 0.42
const MAX_POLAR_ANGLE = 1.25

export function stagedRepairDamageState(
  initial: DamageState,
  current: DamageState,
  assigned: boolean,
  progress: number,
): DamageState {
  if (!assigned || initial === 'intact') return current
  if (progress >= 0.995) return 'intact'
  if (initial === 'rubble') return progress < 0.34 ? 'rubble' : progress < 0.72 ? 'moderate' : 'slight'
  if (initial === 'moderate') return progress < 0.58 ? 'moderate' : 'slight'
  return 'slight'
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReduced(query.matches)
    update()
    query.addEventListener?.('change', update)
    return () => query.removeEventListener?.('change', update)
  }, [])
  return reduced
}

function CityCamera() {
  const controls = useRef<React.ElementRef<typeof OrbitControls>>(null)
  const initialized = useRef(false)
  const { camera, size } = useThree()
  const target = useMemo(() => new THREE.Vector3(...CAMERA_TARGET), [])
  const offset = useMemo(() => new THREE.Vector3(), [])
  const spherical = useMemo(() => new THREE.Spherical(), [])
  const fov = cityCameraVerticalFov(size.width, size.height)
  const framing = useMemo(() => cityCameraFraming({
    width: size.width,
    height: size.height,
    verticalFovDegrees: fov,
    minPolarAngle: MIN_POLAR_ANGLE,
    maxPolarAngle: MAX_POLAR_ANGLE,
  }), [fov, size.height, size.width])

  const enforceContainment = useCallback((addInitialBreathingRoom = false) => {
    offset.copy(camera.position).sub(target)
    spherical.setFromVector3(offset)
    const containmentInput = {
      width: size.width,
      height: size.height,
      verticalFovDegrees: fov,
      polarAngle: spherical.phi,
      azimuthAngle: spherical.theta,
    }
    const poseMinimum = Math.max(
      framing.minDistance,
      cityCameraRequiredDistance(containmentInput),
    )
    if (controls.current) controls.current.minDistance = poseMinimum
    const safeDistance = cityCameraContainedDistance(
      Math.max(offset.length(), framing.minDistance),
      containmentInput,
    )
    const desiredDistance = addInitialBreathingRoom ? safeDistance * 1.035 : safeDistance
    if (offset.length() + 0.001 < desiredDistance) {
      camera.position.copy(target).add(offset.normalize().multiplyScalar(desiredDistance))
      camera.updateMatrixWorld()
    }
  }, [camera, fov, framing.minDistance, offset, size.height, size.width, spherical, target])

  useLayoutEffect(() => {
    if (camera instanceof THREE.PerspectiveCamera) camera.fov = fov
    camera.far = framing.far
    camera.updateProjectionMatrix()
    if (!initialized.current && size.width < size.height) {
      offset.copy(camera.position).sub(target)
      spherical.setFromVector3(offset)
      spherical.phi = CITY_CAMERA_LAYOUT.portraitDefaultPolarAngle
      spherical.theta = CITY_CAMERA_LAYOUT.portraitDefaultAzimuthAngle
      camera.position.copy(target).add(offset.setFromSpherical(spherical))
      camera.lookAt(target)
      camera.updateMatrixWorld()
    }
    enforceContainment(!initialized.current)
    if (controls.current) {
      controls.current.target.copy(target)
      controls.current.update()
    }
    initialized.current = true
  }, [camera, enforceContainment, fov, framing.far, target])

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.065}
      enablePan={false}
      minDistance={framing.minDistance}
      maxDistance={framing.maxDistance}
      minPolarAngle={MIN_POLAR_ANGLE}
      maxPolarAngle={MAX_POLAR_ANGLE}
      target={CAMERA_TARGET}
      onChange={() => enforceContainment(false)}
    />
  )
}

type BuildingPlacement = {
  service: Service
  buildingIndex: number
  archetype: BuildingArchetype
  position: [number, number, number]
  rotation: number
  scale: number
}

function buildingPlacements(district: DistrictDefinition): BuildingPlacement[] {
  return CITY_BUILDING_PLACEMENTS
    .filter((placement) => placement.service === district.service)
    .map((placement) => ({
      service: placement.service,
      buildingIndex: placement.buildingIndex,
      archetype: placement.archetype,
      position: [...placement.position] as [number, number, number],
      rotation: placement.rotation,
      scale: placement.scale,
    }))
}

function BaseplateStuds() {
  const mesh = useRef<InstancedMesh>(null)
  const columns = CITY_PLATE.studColumns
  const rows = CITY_PLATE.studRows
  useLayoutEffect(() => {
    if (!mesh.current) return
    const matrix = new THREE.Matrix4()
    let index = 0
    for (let x = 0; x < columns; x += 1) {
      for (let z = 0; z < rows; z += 1) {
        matrix.makeTranslation(
          x - (columns - 1) / 2,
          CITY_PLATE.studPositionY,
          z - (rows - 1) / 2,
        )
        mesh.current.setMatrixAt(index, matrix)
        index += 1
      }
    }
    mesh.current.instanceMatrix.needsUpdate = true
  }, [])
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, columns * rows]} receiveShadow>
      <cylinderGeometry args={[0.14, 0.15, CITY_PLATE.studHeight, 10]} />
      <meshStandardMaterial color="#71816d" roughness={0.83} />
    </instancedMesh>
  )
}

function Baseplate() {
  return (
    <group>
      <mesh position={[0, CITY_PLATE.positionY, 0]} receiveShadow castShadow>
        <boxGeometry args={[CITY_PLATE.width, CITY_PLATE.height, CITY_PLATE.depth]} />
        <meshStandardMaterial color="#687767" roughness={0.9} />
      </mesh>
      <BaseplateStuds />
      <mesh position={[0, CITY_PLATE.borderPositionY, 0]} receiveShadow>
        <boxGeometry args={[CITY_PLATE.borderWidth, CITY_PLATE.borderHeight, CITY_PLATE.borderDepth]} />
        <meshStandardMaterial color="#46534a" roughness={0.92} />
      </mesh>
      <RoadNetwork />
    </group>
  )
}

function RoadNetwork() {
  const roads = useRef<InstancedMesh>(null)
  const lanes = useRef<InstancedMesh>(null)
  useLayoutEffect(() => {
    const matrix = new THREE.Matrix4()
    const position = new THREE.Vector3()
    const quaternion = new THREE.Quaternion()
    const scale = new THREE.Vector3()
    const apply = (mesh: InstancedMesh | null, items: typeof CITY_ROAD_SEGMENTS) => {
      if (!mesh) return
      items.forEach((item, index) => {
        position.set(...item.position)
        quaternion.setFromEuler(new THREE.Euler(0, item.rotation, 0))
        scale.set(...item.scale)
        matrix.compose(position, quaternion, scale)
        mesh.setMatrixAt(index, matrix)
      })
      mesh.instanceMatrix.needsUpdate = true
      mesh.computeBoundingSphere()
    }
    apply(roads.current, CITY_ROAD_SEGMENTS)
    apply(lanes.current, CITY_LANE_MARKERS)
  }, [])
  return (
    <group>
      <instancedMesh ref={roads} args={[undefined, undefined, CITY_ROAD_SEGMENTS.length]} receiveShadow>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#555b59" roughness={0.97} />
      </instancedMesh>
      <instancedMesh ref={lanes} args={[undefined, undefined, CITY_LANE_MARKERS.length]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial color="#d7cba9" />
      </instancedMesh>
    </group>
  )
}

function CityTrees({ weatherStrength, reducedMotion, motionTime, quality }: {
  weatherStrength: number
  reducedMotion: boolean
  motionTime: number
  quality: RenderQualityProfile
}) {
  const trunks = useRef<InstancedMesh>(null)
  const canopies = useRef<InstancedMesh>(null)
  const treeWorld = useRef<Group>(null)
  const placements = useMemo(
    () => CITY_TREE_PLACEMENTS.filter((_, index) => index % quality.treeStride === 0),
    [quality.treeStride],
  )
  useLayoutEffect(() => {
    const matrix = new THREE.Matrix4()
    const position = new THREE.Vector3()
    const quaternion = new THREE.Quaternion()
    const scale = new THREE.Vector3()
    placements.forEach((tree, index) => {
      quaternion.setFromEuler(new THREE.Euler(0, tree.rotation, 0))
      position.set(tree.position[0], tree.position[1] + 0.55 * tree.scale, tree.position[2])
      scale.set(0.18 * tree.scale, tree.scale, 0.18 * tree.scale)
      matrix.compose(position, quaternion, scale)
      trunks.current?.setMatrixAt(index, matrix)
      position.set(tree.position[0], tree.position[1] + 1.35 * tree.scale, tree.position[2])
      scale.setScalar(0.62 * tree.scale)
      matrix.compose(position, quaternion, scale)
      canopies.current?.setMatrixAt(index, matrix)
    })
    for (const mesh of [trunks.current, canopies.current]) {
      if (!mesh) continue
      mesh.instanceMatrix.needsUpdate = true
      mesh.computeBoundingSphere()
    }
  }, [placements])
  useFrame(() => {
    if (!treeWorld.current) return
    const sway = reducedMotion ? 0 : Math.sin(motionTime * (0.45 + weatherStrength * 1.4)) * (0.0015 + weatherStrength * 0.008)
    treeWorld.current.rotation.x = sway
    treeWorld.current.rotation.z = sway * 0.72
  })
  return (
    <group ref={treeWorld}>
      <instancedMesh ref={trunks} args={[undefined, undefined, placements.length]} castShadow={quality.smallObjectShadows}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#6c4f37" roughness={1} />
      </instancedMesh>
      <instancedMesh ref={canopies} args={[undefined, undefined, placements.length]} castShadow={quality.smallObjectShadows}>
        <dodecahedronGeometry args={[1, 0]} />
        <meshStandardMaterial color="#587255" roughness={0.95} flatShading />
      </instancedMesh>
    </group>
  )
}

function RoofStuds({ width, depth, height, color }: { width: number; depth: number; height: number; color: string }) {
  const countX = Math.max(1, Math.min(3, Math.round(width / 0.72)))
  const countZ = Math.max(1, Math.min(2, Math.round(depth / 0.78)))
  const positions = []
  for (let x = 0; x < countX; x += 1) {
    for (let z = 0; z < countZ; z += 1) {
      positions.push([
        (x - (countX - 1) / 2) * 0.56,
        height,
        (z - (countZ - 1) / 2) * 0.56,
      ] as [number, number, number])
    }
  }
  return positions.map((position, index) => (
    <mesh key={index} position={position} castShadow>
      <cylinderGeometry args={[0.13, 0.14, 0.12, 12]} />
      <meshStandardMaterial color={color} roughness={0.78} />
    </mesh>
  ))
}

function WindowBand({ width, y, z, count = 3 }: { width: number; y: number; z: number; count?: number }) {
  return (
    <group>
      {Array.from({ length: count }, (_, index) => (
        <mesh key={index} position={[(index - (count - 1) / 2) * (width / count), y, z]}>
          <boxGeometry args={[Math.min(0.34, width / (count + 1)), 0.28, 0.035]} />
          <meshStandardMaterial color="#34454a" roughness={0.55} />
        </mesh>
      ))}
    </group>
  )
}

function HospitalMark({ z }: { z: number }) {
  return (
    <group position={[0, 1.05, z]}>
      <mesh><boxGeometry args={[0.13, 0.62, 0.045]} /><meshStandardMaterial color="#f1eee5" roughness={0.8} /></mesh>
      <mesh position={[0.3, 0, 0]}><boxGeometry args={[0.13, 0.62, 0.045]} /><meshStandardMaterial color="#f1eee5" roughness={0.8} /></mesh>
      <mesh position={[0.15, 0, 0]}><boxGeometry args={[0.3, 0.12, 0.05]} /><meshStandardMaterial color="#f1eee5" roughness={0.8} /></mesh>
    </group>
  )
}

function ArchetypeShell({ archetype, color, accent, state }: {
  archetype: BuildingArchetype
  color: string
  accent: string
  state: DamageState
}) {
  const spec = ARCHETYPE_SPECS[archetype]
  const [width, height, depth] = spec.size
  const shellScaleY = state === 'moderate' ? 0.67 : 1
  const lean = state === 'slight' ? 0.025 : 0
  const tallMidStoryDamage = state === 'moderate' && height >= 3
  const lowRiseCornerDamage = state === 'moderate' && height < 3
  return (
    <group rotation={[0, 0, lean]} scale={[1, shellScaleY, 1]}>
      <mesh position={[0, height / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[width, height, depth]} />
        <meshStandardMaterial color={color} roughness={0.8} />
      </mesh>
      <mesh position={spec.capOffset} castShadow>
        <boxGeometry args={spec.capScale} />
        <meshStandardMaterial color={accent} roughness={0.76} />
      </mesh>
      <RoofStuds width={spec.capScale[0]} depth={spec.capScale[2]} height={spec.capOffset[1] + spec.capScale[1] / 2 + 0.06} color={accent} />
      <WindowBand width={width * 0.76} y={Math.min(height * 0.55, 1.55)} z={depth / 2 + 0.025} count={archetype === 'rowhouse' ? 2 : 3} />
      {height > 2.6 ? <WindowBand width={width * 0.76} y={height * 0.76} z={depth / 2 + 0.025} /> : null}
      {archetype === 'hospital' ? <HospitalMark z={depth / 2 + 0.055} /> : null}
      {archetype === 'market' ? (
        <mesh position={[0, 0.52, depth / 2 + 0.22]} castShadow>
          <boxGeometry args={[width * 0.84, 0.13, 0.42]} />
          <meshStandardMaterial color={accent} roughness={0.75} />
        </mesh>
      ) : null}
      {archetype === 'transit' ? (
        <group>
          {[-0.85, 0, 0.85].map((x) => (
            <mesh key={x} position={[x, 0.57, depth / 2 + 0.34]} castShadow>
              <boxGeometry args={[0.1, 1.05, 0.1]} />
              <meshStandardMaterial color="#d9d5c9" roughness={0.8} />
            </mesh>
          ))}
          <mesh position={[0, 1.12, depth / 2 + 0.34]} castShadow>
            <boxGeometry args={[2.15, 0.12, 0.56]} />
            <meshStandardMaterial color={accent} roughness={0.78} />
          </mesh>
        </group>
      ) : null}
      {archetype === 'civic' ? (
        <group>
          {[-0.63, -0.21, 0.21, 0.63].map((x) => (
            <mesh key={x} position={[x, 0.65, depth / 2 + 0.12]} castShadow>
              <boxGeometry args={[0.11, 1.08, 0.11]} />
              <meshStandardMaterial color="#ddd8ca" roughness={0.86} />
            </mesh>
          ))}
          <mesh position={[0, 0.1, depth / 2 + 0.22]} receiveShadow>
            <boxGeometry args={[1.9, 0.18, 0.48]} />
            <meshStandardMaterial color="#c9c2b3" roughness={0.9} />
          </mesh>
        </group>
      ) : null}
      {state === 'slight' ? (
        <group position={[width * 0.18, height * 0.72, depth / 2 + 0.045]} rotation={[0, 0, 0.6]}>
          <mesh><boxGeometry args={[0.035, 0.52, 0.04]} /><meshStandardMaterial color="#514b45" /></mesh>
          <mesh position={[0.1, -0.22, 0]} rotation={[0, 0, -1.0]}><boxGeometry args={[0.035, 0.3, 0.04]} /><meshStandardMaterial color="#514b45" /></mesh>
        </group>
      ) : null}
      {tallMidStoryDamage ? (
        <group position={[0, height * 0.53, depth / 2 + 0.055]}>
          <mesh rotation={[0, 0, 0.62]}><boxGeometry args={[0.05, width * 0.72, 0.05]} /><meshStandardMaterial color="#4d4944" /></mesh>
          <mesh position={[0.35, -0.18, 0]} rotation={[0, 0, -0.72]}><boxGeometry args={[0.045, width * 0.44, 0.05]} /><meshStandardMaterial color="#4d4944" /></mesh>
        </group>
      ) : null}
      {lowRiseCornerDamage ? (
        <group position={[width * 0.42, 0.15, depth * 0.36]}>
          {[0, 1, 2, 3].map((index) => (
            <mesh key={index} position={[(index % 2) * 0.28, (index % 3) * 0.12, Math.floor(index / 2) * 0.22]} rotation={[index * 0.2, index * 0.75, 0.12]}>
              <boxGeometry args={[0.42, 0.2, 0.28]} /><meshStandardMaterial color="#746b61" roughness={0.95} />
            </mesh>
          ))}
        </group>
      ) : null}
    </group>
  )
}

function RubbleBrickPool({ color, amount = 10 }: { color: string; amount?: number }) {
  const mesh = useRef<InstancedMesh>(null)
  useLayoutEffect(() => {
    if (!mesh.current) return
    const matrix = new THREE.Matrix4()
    const position = new THREE.Vector3()
    const rotation = new THREE.Quaternion()
    const scale = new THREE.Vector3(1, 1, 1)
    const euler = new THREE.Euler()
    for (let index = 0; index < amount; index += 1) {
      const angle = index * 2.399
      const radius = 0.25 + (index % 4) * 0.22
      position.set(Math.cos(angle) * radius, 0.12 + (index % 3) * 0.11, Math.sin(angle) * radius)
      euler.set((index % 3) * 0.22, angle, (index % 4) * 0.17)
      rotation.setFromEuler(euler)
      matrix.compose(position, rotation, scale)
      mesh.current.setMatrixAt(index, matrix)
      mesh.current.setColorAt(index, new THREE.Color(index % 4 === 0 ? '#716a62' : color))
    }
    mesh.current.instanceMatrix.needsUpdate = true
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true
  }, [amount, color])
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, amount]} castShadow>
      <boxGeometry args={[0.5, 0.22, 0.28]} />
      <meshStandardMaterial color="#ffffff" roughness={0.94} />
    </instancedMesh>
  )
}

function ScaffoldCrane({ height, accent }: { height: number; accent: string }) {
  const scaffoldHeight = Math.max(1.4, height * 0.84)
  return (
    <group>
      {[-0.95, 0.95].flatMap((x) => [-0.82, 0.82].map((z) => (
        <mesh key={`${x}-${z}`} position={[x, scaffoldHeight / 2, z]} castShadow>
          <boxGeometry args={[0.07, scaffoldHeight, 0.07]} />
          <meshStandardMaterial color="#aa7a35" roughness={0.82} />
        </mesh>
      )))}
      {[0.65, 1.35, 2.05].filter((y) => y < scaffoldHeight).map((y) => (
        <group key={y} position={[0, y, 0]}>
          <mesh><boxGeometry args={[2.0, 0.06, 0.06]} /><meshStandardMaterial color="#c19248" /></mesh>
          <mesh><boxGeometry args={[0.06, 0.06, 1.7]} /><meshStandardMaterial color="#c19248" /></mesh>
        </group>
      ))}
      <group position={[1.2, 0, -0.8]}>
        <mesh position={[0, 1.65, 0]} castShadow><boxGeometry args={[0.11, 3.3, 0.11]} /><meshStandardMaterial color={accent} roughness={0.76} /></mesh>
        <mesh position={[-0.38, 3.26, 0]} castShadow><boxGeometry args={[1.0, 0.1, 0.1]} /><meshStandardMaterial color={accent} roughness={0.76} /></mesh>
        <mesh position={[-0.82, 2.72, 0]}><boxGeometry args={[0.035, 1.05, 0.035]} /><meshStandardMaterial color="#5d5548" /></mesh>
      </group>
    </group>
  )
}

function SmallLift({ accent }: { accent: string }) {
  return (
    <group position={[1.1, 0.1, -0.75]}>
      <mesh position={[0, 0.28, 0]} castShadow><boxGeometry args={[0.82, 0.46, 1.05]} /><meshStandardMaterial color="#686d66" roughness={0.88} /></mesh>
      <mesh position={[0, 1.0, 0]} rotation={[0.2, 0, -0.16]} castShadow><boxGeometry args={[0.12, 1.45, 0.12]} /><meshStandardMaterial color={accent} roughness={0.8} /></mesh>
      <mesh position={[-0.16, 1.72, 0]}><boxGeometry args={[0.58, 0.12, 0.48]} /><meshStandardMaterial color={accent} roughness={0.8} /></mesh>
    </group>
  )
}

function BuildingRecoveryDressing({
  state,
  placard,
  stage,
  rebuilding,
  height,
  accent,
}: {
  state: DamageState
  placard: Placard | null
  stage: RecoveryStage
  rebuilding: boolean
  height: number
  accent: string
}) {
  const placardColor = placard === 'GREEN' ? '#5d8265' : placard === 'YELLOW' ? '#c19b4c' : '#9c5f55'
  // Placards, boards, tarps, and damage remain state-bound. Every crew object is
  // additionally gated by today's realized repair cohort in District below.
  const active = rebuilding
  return (
    <group>
      {placard ? (
        <mesh position={[1.12, 0.68, 1.22]}>
          <boxGeometry args={[0.32, 0.46, 0.05]} />
          <meshStandardMaterial color={placardColor} roughness={0.88} />
        </mesh>
      ) : null}
      {(state === 'moderate' || state === 'slight') ? (
        <>
          {[-0.5, 0.12, 0.7].map((x) => (
            <mesh key={x} position={[x, 0.8, 1.24]} rotation={[0, 0, x * 0.08]}>
              <boxGeometry args={[0.46, 0.09, 0.05]} />
              <meshStandardMaterial color="#6e604f" roughness={0.96} />
            </mesh>
          ))}
          {state === 'moderate' ? (
            <mesh position={[0.1, height * 0.7, 0]} rotation={[0, 0, -0.06]} castShadow>
              <boxGeometry args={[2.15, 0.08, 1.85]} />
              <meshStandardMaterial color="#557c93" roughness={0.94} />
            </mesh>
          ) : null}
        </>
      ) : null}
      {active ? (
        <>
          {[-1.28, 1.28].flatMap((x) => [-1.08, 1.08].map((z) => (
            <mesh key={`${x}-${z}`} position={[x, 0.42, z]}><boxGeometry args={[0.06, 0.82, 0.06]} /><meshStandardMaterial color="#8e784f" /></mesh>
          )))}
          {stage === 'wrap' || stage === 'active-work' ? (
            <mesh position={[0, height * 0.53, 1.33]}>
              <boxGeometry args={[2.65, Math.max(1.2, height * 0.86), 0.04]} />
              <meshStandardMaterial color="#d7d1c3" transparent opacity={0.68} roughness={0.78} />
            </mesh>
          ) : null}
          <group position={[1.72, 0.17, -1.18]} rotation={[0, -0.42, 0]}>
            <mesh position={[0, 0.25, 0]} castShadow><boxGeometry args={[0.82, 0.4, 1.18]} /><meshStandardMaterial color="#66736d" roughness={0.88} /></mesh>
            <mesh position={[0, 0.43, 0.38]}><boxGeometry args={[0.7, 0.32, 0.38]} /><meshStandardMaterial color={accent} roughness={0.82} /></mesh>
            {[-0.38, 0.38].flatMap((x) => [-0.38, 0.38].map((z) => (
              <mesh key={`${x}-${z}`} position={[x, 0.09, z]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.12, 0.12, 0.08, 10]} /><meshStandardMaterial color="#303633" roughness={0.92} /></mesh>
            )))}
          </group>
          {[-1.75, -1.35, -0.95].map((x) => (
            <mesh key={x} position={[x, 0.18, 1.42]} castShadow><coneGeometry args={[0.13, 0.36, 8]} /><meshStandardMaterial color="#b57d45" roughness={0.9} /></mesh>
          ))}
        </>
      ) : null}
      {active && placard === 'RED' && (stage === 'assessment' || stage === 'debris') ? (
        <group position={[-1.3, 0.14, -0.9]}>
          <mesh position={[0, 0.32, 0]} castShadow><boxGeometry args={[0.86, 0.58, 1.15]} /><meshStandardMaterial color="#c09449" roughness={0.86} /></mesh>
          <mesh position={[0.22, 0.82, 0.18]} rotation={[0.1, 0, -0.48]} castShadow><boxGeometry args={[0.14, 1.28, 0.14]} /><meshStandardMaterial color="#8a6c3f" /></mesh>
          {stage === 'debris' ? (
            <group position={[1.25, 0.1, 0.75]} rotation={[0, 0.4, 0]}>
              <mesh position={[0, 0.28, 0]} castShadow><boxGeometry args={[0.78, 0.48, 1.12]} /><meshStandardMaterial color="#766d63" roughness={0.92} /></mesh>
              <mesh position={[0, 0.56, -0.18]}><boxGeometry args={[0.7, 0.34, 0.62]} /><meshStandardMaterial color="#9b7d48" roughness={0.88} /></mesh>
            </group>
          ) : null}
        </group>
      ) : null}
      {active && stage !== 'assessment' && stage !== 'debris'
        ? (height >= 3 ? <ScaffoldCrane height={height} accent={accent} /> : <SmallLift accent={accent} />)
        : null}
    </group>
  )
}

function Building({ placement, district, state, rebuilding, placard, repairProgress, recoveryStage, rebuiltVariant, freshness }: {
  placement: BuildingPlacement
  district: DistrictDefinition
  state: DamageState
  rebuilding: boolean
  placard: Placard | null
  repairProgress: number
  recoveryStage: RecoveryStage
  rebuiltVariant: boolean
  freshness: number
}) {
  const spec = ARCHETYPE_SPECS[placement.archetype]
  const [hovered, setHovered] = useState(false)
  const bodyColor = useMemo(() => {
    const color = new THREE.Color(district.body)
    if (state === 'moderate') color.lerp(new THREE.Color('#77736b'), 0.42)
    if (state === 'slight') color.lerp(new THREE.Color('#8d8579'), 0.16)
    if (state === 'intact' && freshness > 0) color.lerp(new THREE.Color('#dfd4c1'), freshness * 0.12)
    return `#${color.getHexString()}`
  }, [district.body, freshness, state])
  return (
    <group
      position={placement.position}
      rotation={[0, placement.rotation, 0]}
      scale={placement.scale}
      onPointerOver={(event) => { event.stopPropagation(); setHovered(true); document.body.style.cursor = 'help' }}
      onPointerOut={() => { setHovered(false); document.body.style.cursor = '' }}
    >
      <mesh position={[0, -0.04, 0]} receiveShadow>
        <boxGeometry args={[2.9, 0.18, 2.35]} />
        <meshStandardMaterial color="#8f968d" roughness={0.95} />
      </mesh>
      {state === 'rubble' ? <RubbleBrickPool color={bodyColor} amount={12} /> : (
        <>
          <group scale={rebuiltVariant ? [0.92, 1.08, 1.08] : [1, 1, 1]}>
            <ArchetypeShell archetype={placement.archetype} color={bodyColor} accent={district.accent} state={state} />
          </group>
          {rebuiltVariant ? (
            <group position={[0, spec.size[1] * 0.72, 0]}>
              <mesh position={[0.62, 0.1, 0]} castShadow>
                <boxGeometry args={[0.78, Math.max(0.7, spec.size[1] * 0.42), 1.34]} />
                <meshStandardMaterial color={district.accent} roughness={0.86} />
              </mesh>
              <mesh position={[-0.3, spec.size[1] * 0.28, 0]} castShadow>
                <boxGeometry args={[1.18, 0.34, 1.08]} />
                <meshStandardMaterial color={bodyColor} roughness={0.9} />
              </mesh>
            </group>
          ) : null}
          {state === 'moderate' ? <RubbleBrickPool color={bodyColor} amount={5} /> : null}
        </>
      )}
      <BuildingRecoveryDressing
        state={state}
        placard={placard}
        stage={recoveryStage}
        rebuilding={rebuilding}
        height={spec.size[1]}
        accent={district.accent}
      />
      {hovered ? (
        <Html position={[0, spec.size[1] + 1.2, 0]} center className="scene-inspector-anchor" zIndexRange={[38, 0]}>
          <div className="scene-entity-card" data-placard={placard?.toLowerCase() ?? 'unassessed'}>
            <b>{district.shortLabel} building {placement.buildingIndex + 1}</b>
            <span>{state} damage · {placard ? `${placard} placard` : 'assessment pending'}</span>
            <small>Repair arc {Math.round(repairProgress * 100)}% · {recoveryStage.replace('-', ' ')}</small>
            <em>{rebuilding ? (spec.size[1] >= 3 ? 'heavy recovery crew assigned' : 'light recovery crew assigned') : 'no active crew this day'}</em>
          </div>
        </Html>
      ) : null}
    </group>
  )
}

function District({ district, result, dayIndex, dayProgress, presentationLevel, presentationStartLevel, presentationEndLevel, placardsVisible, aimed, dark, darkBlend, selected, onSelect, responseEnabled, quality }: {
  district: DistrictDefinition
  result: CompareResponse
  dayIndex: number
  dayProgress: number
  presentationLevel: number
  presentationStartLevel: number
  presentationEndLevel: number
  placardsVisible: boolean
  aimed: boolean
  dark: boolean
  darkBlend: number
  selected: boolean
  onSelect?: (service: Service) => void
  responseEnabled: boolean
  quality: RenderQualityProfile
}) {
  const level = presentationLevel
  const placements = useMemo(() => buildingPlacements(district), [district])
  const arc = useMemo(
    () => recoveryArcForService(result, dayIndex, district.service),
    [dayIndex, district.service, result],
  )
  const depotIndex = CITY_DEPOTS.find((depot) => depot.service === district.service)?.reservedBuildingIndex
  const presentedRepairCursor = useMemo(
    () => repairPresentationCursor(dayIndex, dayProgress, responseEnabled),
    [dayIndex, dayProgress, responseEnabled],
  )
  const visualDistrict = useMemo(() => {
    if (darkBlend <= 0.001) return district
    const accent = new THREE.Color(district.accent).lerp(new THREE.Color('#747b76'), darkBlend)
    const body = new THREE.Color(district.body).lerp(new THREE.Color('#6d736e'), darkBlend)
    return {
      ...district,
      accent: `#${accent.getHexString()}`,
      body: `#${body.getHexString()}`,
    }
  }, [darkBlend, district])
  const buildings = useMemo(() => placements.map((placement) => {
    const repairProgress = presentedRepairCursor
      ? repairProgressForBuildingAt(
          result,
          presentedRepairCursor.dayIndex,
          presentedRepairCursor.progress,
          district.service,
          placement.buildingIndex,
        )
      : 0
    const assigned = Boolean(presentedRepairCursor && buildingRepairStartedAt(
      result,
      presentedRepairCursor.dayIndex,
      presentedRepairCursor.progress,
      district.service,
      placement.buildingIndex,
    ))
    const initialState = damageStateFor(arc.lowLevel, placement.buildingIndex)
    const currentState = damageStateFor(level, placement.buildingIndex)
    const dayStartState = damageStateFor(presentationStartLevel, placement.buildingIndex)
    const dayEndState = damageStateFor(presentationEndLevel, placement.buildingIndex)
    const stateTravel = Math.abs(presentationEndLevel - presentationStartLevel) <= 1e-9
      ? 1
      : THREE.MathUtils.clamp(
          (level - presentationStartLevel) / (presentationEndLevel - presentationStartLevel),
          0,
          1,
        )
    const transitionScale = dayStartState === dayEndState
      ? 1
      : 0.94 + Math.abs(stateTravel * 2 - 1) * 0.06
    return {
      placement: transitionScale >= 0.999
        ? placement
        : { ...placement, scale: placement.scale * transitionScale },
      state: (dark && darkBlend >= 0.995
        ? 'rubble'
        : stagedRepairDamageState(initialState, currentState, assigned, repairProgress)) as DamageState,
      placard: placardsVisible ? buildingPlacard(level, placement.buildingIndex) : null,
      repairProgress,
      recoveryStage: presentedRepairCursor
        ? repairStageForBuildingAt(
            result,
            presentedRepairCursor.dayIndex,
            presentedRepairCursor.progress,
            district.service,
            placement.buildingIndex,
          )
        : 'assessment' as RecoveryStage,
      rebuiltVariant: assigned
        && initialState === 'rubble'
        && repairProgress > 0.985,
      freshness: presentedRepairCursor
        ? repairFreshnessForBuildingAt(
            result,
            presentedRepairCursor.dayIndex,
            presentedRepairCursor.progress,
            district.service,
            placement.buildingIndex,
          )
        : 0,
      rebuilding: darkBlend < 0.995
        && assigned
        && repairProgress < 0.995,
    }
  }).filter(({ placement }) => placement.buildingIndex !== depotIndex), [arc.lowLevel, dark, darkBlend, depotIndex, district.service, level, placardsVisible, placements, presentationEndLevel, presentationStartLevel, presentedRepairCursor, result])
  return (
    <group>
      <mesh
        position={[district.center[0], 0.035, district.center[2]]}
        receiveShadow
        onClick={(event) => { event.stopPropagation(); onSelect?.(district.service) }}
      >
        <boxGeometry args={CITY_DISTRICT_PAD_SIZE} />
        <meshStandardMaterial color={visualDistrict.accent} opacity={aimed ? 0.52 : selected ? 0.34 : 0.2 + darkBlend * 0.14} transparent roughness={1} />
      </mesh>
      {aimed ? (
        <mesh position={[district.center[0], 0.31, district.center[2]]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[CITY_AIM_RING_RADIUS - 0.1, CITY_AIM_RING_RADIUS, 64]} />
          <meshBasicMaterial color="#f3eee1" transparent opacity={0.9} depthWrite={false} />
        </mesh>
      ) : null}
      <DenseDistrictInfill
        buildings={buildings.filter(({ placement }) => !usesDetailedBuilding(placement.buildingIndex, quality.detailTier))}
        district={visualDistrict}
        castShadows={quality.smallObjectShadows}
      />
      {buildings.filter(({ placement }) => usesDetailedBuilding(placement.buildingIndex, quality.detailTier)).map((building) => (
        <Building
          key={`${district.service}-${building.placement.buildingIndex}`}
          placement={building.placement}
          district={visualDistrict}
           state={building.state}
           rebuilding={building.rebuilding}
           placard={building.placard}
           repairProgress={building.repairProgress}
           recoveryStage={building.recoveryStage}
           rebuiltVariant={building.rebuiltVariant}
           freshness={building.freshness}
         />
      ))}
    </group>
  )
}

export const CRITICAL_SMOKE_PUFF_COUNT = 5

export function criticalSmokePresentation(strength: number) {
  const intensity = THREE.MathUtils.clamp(Number.isFinite(strength) ? strength : 0, 0, 1)
  return {
    puffCount: CRITICAL_SMOKE_PUFF_COUNT,
    intensity,
    rise: 2.25 + intensity * 0.85,
    spread: 0.2 + intensity * 0.05,
    radius: 0.26 + intensity * 0.12,
    growth: 0.48 + intensity * 0.22,
    opacity: intensity * (0.22 + intensity * 0.12),
    colorMix: intensity,
  }
}

function CriticalSmoke({ district, reducedMotion, motionTime, strength }: {
  district: DistrictDefinition
  reducedMotion: boolean
  motionTime: number
  strength: number
}) {
  const group = useRef<Group>(null)
  const appearance = criticalSmokePresentation(strength)
  const color = useMemo(() => {
    const value = new THREE.Color('#6b6e69').lerp(new THREE.Color('#555a57'), appearance.colorMix)
    return `#${value.getHexString()}`
  }, [appearance.colorMix])
  useFrame(() => {
    if (!group.current) return
    group.current.children.forEach((child, index) => {
      const phase = reducedMotion
        ? (index + 1) / (group.current!.children.length + 1)
        : presentationMotionPhase(motionTime, 0.22 + index * 0.025, index * 0.31)
      child.position.set(
        Math.sin(index * 2.4) * (0.34 + phase * appearance.spread),
        0.72 + phase * appearance.rise,
        Math.cos(index * 2.4) * (0.34 + phase * appearance.spread * 0.8),
      )
      child.scale.setScalar(appearance.radius + phase * appearance.growth)
    })
  })
  return (
    <group
      ref={group}
      position={[district.center[0], 0.3, district.center[2]]}
    >
      {Array.from({ length: appearance.puffCount }, (_, index) => (
        <mesh key={index}>
          <dodecahedronGeometry args={[0.52, 0]} />
          <meshStandardMaterial
            color={color}
            transparent
            opacity={appearance.opacity * (0.82 + index * 0.035)}
            depthWrite={false}
            roughness={1}
            flatShading
          />
        </mesh>
      ))}
    </group>
  )
}

function RelayOrb({
  narration,
  day,
  reducedMotion,
  announce,
  motionTime,
}: {
  narration: string
  day: number
  reducedMotion: boolean
  announce: boolean
  motionTime: number
}) {
  const orb = useRef<Group>(null)
  const face = useRef<Group>(null)
  const pulse = useRef<Mesh>(null)
  const wavePoints = useMemo(() => Array.from({ length: 15 }, (_, index) => {
    const x = -0.62 + index * (1.24 / 14)
    return [x, Math.sin(index * 1.34) * 0.045, 0] as [number, number, number]
  }), [])
  useFrame(({ camera }) => {
    if (orb.current) {
      orb.current.position.y = reducedMotion ? 5.42 : 5.42 + Math.sin(motionTime * 1.05) * 0.08
    }
    if (face.current) {
      face.current.quaternion.copy(camera.quaternion)
      face.current.scale.x = reducedMotion ? 1 : 0.9 + Math.sin(motionTime * 5.2 + day) * 0.08
    }
    if (pulse.current) {
      const scale = reducedMotion ? 1 : 1 + Math.sin(motionTime * 2.4) * 0.025
      pulse.current.scale.setScalar(scale)
    }
  })
  return (
    <group ref={orb} position={[0, 5.42, 0]}>
      <mesh ref={pulse} castShadow>
        <sphereGeometry args={[0.86, 32, 24]} />
        <meshStandardMaterial color="#171918" roughness={0.93} metalness={0.03} />
      </mesh>
      <group ref={face}>
        {[-0.28, -0.14, 0, 0.14, 0.28].map((y, index) => (
          <Line
            key={y}
            points={wavePoints}
            position={[0, y, 0.93]}
            scale={[1 - Math.abs(index - 2) * 0.08, 0.7 + (index % 2) * 0.35, 1]}
            color={index === 2 ? '#f0f2ed' : '#9aa9a1'}
            lineWidth={index === 2 ? 2.1 : 1.15}
            transparent
            opacity={0.9 - Math.abs(index - 2) * 0.14}
          />
        ))}
      </group>
      <Html position={[0, 1.06, 0]} center className="relay-anchor" zIndexRange={[20, 0]}>
        <div className="relay-bubble" role={announce ? 'status' : undefined} aria-live={announce ? 'polite' : undefined}>
          <span>RELAY / DAY {day}</span>
          <p>{narration}</p>
        </div>
      </Html>
    </group>
  )
}

function DaylightRig({ absoluteDay, wellbeing, weatherStrength, quality }: {
  absoluteDay: number
  wellbeing: number
  weatherStrength: number
  quality: RenderQualityProfile
}) {
  const sunProgress = 0.5 + Math.sin(absoluteDay * Math.PI * 0.42 - 0.7) * 0.26
  const arc = 0.18 + sunProgress * Math.PI * 0.64
  const conditionLight = 0.86 + Math.max(0, Math.min(1, wellbeing)) * 0.14
  const sunPosition: [number, number, number] = [
    Math.cos(arc) * 22,
    18 + Math.sin(arc) * 10,
    14 - sunProgress * 24,
  ]
  return (
    <>
      <hemisphereLight args={['#f7fbf5', '#9a8060', (1.6 - weatherStrength * 0.18) * conditionLight]} />
      <directionalLight
        position={sunPosition}
        intensity={(2.55 - weatherStrength * 0.72) * conditionLight}
        color="#fff5df"
        castShadow={quality.shadowsEnabled}
        shadow-mapSize={[quality.shadowMapSize, quality.shadowMapSize]}
        shadow-camera-left={-CITY_CAMERA_LAYOUT.shadowHalfSpan}
        shadow-camera-right={CITY_CAMERA_LAYOUT.shadowHalfSpan}
        shadow-camera-top={CITY_CAMERA_LAYOUT.shadowHalfSpan}
        shadow-camera-bottom={-CITY_CAMERA_LAYOUT.shadowHalfSpan}
        shadow-camera-near={CITY_CAMERA_LAYOUT.shadowNear}
        shadow-camera-far={CITY_CAMERA_LAYOUT.shadowFar}
        shadow-normalBias={0.025}
      />
      <directionalLight position={[14, 10, -12]} intensity={(0.6 - weatherStrength * 0.12) * conditionLight} color="#b9d2d3" />
    </>
  )
}

function Tabletop() {
  return (
    <mesh position={[0, CITY_TABLETOP.positionY, 0]} receiveShadow>
      <cylinderGeometry args={[
        CITY_TABLETOP.topRadius,
        CITY_TABLETOP.bottomRadius,
        CITY_TABLETOP.height,
        72,
      ]} />
      <meshStandardMaterial color="#c9ae86" roughness={0.96} />
    </mesh>
  )
}

export function CityScene({
  result,
  dayIndex,
  presentation,
  aimedService,
  pendingImpact,
  activeImpact,
  selectedService = null,
  onSelectService,
  onImpactComplete,
  darkServices = [],
  previousDarkServices = [],
  stumble = false,
  previousStumble = false,
  fallen = false,
  responseEnabled = true,
  incidentPhase = 'CLEAR',
  vehicleMode = responseEnabled ? 'full' : null,
  onDockDwellChange,
  narrationOverride,
  quality,
}: {
  result: CompareResponse
  dayIndex: number
  presentation: PresentationSample
  aimedService: Service | null
  pendingImpact: CityImpactEvent | null
  activeImpact: CityImpactEvent | null
  selectedService?: Service | null
  onSelectService?: (service: Service) => void
  onImpactComplete?: () => void
  darkServices?: readonly Service[]
  previousDarkServices?: readonly Service[]
  stumble?: boolean
  previousStumble?: boolean
  fallen?: boolean
  responseEnabled?: boolean
  incidentPhase?: IncidentPhase
  vehicleMode?: VehicleFleetMode | null
  onDockDwellChange?: (dwell: VehicleDockDwell) => void
  narrationOverride?: string
  quality: RenderQualityProfile
}) {
  const reducedMotion = usePrefersReducedMotion()
  const day = result.candidate.trajectory[dayIndex]
  const presentationServices = presentation.services
  const presentationDay = presentation.visualDay
  const presentationAbsoluteDay = presentation.cursor.dayIndex + presentation.progress
  const motionTime = presentationMotionTime(presentationAbsoluteDay)
  const repairMotionTime = repairPresentationMotionTime(presentationAbsoluteDay)
  const shiftChangeBlend = repairShiftChangeBlend(presentationAbsoluteDay)
  const repairDayProgress = presentation.shockAtBoundary
    ? presentation.recoveryProgress
    : presentation.progress
  const narration = narrationOverride ?? relayOrbNarration(result, dayIndex)
  const conditionWorld = useRef<Group>(null)
  const recentWeather = latestIncidentOfType(result, dayIndex, 'weather')
  const weatherRecovery = recentWeather ? (() => {
    const service = strongestShockService(recentWeather.shock, result.services)
    const serviceOffset = result.services.indexOf(service)
    const incidentDay = result.candidate.trajectory[recentWeather.dayIndex]
    const low = incidentDay.services_after_shock[serviceOffset] ?? 0
    const target = incidentDay.services_before[serviceOffset] ?? low
    const current = presentationServices[serviceOffset] ?? low
    return target <= low + 1e-8 ? 1 : THREE.MathUtils.clamp((current - low) / (target - low), 0, 1)
  })() : 1
  const weatherStrength = pendingImpact?.type === 'weather'
    ? pendingImpact.severity
    : activeImpact?.type === 'weather'
      ? activeImpact.severity
      : recentWeather && weatherRecovery < 1
        ? Math.max(0, recentWeather.shock.severity * (1 - weatherRecovery * 0.72))
        : 0
  useFrame(() => {
    if (!conditionWorld.current) return
    const stumbleBlend = stumble
      ? (previousStumble ? 1 : presentation.easedProgress)
      : (previousStumble ? 1 - presentation.easedProgress : 0)
    const strength = !reducedMotion ? 0.012 * stumbleBlend : 0
    conditionWorld.current.rotation.x = Math.sin(motionTime * 3.1 + day.day) * strength * 0.55
    conditionWorld.current.rotation.z = Math.sin(motionTime * 2.3 + day.day * 0.7) * strength
  })
  const conditionTone = THREE.MathUtils.clamp((0.72 - presentation.wellbeing) / 0.62, 0, 1)
  const background = new THREE.Color('#dce8e2').lerp(
    new THREE.Color('#aeb9b4'),
    conditionTone * 0.34 + weatherStrength * 0.08,
  )
  const backgroundColor = `#${background.getHexString()}`
  const darkBlendFor = (service: Service) => {
    const current = darkServices.includes(service)
    const previous = previousDarkServices.includes(service)
    if (current && previous) return 1
    if (current) return presentation.easedProgress
    if (previous) return 1 - presentation.easedProgress
    return 0
  }
  return (
    <>
      <color attach="background" args={[backgroundColor]} />
      <fog attach="fog" args={[backgroundColor, CITY_CAMERA_LAYOUT.fogNear, CITY_CAMERA_LAYOUT.fogFar]} />
      <DaylightRig absoluteDay={presentation.cursor.dayIndex + presentation.progress} wellbeing={presentation.wellbeing} weatherStrength={weatherStrength} quality={quality} />
      <group ref={conditionWorld}>
        <group>
          <Tabletop />
          <Baseplate />
          <CityTrees weatherStrength={weatherStrength} reducedMotion={reducedMotion} motionTime={motionTime} quality={quality} />
          {DISTRICTS.map((district) => (
            <District
              key={district.service}
              district={district}
              result={result}
              dayIndex={dayIndex}
              dayProgress={repairDayProgress}
              presentationLevel={presentationServices[serviceIndex(district.service)]}
              presentationStartLevel={presentation.serviceEndpoints.start[serviceIndex(district.service)]}
              presentationEndLevel={presentation.serviceEndpoints.end[serviceIndex(district.service)]}
              placardsVisible={responseEnabled}
              aimed={aimedService === district.service}
              dark={darkServices.includes(district.service)}
              darkBlend={darkBlendFor(district.service)}
              selected={selectedService === district.service}
              onSelect={onSelectService}
              responseEnabled={responseEnabled}
              quality={quality}
            />
          ))}
          {DISTRICTS.map((district) => {
            const level = presentationServices[serviceIndex(district.service)] ?? 1
            const criticalStrength = THREE.MathUtils.clamp(
              (CRITICAL_FLOOR - level) / 0.045,
              0,
              1,
            )
            return (
            <CriticalSmoke
              key={`critical-${district.service}`}
              district={district}
              reducedMotion={reducedMotion}
              motionTime={motionTime}
              strength={criticalStrength}
            />
            )
          })}
          <DepotNetwork
            day={presentationDay}
            services={result.services}
            selectedService={selectedService}
            onSelectService={onSelectService}
            reducedMotion={reducedMotion}
            presentationTime={motionTime}
            operationsVisible={responseEnabled}
          />
          <RelayOrb
            narration={narration}
            day={day.day}
            reducedMotion={reducedMotion}
            announce={!narrationOverride}
            motionTime={motionTime}
          />
          <SceneEffects
            result={result}
            dayIndex={dayIndex}
            inactiveServices={darkServices}
            previousInactiveServices={previousDarkServices}
            responseEnabled={responseEnabled}
            reducedMotion={reducedMotion}
            presentationProgress={repairDayProgress}
            presentationTime={motionTime}
            repairPresentationTime={repairMotionTime}
            shiftChangeBlend={shiftChangeBlend}
          />
          {vehicleMode ? (
            <VehicleFleet
              result={result}
              dayIndex={dayIndex}
              mode={vehicleMode}
              presentationAbsoluteDay={presentationAbsoluteDay}
              reducedMotion={reducedMotion}
              onDockDwellChange={onDockDwellChange}
              vehicleParts={quality.vehicleParts}
              castShadows={quality.smallObjectShadows}
            />
          ) : null}
          <InfrastructureScene
            result={result}
            dayIndex={dayIndex}
            reducedMotion={reducedMotion}
            responseEnabled={responseEnabled}
            presentationServices={presentationServices}
            presentationTime={motionTime}
          />
          <RecoveryPhenomenology
            result={result}
            dayIndex={dayIndex}
            presentationServices={presentationServices}
            presentationWellbeing={presentation.wellbeing}
            fallen={fallen}
            responseEnabled={responseEnabled}
            reducedMotion={reducedMotion}
            presentationTime={motionTime}
          />
          <DisasterPresentation
            telegraph={pendingImpact}
            strike={activeImpact}
            services={result.services}
            presentationAbsoluteDay={presentation.cursor.dayIndex + presentation.progress}
            reducedMotion={reducedMotion}
            onStrikeComplete={onImpactComplete}
          />
        </group>
      </group>
      <CityCamera />
    </>
  )
}
