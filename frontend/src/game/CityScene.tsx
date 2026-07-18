import { Html, Line, OrbitControls } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { Group, InstancedMesh, Mesh } from 'three'
import type { CompareResponse, Service } from '../types'
import {
  DISTRICTS,
  damageStateFor,
  isBuildingRebuilding,
  relayNarration,
  serviceIndex,
  type BuildingArchetype,
  type DamageState,
  type DistrictDefinition,
} from './model'

type BuildingPlacement = {
  archetype: BuildingArchetype
  position: [number, number, number]
  rotation: number
  scale: number
}

type ArchetypeSpec = {
  size: [number, number, number]
  capScale: [number, number, number]
  capOffset: [number, number, number]
}

const ARCHETYPES: Record<BuildingArchetype, ArchetypeSpec> = {
  apartment: { size: [1.7, 3.2, 1.55], capScale: [1.3, 0.35, 1.2], capOffset: [0, 3.38, 0] },
  rowhouse: { size: [2.05, 1.65, 1.5], capScale: [1.55, 0.35, 1.05], capOffset: [0.18, 1.84, 0] },
  office: { size: [1.7, 3.8, 1.65], capScale: [1.25, 0.45, 1.2], capOffset: [-0.14, 4.02, 0.08] },
  hospital: { size: [2.55, 1.65, 1.85], capScale: [1.65, 0.32, 1.25], capOffset: [0, 1.86, 0] },
  market: { size: [2.45, 1.35, 1.85], capScale: [2.05, 0.3, 1.35], capOffset: [0, 1.57, 0] },
  warehouse: { size: [2.55, 1.5, 1.75], capScale: [1.85, 0.38, 1.15], capOffset: [-0.18, 1.75, 0] },
  transit: { size: [2.8, 1.25, 1.55], capScale: [2.3, 0.22, 1.05], capOffset: [0, 1.48, 0] },
  civic: { size: [2.35, 2.25, 1.75], capScale: [1.65, 0.36, 1.2], capOffset: [0, 2.48, 0] },
}

const DISTRICT_OFFSETS: Array<[number, number]> = [
  [-2.0, -1.35], [0, -1.55], [2.0, -1.2], [-2.1, 0.75], [0, 0.55], [2.1, 0.8], [0, 2.25],
]

const SERVICE_ARCHETYPES: Record<Service, BuildingArchetype[]> = {
  transport: ['transit', 'warehouse', 'office', 'transit', 'warehouse', 'office', 'civic'],
  housing: ['apartment', 'rowhouse', 'apartment', 'rowhouse', 'office', 'apartment', 'rowhouse'],
  food: ['market', 'warehouse', 'rowhouse', 'market', 'warehouse', 'office', 'market'],
  healthcare: ['hospital', 'office', 'hospital', 'civic', 'office', 'hospital', 'rowhouse'],
  public_services: ['civic', 'office', 'civic', 'market', 'office', 'civic', 'rowhouse'],
}

function buildingPlacements(district: DistrictDefinition): BuildingPlacement[] {
  return DISTRICT_OFFSETS.map(([offsetX, offsetZ], index) => ({
    archetype: SERVICE_ARCHETYPES[district.service][index],
    position: [district.center[0] + offsetX, 0.15, district.center[2] + offsetZ],
    rotation: ((index + serviceIndex(district.service)) % 4) * (Math.PI / 2),
    scale: 0.86 + ((index * 17 + serviceIndex(district.service) * 5) % 4) * 0.045,
  }))
}

function BaseplateStuds() {
  const mesh = useRef<InstancedMesh>(null)
  const columns = 24
  const rows = 22
  useLayoutEffect(() => {
    if (!mesh.current) return
    const matrix = new THREE.Matrix4()
    let index = 0
    for (let x = 0; x < columns; x += 1) {
      for (let z = 0; z < rows; z += 1) {
        matrix.makeTranslation(x - (columns - 1) / 2, 0.12, z - (rows - 1) / 2)
        mesh.current.setMatrixAt(index, matrix)
        index += 1
      }
    }
    mesh.current.instanceMatrix.needsUpdate = true
  }, [])
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, columns * rows]} receiveShadow>
      <cylinderGeometry args={[0.14, 0.15, 0.12, 12]} />
      <meshStandardMaterial color="#71816d" roughness={0.83} />
    </instancedMesh>
  )
}

function Baseplate() {
  const treePositions = useMemo(() => Array.from({ length: 24 }, (_, index) => {
    const angle = (index / 24) * Math.PI * 2
    const radiusX = index % 2 ? 10.25 : 9.65
    const radiusZ = index % 3 ? 8.9 : 9.65
    return [Math.cos(angle) * radiusX, Math.sin(angle) * radiusZ] as const
  }), [])
  return (
    <group>
      <mesh position={[0, -0.34, 0]} receiveShadow castShadow>
        <boxGeometry args={[24.6, 0.7, 22.6]} />
        <meshStandardMaterial color="#687767" roughness={0.9} />
      </mesh>
      <BaseplateStuds />
      <mesh position={[0, -0.52, 0]} receiveShadow>
        <boxGeometry args={[25.2, 0.18, 23.2]} />
        <meshStandardMaterial color="#46534a" roughness={0.92} />
      </mesh>
      <RoadNetwork />
      {treePositions.map(([x, z], index) => (
        <Tree key={index} position={[x, 0.08, z]} scale={0.8 + (index % 4) * 0.08} />
      ))}
    </group>
  )
}

function RoadNetwork() {
  const roads: Array<{ position: [number, number, number]; scale: [number, number, number]; rotation?: number }> = [
    { position: [0, 0.18, 0], scale: [20.5, 0.16, 1.25] },
    { position: [0, 0.19, 0], scale: [18.8, 0.16, 1.25], rotation: Math.PI / 2 },
    { position: [-4.2, 0.2, -2.9], scale: [8.4, 0.14, 0.82], rotation: Math.PI / 4 },
    { position: [4.2, 0.2, -2.9], scale: [8.4, 0.14, 0.82], rotation: -Math.PI / 4 },
    { position: [-4.2, 0.2, 3.0], scale: [8.4, 0.14, 0.82], rotation: -Math.PI / 4 },
    { position: [4.2, 0.2, 3.0], scale: [8.4, 0.14, 0.82], rotation: Math.PI / 4 },
  ]
  return (
    <group>
      {roads.map((road, index) => (
        <mesh key={index} position={road.position} rotation={[0, road.rotation ?? 0, 0]} receiveShadow>
          <boxGeometry args={road.scale} />
          <meshStandardMaterial color="#555b59" roughness={0.97} />
        </mesh>
      ))}
      {[-7, -3.5, 3.5, 7].map((x) => (
        <mesh key={x} position={[x, 0.285, 0]}>
          <boxGeometry args={[1.35, 0.02, 0.07]} />
          <meshBasicMaterial color="#d7cba9" />
        </mesh>
      ))}
    </group>
  )
}

function Tree({ position, scale }: { position: [number, number, number]; scale: number }) {
  return (
    <group position={position} scale={scale}>
      <mesh position={[0, 0.55, 0]} castShadow>
        <boxGeometry args={[0.18, 1.0, 0.18]} />
        <meshStandardMaterial color="#6c4f37" roughness={1} />
      </mesh>
      <mesh position={[0, 1.35, 0]} castShadow>
        <dodecahedronGeometry args={[0.62, 0]} />
        <meshStandardMaterial color="#587255" roughness={0.95} flatShading />
      </mesh>
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
  const spec = ARCHETYPES[archetype]
  const [width, height, depth] = spec.size
  const shellScaleY = state === 'moderate' ? 0.67 : 1
  const lean = state === 'slight' ? 0.025 : 0
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

function Building({ placement, district, state, rebuilding }: {
  placement: BuildingPlacement
  district: DistrictDefinition
  state: DamageState
  rebuilding: boolean
}) {
  const spec = ARCHETYPES[placement.archetype]
  const bodyColor = useMemo(() => {
    const color = new THREE.Color(district.body)
    if (state === 'moderate') color.lerp(new THREE.Color('#77736b'), 0.42)
    if (state === 'slight') color.lerp(new THREE.Color('#8d8579'), 0.16)
    return `#${color.getHexString()}`
  }, [district.body, state])
  return (
    <group position={placement.position} rotation={[0, placement.rotation, 0]} scale={placement.scale}>
      <mesh position={[0, -0.04, 0]} receiveShadow>
        <boxGeometry args={[2.9, 0.18, 2.35]} />
        <meshStandardMaterial color="#8f968d" roughness={0.95} />
      </mesh>
      {state === 'rubble' ? <RubbleBrickPool color={bodyColor} amount={12} /> : (
        <>
          <ArchetypeShell archetype={placement.archetype} color={bodyColor} accent={district.accent} state={state} />
          {state === 'moderate' ? <RubbleBrickPool color={bodyColor} amount={5} /> : null}
        </>
      )}
      {rebuilding ? <ScaffoldCrane height={spec.size[1]} accent={district.accent} /> : null}
    </group>
  )
}

function District({ district, result, dayIndex }: {
  district: DistrictDefinition
  result: CompareResponse
  dayIndex: number
}) {
  const day = result.candidate.trajectory[dayIndex]
  const previous = result.candidate.trajectory[dayIndex - 1]
  const index = serviceIndex(district.service)
  const level = day.services_end[index]
  const placements = useMemo(() => buildingPlacements(district), [district])
  return (
    <group>
      <mesh position={[district.center[0], 0.035, district.center[2]]} receiveShadow>
        <boxGeometry args={[5.5, 0.07, 5.45]} />
        <meshStandardMaterial color={district.accent} opacity={0.2} transparent roughness={1} />
      </mesh>
      {placements.map((placement, buildingIndex) => {
        const state = damageStateFor(level, buildingIndex)
        return (
          <Building
            key={`${district.service}-${buildingIndex}`}
            placement={placement}
            district={district}
            state={state}
            rebuilding={isBuildingRebuilding(day, previous, district.service, buildingIndex)}
          />
        )
      })}
    </group>
  )
}

function Silo({ allocations }: { allocations: number[] }) {
  const maxAllocation = Math.max(...allocations, 1)
  const colors = ['#5a8290', '#bd6b52', '#d49a3d', '#d9ded7', '#71866a']
  return (
    <group position={[0, 0.28, 0]}>
      <mesh position={[0, 1.8, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[1.42, 1.6, 3.6, 16]} />
        <meshStandardMaterial color="#c4b9a4" roughness={0.84} />
      </mesh>
      <mesh position={[0, 3.66, 0]} castShadow>
        <coneGeometry args={[1.48, 0.65, 16]} />
        <meshStandardMaterial color="#6b6f68" roughness={0.86} />
      </mesh>
      <mesh position={[0, 0.22, 0]} receiveShadow>
        <cylinderGeometry args={[1.85, 1.85, 0.28, 16]} />
        <meshStandardMaterial color="#555d57" roughness={0.92} />
      </mesh>
      {allocations.map((allocation, index) => {
        const angle = (index / 5) * Math.PI * 2 - Math.PI / 2
        const height = 0.35 + (allocation / maxAllocation) * 1.5
        return (
          <group key={index} position={[Math.cos(angle) * 1.65, 0, Math.sin(angle) * 1.65]}>
            <mesh position={[0, height / 2 + 0.25, 0]} castShadow>
              <boxGeometry args={[0.38, height, 0.38]} />
              <meshStandardMaterial color={colors[index]} roughness={0.75} />
            </mesh>
            <mesh position={[0, 0.22, 0]}>
              <boxGeometry args={[0.58, 0.18, 0.58]} />
              <meshStandardMaterial color="#514f49" roughness={0.9} />
            </mesh>
          </group>
        )
      })}
    </group>
  )
}

function RelayOrb({ narration, day }: { narration: string; day: number }) {
  const orb = useRef<Group>(null)
  const face = useRef<Group>(null)
  const pulse = useRef<Mesh>(null)
  const wavePoints = useMemo(() => Array.from({ length: 15 }, (_, index) => {
    const x = -0.62 + index * (1.24 / 14)
    return [x, Math.sin(index * 1.34) * 0.045, 0] as [number, number, number]
  }), [])
  useFrame(({ clock, camera }) => {
    if (orb.current) {
      orb.current.position.y = 5.42 + Math.sin(clock.elapsedTime * 1.05) * 0.08
    }
    if (face.current) {
      face.current.quaternion.copy(camera.quaternion)
      const cadence = 0.9 + Math.sin(clock.elapsedTime * 5.2 + day) * 0.08
      face.current.scale.x = cadence
    }
    if (pulse.current) {
      const scale = 1 + Math.sin(clock.elapsedTime * 2.4) * 0.025
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
      <Html position={[0, 1.38, 0]} center className="relay-anchor" zIndexRange={[20, 0]}>
        <div className="relay-bubble" role="status" aria-live="polite">
          <span>RELAY / DAY {day}</span>
          <p>{narration}</p>
        </div>
      </Html>
    </group>
  )
}

function Tabletop() {
  return (
    <mesh position={[0, -0.72, 0]} receiveShadow>
      <cylinderGeometry args={[26, 27, 0.35, 64]} />
      <meshStandardMaterial color="#c9ae86" roughness={0.96} />
    </mesh>
  )
}

export function CityScene({ result, dayIndex }: { result: CompareResponse; dayIndex: number }) {
  const day = result.candidate.trajectory[dayIndex]
  const narration = relayNarration(result, dayIndex)
  return (
    <>
      <color attach="background" args={['#dce8e2']} />
      <fog attach="fog" args={['#dce8e2', 31, 56]} />
      <hemisphereLight args={['#f7fbf5', '#9a8060', 1.6]} />
      <directionalLight
        position={[-12, 22, 14]}
        intensity={2.6}
        color="#fff5df"
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-16}
        shadow-camera-right={16}
        shadow-camera-top={15}
        shadow-camera-bottom={-15}
        shadow-camera-near={3}
        shadow-camera-far={46}
        shadow-normalBias={0.025}
      />
      <directionalLight position={[14, 10, -12]} intensity={0.6} color="#b9d2d3" />
      <Tabletop />
      <Baseplate />
      {DISTRICTS.map((district) => (
        <District key={district.service} district={district} result={result} dayIndex={dayIndex} />
      ))}
      <Silo allocations={day.allocation} />
      <RelayOrb narration={narration} day={day.day} />
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.065}
        enablePan={false}
        minDistance={22}
        maxDistance={38}
        minPolarAngle={0.42}
        maxPolarAngle={1.25}
        target={[0, 1.4, 0]}
      />
    </>
  )
}
