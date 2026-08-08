import type { Service } from '../types'

export type WorldPoint = readonly [number, number, number]

export const CITY_BUILDING_ARCHETYPES = [
  'apartment',
  'rowhouse',
  'office',
  'hospital',
  'market',
  'warehouse',
  'transit',
  'civic',
] as const

export type CityBuildingArchetype = (typeof CITY_BUILDING_ARCHETYPES)[number]

export type CityDistrictLayout = {
  service: Service
  center: WorldPoint
}

export type CityBuildingPlacement = {
  service: Service
  buildingIndex: number
  archetype: CityBuildingArchetype
  position: WorldPoint
  rotation: number
  scale: number
}

export type CityRoadSegment = {
  id: string
  position: WorldPoint
  scale: WorldPoint
  rotation: number
}

export type CityTreePlacement = {
  position: WorldPoint
  scale: number
  rotation: number
}

export type CityDepotLayout = {
  service: Service
  position: WorldPoint
  /** Road-side loading bay. Vehicles stop here instead of clipping through the depot body. */
  curb: WorldPoint
  rotation: number
  reservedBuildingIndex: number
}

/** One shared source for scene geometry, targeting, and camera containment. */
export const CITY_PLATE = {
  width: 54.6,
  depth: 48.6,
  height: 0.7,
  positionY: -0.34,
  borderWidth: 55.2,
  borderDepth: 49.2,
  borderHeight: 0.18,
  borderPositionY: -0.52,
  studColumns: 54,
  studRows: 48,
  studPositionY: 0.12,
  studHeight: 0.12,
} as const

export const CITY_TABLETOP = {
  topRadius: 56,
  bottomRadius: 57,
  height: 0.35,
  positionY: -0.72,
} as const

export const CITY_CAMERA_TARGET_Y = 0.35

export const CITY_PLATE_CAMERA_BOUNDS = {
  halfWidth: CITY_PLATE.borderWidth / 2,
  halfDepth: CITY_PLATE.borderDepth / 2,
  minY: CITY_PLATE.positionY - CITY_PLATE.height / 2,
  maxY: CITY_PLATE.studPositionY + CITY_PLATE.studHeight / 2,
} as const

const ORIGINAL_PLATE_BORDER_WIDTH = 25.2
const ORIGINAL_PLATE_BORDER_DEPTH = 23.2

export const CITY_LINEAR_SCALE = Math.max(
  CITY_PLATE.borderWidth / ORIGINAL_PLATE_BORDER_WIDTH,
  CITY_PLATE.borderDepth / ORIGINAL_PLATE_BORDER_DEPTH,
)

export const CITY_CAMERA_LAYOUT = {
  defaultPosition: [42, 39, 50] as WorldPoint,
  landscapeCompositionDistance: 40 * CITY_LINEAR_SCALE,
  portraitCompositionDistance: 35.5 * CITY_LINEAR_SCALE,
  portraitDefaultPolarAngle: 0.62,
  portraitDefaultAzimuthAngle: 0.04,
  containmentPadding: 0.35 * CITY_LINEAR_SCALE,
  minimumZoomSpan: 14 * CITY_LINEAR_SCALE,
  orbitClearance: 8 * CITY_LINEAR_SCALE,
  farClearance: 42 * CITY_LINEAR_SCALE,
  fogNear: 48 * CITY_LINEAR_SCALE,
  fogFar: 108 * CITY_LINEAR_SCALE,
  shadowHalfSpan: 34,
  shadowNear: 4,
  shadowFar: 105,
} as const

export const CITY_DISTRICT_PAD_SIZE = [16.9, 0.07, 15.8] as WorldPoint
export const CITY_AIM_RING_RADIUS = 8.25
export const CITY_BUILDING_CLEARANCE_RADIUS = 1.35
export const CITY_HUB_CLEAR_RADIUS = 6

export const CITY_HUB = {
  position: [0, 0.28, 0] as WorldPoint,
  fuelPoint: [3.5, 0.25, 1.75] as WorldPoint,
  fuelLane: [4.25, 0.43, 2.7] as WorldPoint,
  staging: [-3.7, 0.23, 1.7] as WorldPoint,
  dispatchGate: [-2.8, 0.43, -2] as WorldPoint,
  railDock: [0, 0.43, -2.35] as WorldPoint,
  roadIngress: [-27.0, 0.28, -2] as WorldPoint,
  railIngress: [0, 0.24, -23.8] as WorldPoint,
} as const

export const CITY_DISTRICTS = [
  { service: 'housing', center: [-17, 0, -11.5] },
  { service: 'healthcare', center: [17, 0, -11.5] },
  { service: 'food', center: [-17, 0, 7.5] },
  { service: 'transport', center: [17, 0, 7.5] },
  { service: 'public_services', center: [0, 0, 15.5] },
] as const satisfies readonly CityDistrictLayout[]

/** One district building lot is exchanged for the local point of distribution. */
export const CITY_DEPOTS = [
  { service: 'housing', position: [-10, 0.2, -4.5], curb: [-7.4, 0.43, -4.5], rotation: Math.PI, reservedBuildingIndex: 35 },
  { service: 'healthcare', position: [10, 0.2, -4.5], curb: [7.4, 0.43, -4.5], rotation: Math.PI, reservedBuildingIndex: 5 },
  { service: 'food', position: [-10, 0.2, 0.5], curb: [-7.4, 0.43, 0.5], rotation: 0, reservedBuildingIndex: 30 },
  { service: 'transport', position: [10, 0.2, 0.5], curb: [7.4, 0.43, 0.5], rotation: 0, reservedBuildingIndex: 0 },
  { service: 'public_services', position: [-1.8, 0.2, 8.5], curb: [0.8, 0.43, 8.5], rotation: 0, reservedBuildingIndex: 12 },
] as const satisfies readonly CityDepotLayout[]

export const CITY_BRIDGE = {
  position: [7.5, 0.48, -2] as WorldPoint,
  rotation: 0,
  size: [7.6, 0.24, 2.7] as WorldPoint,
} as const

export const CITY_SUBSTATION = [4.8, 0.22, 11.0] as WorldPoint
export const CITY_WATER_TOWER = [-5.1, 0.22, 12.1] as WorldPoint
export const CITY_HOUSING_PARK = [-19.4, 0.22, -12.4] as WorldPoint
export const CITY_CIVIC_FLAG = [4.9, 0.2, 17.2] as WorldPoint

const DISTRICT_GRID_COORDINATES = [-7, -4.4, -1.8, 1.8, 4.4, 7] as const

export const CITY_BUILDING_OFFSETS = DISTRICT_GRID_COORDINATES.flatMap((offsetX) =>
  DISTRICT_GRID_COORDINATES.map((offsetZ) => [offsetX, offsetZ] as const),
)

export const CITY_BUILDINGS_PER_DISTRICT = CITY_BUILDING_OFFSETS.length
export const CITY_BUILDING_COUNT = CITY_BUILDINGS_PER_DISTRICT * CITY_DISTRICTS.length

const SERVICE_ARCHETYPES: Readonly<Record<Service, readonly CityBuildingArchetype[]>> = {
  transport: ['transit', 'warehouse', 'office', 'transit', 'warehouse', 'office', 'civic'],
  housing: ['apartment', 'rowhouse', 'apartment', 'rowhouse', 'office', 'apartment', 'rowhouse'],
  food: ['market', 'warehouse', 'rowhouse', 'market', 'warehouse', 'office', 'market'],
  healthcare: ['hospital', 'office', 'hospital', 'civic', 'office', 'hospital', 'rowhouse'],
  public_services: ['civic', 'office', 'civic', 'market', 'office', 'civic', 'rowhouse'],
}

export function createCityBuildingPlacements(): CityBuildingPlacement[] {
  return CITY_DISTRICTS.flatMap((district, districtIndex) => {
    const archetypes = SERVICE_ARCHETYPES[district.service]
    return CITY_BUILDING_OFFSETS.map(([offsetX, offsetZ], buildingIndex) => ({
      service: district.service,
      buildingIndex,
      archetype: archetypes[buildingIndex % archetypes.length],
      position: [district.center[0] + offsetX, 0.15, district.center[2] + offsetZ] as const,
      rotation: ((buildingIndex + districtIndex) % 4) * (Math.PI / 2),
      scale: 0.76 + ((buildingIndex * 17 + districtIndex * 5) % 4) * 0.04,
    }))
  })
}

export const CITY_BUILDING_PLACEMENTS: readonly CityBuildingPlacement[] = createCityBuildingPlacements()

/** Streets run between the six-by-six building lots, never through them. */
export const CITY_LOCAL_STREET_OFFSETS = [-5.7, -3.1, 0, 3.1, 5.7] as const

const localRoads = CITY_DISTRICTS.flatMap((district) => {
  const [x, , z] = district.center
  const roads: CityRoadSegment[] = CITY_LOCAL_STREET_OFFSETS.flatMap((offset) => ([
    {
      id: `${district.service}-east-west-${offset}`,
      position: [x, 0.2, z + offset] as WorldPoint,
      scale: [16.8, 0.14, 0.72] as WorldPoint,
      rotation: 0,
    },
    {
      id: `${district.service}-north-south-${offset}`,
      position: [x + offset, 0.2, z] as WorldPoint,
      scale: [15.8, 0.14, 0.72] as WorldPoint,
      rotation: Math.PI / 2,
    },
  ]))
  if (district.center[0] !== 0) {
    roads.push({
      id: `${district.service}-depot-connector`,
      position: [x / 4, 0.195, z],
      scale: [Math.abs(x) / 2, 0.13, 0.7],
      rotation: 0,
    })
  }
  return roads
})

/** Depot loading bays and their district-side detours share the same paved graph as routes. */
const depotFeeders = CITY_DEPOTS.flatMap((depot) => {
  const district = CITY_DISTRICTS.find((entry) => entry.service === depot.service)!
  const directCenterX = depot.curb[0] / 2
  const directLength = Math.abs(depot.curb[0])
  const districtCenterX = (district.center[0] + depot.curb[0]) / 2
  const districtLength = Math.abs(district.center[0] - depot.curb[0])
  return [
    {
      id: `${depot.service}-depot-direct-feeder`,
      position: [directCenterX, 0.195, depot.curb[2]] as WorldPoint,
      scale: [Math.max(0.9, directLength), 0.13, 0.72] as WorldPoint,
      rotation: 0,
    },
    {
      id: `${depot.service}-depot-district-feeder`,
      position: [districtCenterX, 0.196, depot.curb[2]] as WorldPoint,
      scale: [Math.max(0.9, districtLength), 0.13, 0.72] as WorldPoint,
      rotation: 0,
    },
  ] satisfies CityRoadSegment[]
})

export const CITY_ROAD_SEGMENTS: readonly CityRoadSegment[] = [
  {
    id: 'central-east-west-arterial',
    position: [0, 0.18, -2],
    scale: [50, 0.16, 1.5],
    rotation: 0,
  },
  {
    id: 'central-north-south-arterial',
    position: [0, 0.19, 0.5],
    scale: [44, 0.16, 1.5],
    rotation: Math.PI / 2,
  },
  {
    id: 'west-external-road-stub',
    position: [-25.4, 0.18, -2],
    scale: [5.2, 0.16, 1.5],
    rotation: 0,
  },
  {
    id: 'east-external-road-stub',
    position: [25.4, 0.18, -2],
    scale: [5.2, 0.16, 1.5],
    rotation: 0,
  },
  {
    id: 'south-hub-rail-spur',
    position: [0, 0.18, -17.4],
    scale: [13, 0.11, 1.25],
    rotation: Math.PI / 2,
  },
  {
    id: 'north-civic-loop',
    position: [0, 0.18, 15.5],
    scale: [17.5, 0.14, 0.85],
    rotation: 0,
  },
  {
    id: 'hub-staging-apron',
    position: [-3.65, 0.205, 0.45],
    scale: [4.7, 0.13, 5.1],
    rotation: 0,
  },
  {
    id: 'hub-fuel-spur-east-west',
    position: [2.125, 0.198, 2.7],
    scale: [4.25, 0.13, 0.72],
    rotation: 0,
  },
  ...localRoads,
  ...depotFeeders,
]

export const CITY_LANE_MARKERS = [
  ...Array.from({ length: 17 }, (_, index) => ({
    id: `east-west-${index}`,
    position: [-24 + index * 3, 0.285, -2] as WorldPoint,
    scale: [1.35, 0.02, 0.07] as WorldPoint,
    rotation: 0,
  })),
  ...Array.from({ length: 15 }, (_, index) => ({
    id: `north-south-${index}`,
    position: [0, 0.285, -20.5 + index * 3] as WorldPoint,
    scale: [1.35, 0.02, 0.07] as WorldPoint,
    rotation: Math.PI / 2,
  })),
] as const satisfies readonly CityRoadSegment[]

export const CITY_RAIL_TIES = Array.from({ length: 17 }, (_, index) => ({
  id: `rail-tie-${index}`,
  position: [0, 0.315, -23.2 + index * 1.35] as WorldPoint,
  scale: [1.65, 0.06, 0.11] as WorldPoint,
  rotation: 0,
})) satisfies readonly CityRoadSegment[]

export const CITY_ROUTE_WAYPOINTS: Readonly<Record<Service, {
  direct: readonly WorldPoint[]
  detour: readonly WorldPoint[]
}>> = {
  housing: {
    direct: [CITY_HUB.dispatchGate, [0, 0.43, -2], [0, 0.43, -4.5], CITY_DEPOTS[0].curb],
    detour: [CITY_HUB.dispatchGate, [0, 0.43, -2], [0, 0.43, -11.5], [-17, 0.43, -11.5], [-17, 0.43, -4.5], CITY_DEPOTS[0].curb],
  },
  healthcare: {
    direct: [CITY_HUB.dispatchGate, [0, 0.43, -2], [0, 0.43, -4.5], CITY_DEPOTS[1].curb],
    detour: [CITY_HUB.dispatchGate, [0, 0.43, -2], [0, 0.43, -11.5], [17, 0.43, -11.5], [17, 0.43, -4.5], CITY_DEPOTS[1].curb],
  },
  food: {
    direct: [CITY_HUB.dispatchGate, [0, 0.43, -2], [0, 0.43, 0.5], CITY_DEPOTS[2].curb],
    detour: [CITY_HUB.dispatchGate, [0, 0.43, -2], [0, 0.43, 7.5], [-17, 0.43, 7.5], [-17, 0.43, 0.5], CITY_DEPOTS[2].curb],
  },
  transport: {
    direct: [CITY_HUB.dispatchGate, [0, 0.43, -2], [0, 0.43, 0.5], CITY_DEPOTS[3].curb],
    detour: [CITY_HUB.dispatchGate, [0, 0.43, -2], [0, 0.43, 7.5], [17, 0.43, 7.5], [17, 0.43, 0.5], CITY_DEPOTS[3].curb],
  },
  public_services: {
    direct: [CITY_HUB.dispatchGate, [0, 0.43, -2], [0, 0.43, 8.5], CITY_DEPOTS[4].curb],
    detour: [CITY_HUB.dispatchGate, [0, 0.43, -2], [0, 0.43, 15.5], [5.7, 0.43, 15.5], [5.7, 0.43, 9.8], [0, 0.43, 9.8], [0, 0.43, 8.5], CITY_DEPOTS[4].curb],
  },
} as const

/** Pure road-containment predicate used by route tests and future procedural traffic. */
export function isPointOnCityRoad(point: WorldPoint, padding = 0.04): boolean {
  return CITY_ROAD_SEGMENTS.some((road) => {
    const dx = point[0] - road.position[0]
    const dz = point[2] - road.position[2]
    const cosine = Math.cos(road.rotation)
    const sine = Math.sin(road.rotation)
    const localX = cosine * dx + sine * dz
    const localZ = -sine * dx + cosine * dz
    return Math.abs(localX) <= road.scale[0] / 2 + padding
      && Math.abs(localZ) <= road.scale[2] / 2 + padding
  })
}

function createTreePlacements(): CityTreePlacement[] {
  const candidateCount = 72
  return Array.from({ length: candidateCount }, (_, index) => {
    const angle = (index / candidateCount) * Math.PI * 2
    const radiusX = index % 2 === 0 ? 25.2 : 26
    const radiusZ = index % 3 === 0 ? 21.9 : 22.6
    return {
      position: [Math.cos(angle) * radiusX, 0.08, Math.sin(angle) * radiusZ] as WorldPoint,
      scale: 0.76 + (index % 5) * 0.055,
      rotation: angle + (index % 4) * 0.17,
    }
  }).filter((tree) => CITY_BUILDING_PLACEMENTS.every((building) => (
    Math.hypot(
      tree.position[0] - building.position[0],
      tree.position[2] - building.position[2],
    ) >= 2.05
  )))
}

export const CITY_TREE_PLACEMENTS = createTreePlacements()
