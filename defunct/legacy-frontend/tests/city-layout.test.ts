import { describe, expect, it } from 'vitest'
import type { Service } from '../src/types'
import {
  CITY_BUILDING_ARCHETYPES,
  CITY_BUILDING_CLEARANCE_RADIUS,
  CITY_BUILDING_COUNT,
  CITY_BUILDING_OFFSETS,
  CITY_BUILDING_PLACEMENTS,
  CITY_BUILDINGS_PER_DISTRICT,
  CITY_DISTRICTS,
  CITY_LANE_MARKERS,
  CITY_PLATE_CAMERA_BOUNDS,
  CITY_ROAD_SEGMENTS,
  CITY_HUB_CLEAR_RADIUS,
  CITY_TREE_PLACEMENTS,
  createCityBuildingPlacements,
} from '../src/game/worldLayout'

describe('enlarged deterministic city layout', () => {
  it('places 36 buildings in each district for a 180-building city', () => {
    expect(CITY_BUILDINGS_PER_DISTRICT).toBe(36)
    expect(CITY_BUILDING_COUNT).toBe(180)
    expect(CITY_BUILDING_OFFSETS).toHaveLength(CITY_BUILDINGS_PER_DISTRICT)
    expect(CITY_BUILDING_PLACEMENTS).toHaveLength(CITY_BUILDING_COUNT)

    const counts = CITY_BUILDING_PLACEMENTS.reduce<Partial<Record<Service, number>>>((result, building) => {
      result[building.service] = (result[building.service] ?? 0) + 1
      return result
    }, {})

    for (const district of CITY_DISTRICTS) {
      expect(counts[district.service]).toBe(CITY_BUILDINGS_PER_DISTRICT)
    }
  })

  it('is repeatable, position-unique, and uses all eight authored archetypes', () => {
    expect(createCityBuildingPlacements()).toEqual(CITY_BUILDING_PLACEMENTS)
    expect(createCityBuildingPlacements()).toEqual(createCityBuildingPlacements())

    const positions = CITY_BUILDING_PLACEMENTS.map((building) => (
      `${building.position[0].toFixed(3)}:${building.position[2].toFixed(3)}`
    ))
    expect(new Set(positions).size).toBe(CITY_BUILDING_PLACEMENTS.length)
    expect(new Set(CITY_BUILDING_PLACEMENTS.map((building) => building.archetype))).toEqual(
      new Set(CITY_BUILDING_ARCHETYPES),
    )
  })

  it('keeps every building inside the plate and outside the central intake-hub apron', () => {
    for (const building of CITY_BUILDING_PLACEMENTS) {
      expect(Math.abs(building.position[0]) + CITY_BUILDING_CLEARANCE_RADIUS)
        .toBeLessThan(CITY_PLATE_CAMERA_BOUNDS.halfWidth)
      expect(Math.abs(building.position[2]) + CITY_BUILDING_CLEARANCE_RADIUS)
        .toBeLessThan(CITY_PLATE_CAMERA_BOUNDS.halfDepth)
      expect(Math.hypot(building.position[0], building.position[2]))
        .toBeGreaterThan(CITY_HUB_CLEAR_RADIUS)
    }
  })

  it('keeps dense neighboring prefabs separated while reserving central cross-streets', () => {
    for (const district of CITY_DISTRICTS) {
      const buildings = CITY_BUILDING_PLACEMENTS.filter((building) => building.service === district.service)
      let nearest = Number.POSITIVE_INFINITY
      for (let left = 0; left < buildings.length; left += 1) {
        for (let right = left + 1; right < buildings.length; right += 1) {
          nearest = Math.min(nearest, Math.hypot(
            buildings[left].position[0] - buildings[right].position[0],
            buildings[left].position[2] - buildings[right].position[2],
          ))
        }
      }
      expect(nearest).toBeGreaterThanOrEqual(2.59)
      expect(buildings.some((building) => Math.abs(building.position[0] - district.center[0]) < 0.01)).toBe(false)
      expect(buildings.some((building) => Math.abs(building.position[2] - district.center[2]) < 0.01)).toBe(false)
    }
  })

  it('provides a dense reusable road kit and bounded procedural tree ring', () => {
    expect(CITY_ROAD_SEGMENTS.length).toBeGreaterThanOrEqual(70)
    expect(CITY_LANE_MARKERS.length).toBeGreaterThanOrEqual(30)
    expect(CITY_TREE_PLACEMENTS.length).toBeGreaterThanOrEqual(20)

    for (const tree of CITY_TREE_PLACEMENTS) {
      expect(Math.abs(tree.position[0])).toBeLessThan(CITY_PLATE_CAMERA_BOUNDS.halfWidth)
      expect(Math.abs(tree.position[2])).toBeLessThan(CITY_PLATE_CAMERA_BOUNDS.halfDepth)
    }
  })
})
