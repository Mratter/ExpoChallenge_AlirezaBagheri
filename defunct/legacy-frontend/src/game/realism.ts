import type { CompareResponse, DayResult, Service, Shock, ShockType } from '../types'
import {
  DISTRICTS,
  SERVICE_LABELS,
  damageStateFor,
  rebuildingCohortForDay,
  serviceIndex,
  type DamageState,
} from './model'
import { CITY_BUILDINGS_PER_DISTRICT } from './worldLayout'

export const HEAVY_TRUCK_CAPACITY = 12
export const LAST_MILE_CAPACITY = 4
export const LEGACY_V1_DEPOT_DISCLOSURE = 'legacy v1 result — depot state was not recorded'

export type Placard = 'GREEN' | 'YELLOW' | 'RED'
export type IncidentPhase = 'CLEAR' | 'TELEGRAPH' | 'IMPACT' | 'ASSESSMENT' | 'RESPONSE' | 'RECOVERY'
export type RecoveryStage = 'assessment' | 'debris' | 'frame' | 'wrap' | 'active-work' | 'complete'

export type DepotStatus = {
  service: Service
  district: string
  damage: DamageState | null
  placard: Placard | null
  serviceLevel: number
  allocationUnits: number
  palletUnits: number | null
  stockCapacity: number | null
  pendingUnits: number | null
  spoilageUnits: number | null
  throughputSignal: number | null
  dispatchedVehicles: number
  lastMileVehicles: number
  dockQueue: number | null
  dockQueueUnits: number | null
  reroutedFrom: Service | null
  mutualAidFrom: Service | null
  inboundWindow: string
  source: string
  engineTruth: boolean
  recorded: {
    stockBeforeUnits: number
    stockReadyUnits: number
    stockEndUnits: number
    pendingLandedUnits: number
    sameDayLandedUnits: number
    landedUnits: number
    capacityHeldUnits: number
    repairDispatchUnits: number
    repairSupplyUnits: number
    damagePenalty: number
    damageDaysRemaining: number
    damageFactor: number
    throughputFactor: number
    scheduledOrHeldNextDayUnits: number
    spoilageUnits: number
    mutualAidNetUnits: number
  } | null
}

export type VehicleManifest = {
  id: string
  service: Service | 'hub'
  fleet: string
  origin: string
  destination: string
  cargoUnits: number
  cargoPallets: number
  returnLeg: string
  scheduledStop?: string
  wave: 'inbound' | 'line-haul' | 'mutual-aid' | 'last-mile' | 'emergency' | 'maintenance' | 'civilian'
  dispatchRank: number
}

export const FUEL_POINT_STOP_COPY = 'Scheduled hub fuel-point visit before departure (deterministic presentation cadence)'

export type RecoveryArc = {
  service: Service
  startedDay: number | null
  targetLevel: number
  lowLevel: number
  currentLevel: number
  completionDay: number | null
  durationDays: number
  progress: number
  stage: RecoveryStage
}

export type IncidentHistory = {
  shock: Shock
  type: ShockType
  dayIndex: number
  ageDays: number
}

export type RecoveryMilestone = {
  day: number
  service: Service | 'city'
  label: string
  source: string
}

export type DisasterAfterAction = {
  day: number
  type: ShockType
  severity: number
  strongestService: Service
  emergencyWaveVehicles: number
  lineHaulHeavyTrucks: number
  lastMileVehicles: number
  mutualAidVehicles: number
  logisticsRecorded: boolean
  recoveryDays: Partial<Record<Service, number | null>>
}

export type VehicleDispatchCounts = {
  lineHaulHeavyTrucks: number
  lastMileVehicles: number
  mutualAidVehicles: number
}

const FLEET_LABELS: Readonly<Record<Service, string>> = {
  healthcare: 'ambulance',
  housing: 'brick flatbed',
  food: 'refrigerated truck',
  public_services: 'bucket truck',
  transport: 'recovery bus',
}

export function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.max(minimum, Math.min(maximum, value))
}

export function splitExactCargo(total: number, capacity: number): number[] {
  const safeTotal = Math.max(0, total)
  const safeCapacity = Math.max(0.001, capacity)
  const fullLoads = Math.floor((safeTotal + 1e-8) / safeCapacity)
  const loads = Array.from({ length: fullLoads }, () => safeCapacity)
  const remainder = Number((safeTotal - fullLoads * safeCapacity).toFixed(8))
  if (remainder > 1e-7) loads.push(remainder)
  return loads
}

export function depotDamageStateFromFactor(damageFactor: number): DamageState {
  // Engine v2 records retained depot function, with a deliberate 0.30 floor.
  // These four bands cover that entire recorded range. Legacy-v1 service level
  // remains district truth and is never repurposed as unrecorded depot damage.
  if (damageFactor >= 0.85) return 'intact'
  if (damageFactor >= 0.68) return 'slight'
  if (damageFactor >= 0.48) return 'moderate'
  return 'rubble'
}

export function placardForDamage(damage: DamageState): Placard {
  if (damage === 'intact') return 'GREEN'
  if (damage === 'slight') return 'YELLOW'
  return 'RED'
}

function levelFor(day: DayResult, service: Service, services: readonly Service[]): number {
  const index = services.indexOf(service)
  return index < 0 ? 0 : day.services_end[index] ?? 0
}

function mutualAidDonor(
  day: DayResult,
  receiverIndex: number,
  services: readonly Service[],
): Service | null {
  const ledger = day.logistics
  if (!ledger || (ledger.mutual_aid_net[receiverIndex] ?? 0) <= 1e-7) return null
  const receiver = services[receiverIndex]
  const transfer = ledger.mutual_aid_transfers.find((event) => event.to_service === receiver)
  const donorIndex = transfer ? services.indexOf(transfer.from_service) : -1
  if (
    !transfer
    || donorIndex < 0
    || Math.abs((ledger.mutual_aid_net[receiverIndex] ?? 0) - transfer.units) > 1e-6
    || Math.abs((ledger.mutual_aid_net[donorIndex] ?? 0) + transfer.units) > 1e-6
  ) return null
  return transfer.from_service
}

function nearestHealthyDepotForRecordedDamage(
  day: DayResult,
  service: Service,
  services: readonly Service[],
): Service | null {
  const ledger = day.logistics
  const target = DISTRICTS.find((district) => district.service === service)
  const targetIndex = services.indexOf(service)
  if (!ledger || !target || targetIndex < 0) return null
  if (depotDamageStateFromFactor(ledger.depot_damage_factor[targetIndex] ?? 1) !== 'rubble') return null
  return DISTRICTS
    .filter((district) => {
      const index = services.indexOf(district.service)
      return index >= 0
        && district.service !== service
        && depotDamageStateFromFactor(ledger.depot_damage_factor[index] ?? 0) !== 'rubble'
    })
    .sort((left, right) => (
      Math.hypot(left.center[0] - target.center[0], left.center[2] - target.center[2])
        - Math.hypot(right.center[0] - target.center[0], right.center[2] - target.center[2])
    ))[0]?.service ?? null
}

export function depotStatusesForDay(
  day: DayResult,
  services: readonly Service[],
): DepotStatus[] {
  return DISTRICTS.map((district) => {
    const index = services.indexOf(district.service)
    const ledger = day.logistics
    const serviceLevel = day.services_end[index] ?? 0
    const allocationUnits = Math.max(0, day.allocation[index] ?? 0)
    const damageFactor = ledger?.depot_damage_factor[index]
    const damage = damageFactor === undefined ? null : depotDamageStateFromFactor(damageFactor)
    const throughputSignal = ledger ? clamp(ledger.throughput_factor[index] ?? 0) : null
    const pendingUnits = ledger ? Math.max(0, ledger.pending_next_day[index] ?? 0) : null
    const capacityHeldUnits = ledger ? Math.max(0, ledger.capacity_overflow[index] ?? 0) : 0
    const dockQueueUnits = ledger ? Math.max(
      capacityHeldUnits,
      Math.max(0, ledger.pending_arrivals_held[index] ?? 0)
        + Math.max(0, ledger.same_day_delivery_held[index] ?? 0),
    ) : null
    const deliveredUnits = ledger
      ? Math.max(0, (ledger.pending_arrivals_landed[index] ?? 0) + (ledger.same_day_delivery_landed[index] ?? 0))
      : allocationUnits
    const repairSupply = Math.max(0, ledger?.repair_supply[index] ?? allocationUnits)
    const dispatchedVehicles = splitExactCargo(deliveredUnits, HEAVY_TRUCK_CAPACITY).length
    const lastMileVehicles = splitExactCargo(repairSupply, LAST_MILE_CAPACITY).length
    return {
      service: district.service,
      district: district.label,
      damage,
      placard: damage ? placardForDamage(damage) : null,
      serviceLevel,
      allocationUnits,
      palletUnits: ledger ? Math.max(0, ledger.depot_stock_end[index] ?? 0) : null,
      stockCapacity: ledger ? Math.max(0, ledger.depot_capacity[index] ?? 0) : null,
      pendingUnits,
      spoilageUnits: ledger ? Math.max(0, ledger.spoilage[index] ?? 0) : null,
      throughputSignal,
      dispatchedVehicles,
      lastMileVehicles,
      dockQueue: ledger
        ? splitExactCargo(dockQueueUnits ?? 0, HEAVY_TRUCK_CAPACITY).length
        : null,
      dockQueueUnits,
      reroutedFrom: ledger ? nearestHealthyDepotForRecordedDamage(day, district.service, services) : null,
      mutualAidFrom: ledger ? mutualAidDonor(day, index, services) : null,
      inboundWindow: ledger
        ? (pendingUnits ?? 0) > 1e-7
          ? `${(pendingUnits ?? 0).toFixed(1)} units scheduled or held for day ${day.day + 1}${(dockQueueUnits ?? 0) > 1e-7 ? ` · ${(dockQueueUnits ?? 0).toFixed(1)} constrained at docks` : ''}`
          : 'no next-day freight scheduled or held'
        : 'unavailable — legacy v1 did not record freight timing',
      source: ledger
        ? 'candidate.trajectory[day].logistics — recorded engine-v2 stock, throughput, held/overflow queue, inbound schedule, and transfer state'
        : LEGACY_V1_DEPOT_DISCLOSURE,
      engineTruth: Boolean(ledger),
      recorded: ledger ? {
        stockBeforeUnits: Math.max(0, ledger.depot_stock_before[index] ?? 0),
        stockReadyUnits: Math.max(0, ledger.depot_stock_ready[index] ?? 0),
        stockEndUnits: Math.max(0, ledger.depot_stock_end[index] ?? 0),
        pendingLandedUnits: Math.max(0, ledger.pending_arrivals_landed[index] ?? 0),
        sameDayLandedUnits: Math.max(0, ledger.same_day_delivery_landed[index] ?? 0),
        landedUnits: deliveredUnits,
        capacityHeldUnits,
        repairDispatchUnits: Math.max(0, ledger.repair_dispatch[index] ?? 0),
        repairSupplyUnits: repairSupply,
        damagePenalty: Math.max(0, ledger.depot_damage_penalty[index] ?? 0),
        damageDaysRemaining: Math.max(0, ledger.depot_damage_days_remaining[index] ?? 0),
        damageFactor: clamp(ledger.depot_damage_factor[index] ?? 0),
        throughputFactor: throughputSignal ?? 0,
        scheduledOrHeldNextDayUnits: pendingUnits ?? 0,
        spoilageUnits: Math.max(0, ledger.spoilage[index] ?? 0),
        mutualAidNetUnits: ledger.mutual_aid_net[index] ?? 0,
      } : null,
    }
  })
}

export function serviceDestination(service: Service, buildingIndex: number): string {
  const district = DISTRICTS.find((item) => item.service === service)
  return `${district?.shortLabel ?? SERVICE_LABELS[service]} site ${buildingIndex + 1}`
}

/** Shared selector for both manifests and rendered 3D routes. */
export function rebuildingDestinationsForDay(
  day: DayResult,
  previous: DayResult | undefined,
  service: Service,
  dispatchRank: number,
  buildingCount = CITY_BUILDINGS_PER_DISTRICT,
): number[] {
  const activeCohort = rebuildingCohortForDay(day, previous, service, buildingCount)
  return activeCohort.length ? activeCohort : [(dispatchRank * 7) % buildingCount]
}

/**
 * Selects at most one real, active line-haul mission for a visible hub fuel-point visit.
 * This is deliberately presentation cadence, not a claim of simulated fuel inventory.
 */
export function scheduledFuelServiceForDay(
  day: DayResult,
  services: readonly Service[],
): Service | null {
  const candidates = services
    .map((service, serviceIndex) => ({
      service,
      serviceIndex,
      allocation: day.allocation[serviceIndex] ?? 0,
      lineHaulUnits: day.logistics
        ? Math.max(
            0,
            (day.logistics.pending_arrivals_landed[serviceIndex] ?? 0)
              + (day.logistics.same_day_delivery_landed[serviceIndex] ?? 0),
          )
        : Math.max(0, day.allocation[serviceIndex] ?? 0),
    }))
    .filter((candidate) => candidate.lineHaulUnits > 1e-8)
    .sort((left, right) => (
      right.allocation - left.allocation || left.serviceIndex - right.serviceIndex
    ))
  if (!candidates.length) return null
  const cadenceIndex = Math.max(0, Math.trunc(day.day) - 1) % candidates.length
  return candidates[cadenceIndex].service
}

export function vehicleManifestsForDay(
  day: DayResult,
  previous: DayResult | undefined,
  services: readonly Service[],
): VehicleManifest[] {
  const allocationRank = services
    .map((service, index) => ({ service, allocation: day.allocation[index] ?? 0 }))
    .sort((left, right) => right.allocation - left.allocation)
  const scheduledFuelService = scheduledFuelServiceForDay(day, services)
  const statuses = depotStatusesForDay(day, services)
  const manifests: VehicleManifest[] = splitExactCargo(day.available_budget, HEAVY_TRUCK_CAPACITY)
    .map((cargo, index) => ({
      id: `d${day.day}-inbound-${index}`,
      service: 'hub',
      fleet: levelFor(day, 'transport', services) >= 0.55 ? 'rail freight wagon' : 'inbound freight truck',
      origin: 'regional supply edge',
      destination: 'central intake hub',
      cargoUnits: cargo,
      cargoPallets: cargo,
      returnLeg: 'returns empty beyond the plate edge',
      wave: 'inbound' as const,
      dispatchRank: index,
    }))

  allocationRank.forEach(({ service, allocation }, rank) => {
    const serviceIndex = services.indexOf(service)
    const ledger = day.logistics
    const depot = DISTRICTS.find((district) => district.service === service)
    const status = statuses.find((entry) => entry.service === service)
    const receivingService = status?.reroutedFrom ?? service
    const receivingDepot = DISTRICTS.find((district) => district.service === receivingService)
    const routeDisclosure = status?.reroutedFrom
      ? `Recorded local depot rubble: nearest-healthy ${receivingDepot?.shortLabel ?? receivingService} point of distribution presentation route for ${depot?.shortLabel ?? service}`
      : undefined
    const lineHaulUnits = ledger
      ? Math.max(0, (ledger.pending_arrivals_landed[serviceIndex] ?? 0) + (ledger.same_day_delivery_landed[serviceIndex] ?? 0))
      : allocation
    const lastMileUnits = Math.max(0, ledger?.repair_supply[serviceIndex] ?? allocation)
    splitExactCargo(lineHaulUnits, HEAVY_TRUCK_CAPACITY).forEach((cargo, index) => {
      manifests.push({
        id: `d${day.day}-${service}-line-${index}`,
        service,
        fleet: 'line-haul truck',
        origin: 'central intake hub',
        destination: status?.reroutedFrom
          ? `${receivingDepot?.shortLabel ?? receivingService} point of distribution · support route for ${depot?.shortLabel ?? service}`
          : `${depot?.shortLabel ?? service} point of distribution`,
        cargoUnits: cargo,
        cargoPallets: cargo,
        returnLeg: 'returns empty to hub staging',
        scheduledStop: [
          service === scheduledFuelService && index === 0 ? FUEL_POINT_STOP_COPY : null,
          routeDisclosure,
        ].filter(Boolean).join(' · ') || undefined,
        wave: 'line-haul',
        dispatchRank: rank,
      })
    })
    const destinations = rebuildingDestinationsForDay(day, previous, service, rank)
    splitExactCargo(lastMileUnits, LAST_MILE_CAPACITY).forEach((cargo, index) => {
      const buildingIndex = destinations[index % destinations.length]
      manifests.push({
        id: `d${day.day}-${service}-last-${index}`,
        service,
        fleet: FLEET_LABELS[service],
        origin: `${receivingDepot?.shortLabel ?? receivingService} point of distribution`,
        destination: serviceDestination(service, buildingIndex),
        cargoUnits: cargo,
        cargoPallets: cargo,
        returnLeg: 'returns empty to district depot',
        wave: 'last-mile',
        dispatchRank: rank,
      })
    })
  })
  if (day.logistics) {
    day.logistics.mutual_aid_transfers.forEach((transfer) => {
      const receiverIndex = services.indexOf(transfer.to_service)
      const donorIndex = services.indexOf(transfer.from_service)
      const netMatches = receiverIndex >= 0
        && donorIndex >= 0
        && Math.abs((day.logistics?.mutual_aid_net[receiverIndex] ?? 0) - transfer.units) <= 1e-6
        && Math.abs((day.logistics?.mutual_aid_net[donorIndex] ?? 0) + transfer.units) <= 1e-6
      if (!netMatches) return
      splitExactCargo(transfer.units, HEAVY_TRUCK_CAPACITY).forEach((cargo, index) => {
        manifests.push({
          id: `d${day.day}-${transfer.from_service}-${transfer.to_service}-mutual-aid-${index}`,
          service: transfer.to_service,
          fleet: 'mutual-aid line-haul truck',
          origin: `${DISTRICTS.find((district) => district.service === transfer.from_service)?.shortLabel} point of distribution`,
          destination: `${DISTRICTS.find((district) => district.service === transfer.to_service)?.shortLabel} point of distribution`,
          cargoUnits: cargo,
          cargoPallets: cargo,
          returnLeg: 'returns empty to donor depot',
          wave: 'mutual-aid',
          dispatchRank: services.length,
        })
      })
    })
  }
  return manifests
}

export function vehicleDispatchCountsForDay(
  day: DayResult,
  previous: DayResult | undefined,
  services: readonly Service[],
): VehicleDispatchCounts {
  const manifests = vehicleManifestsForDay(day, previous, services)
  return {
    lineHaulHeavyTrucks: manifests.filter((manifest) => manifest.wave === 'line-haul').length,
    lastMileVehicles: manifests.filter((manifest) => manifest.wave === 'last-mile').length,
    mutualAidVehicles: manifests.filter((manifest) => manifest.wave === 'mutual-aid').length,
  }
}

export function latestIncident(
  result: CompareResponse,
  dayIndex: number,
  maxAgeDays = 4,
): IncidentHistory | null {
  for (let index = dayIndex; index >= Math.max(0, dayIndex - maxAgeDays); index -= 1) {
    const shock = result.shock_schedule[index]
    if (shock?.type) {
      return {
        shock,
        type: shock.type as ShockType,
        dayIndex: index,
        ageDays: dayIndex - index,
      }
    }
  }
  return null
}

export function latestIncidentOfType(
  result: CompareResponse,
  dayIndex: number,
  type: ShockType,
): IncidentHistory | null {
  for (let index = dayIndex; index >= 0; index -= 1) {
    const shock = result.shock_schedule[index]
    if (shock?.type === type) {
      return { shock, type, dayIndex: index, ageDays: dayIndex - index }
    }
  }
  return null
}

/** Progress against one typed incident's own pre-event service target. */
export function incidentRecoveryProgress(
  result: CompareResponse,
  incidentDayIndex: number,
  currentDayIndex: number,
  service: Service,
): number {
  const serviceOffset = result.services.indexOf(service)
  const incidentDay = result.candidate.trajectory[incidentDayIndex]
  const currentDay = result.candidate.trajectory[currentDayIndex]
  if (serviceOffset < 0 || !incidentDay || !currentDay) return 1
  const target = incidentDay.services_before[serviceOffset]
  const low = incidentDay.services_after_shock[serviceOffset]
  if (target <= low + 1e-8) return 1
  for (let index = incidentDayIndex; index <= currentDayIndex; index += 1) {
    if ((result.candidate.trajectory[index]?.services_end[serviceOffset] ?? Number.NEGATIVE_INFINITY) >= target - 1e-7) {
      return 1
    }
  }
  return clamp((currentDay.services_end[serviceOffset] - low) / (target - low))
}

/**
 * The same incident recovery ratio sampled from a value on the shared visual
 * clock. Earlier exact returned days can permanently complete the incident;
 * otherwise the supplied level is bounded to the incident's returned low and
 * pre-event target. This is presentation-only and never creates a sub-day
 * simulator state.
 */
export function incidentRecoveryProgressAt(
  result: CompareResponse,
  incidentDayIndex: number,
  currentDayIndex: number,
  service: Service,
  presentedLevel: number,
): number {
  const serviceOffset = result.services.indexOf(service)
  const incidentDay = result.candidate.trajectory[incidentDayIndex]
  if (serviceOffset < 0 || !incidentDay || !Number.isFinite(presentedLevel)) return 1
  const target = incidentDay.services_before[serviceOffset]
  const low = incidentDay.services_after_shock[serviceOffset]
  if (target <= low + 1e-8) return 1
  for (let index = incidentDayIndex; index < currentDayIndex; index += 1) {
    if ((result.candidate.trajectory[index]?.services_end[serviceOffset] ?? Number.NEGATIVE_INFINITY) >= target - 1e-7) {
      return 1
    }
  }
  return clamp((presentedLevel - low) / (target - low))
}

export function strongestShockService(shock: Shock, services: readonly Service[]): Service {
  let strongest = 0
  shock.impact.forEach((value, index) => {
    if (value > (shock.impact[strongest] ?? -Infinity)) strongest = index
  })
  return services[strongest] ?? 'public_services'
}

export function intensityBand(type: ShockType, severity: number): string {
  const normalized = clamp((severity - 0.05) / 0.35)
  if (type === 'aftershock') return normalized < 0.34 ? 'moderate shaking' : normalized < 0.7 ? 'strong shaking' : 'severe shaking'
  if (type === 'weather') return normalized < 0.34 ? 'organized storm' : normalized < 0.7 ? 'major storm' : 'severe storm'
  if (type === 'epidemic') return normalized < 0.34 ? 'elevated case pressure' : normalized < 0.7 ? 'high case pressure' : 'severe case pressure'
  if (type === 'supply') return normalized < 0.34 ? 'limited disruption' : normalized < 0.7 ? 'major disruption' : 'severe disruption'
  return normalized < 0.34 ? 'localized outage' : normalized < 0.7 ? 'major outage' : 'severe outage'
}

export function recoveryArcForService(
  result: CompareResponse,
  dayIndex: number,
  service: Service,
): RecoveryArc {
  const index = result.services.indexOf(service)
  const trajectory = result.candidate.trajectory
  const current = trajectory[dayIndex]
  let startedIndex: number | null = null
  for (let cursor = dayIndex; cursor >= 0; cursor -= 1) {
    const item = trajectory[cursor]
    const priorEnd = trajectory[cursor - 1]?.services_end[index] ?? item.services_before[index]
    if (item.services_after_shock[index] < priorEnd - 1e-8) {
      startedIndex = cursor
      break
    }
  }
  if (startedIndex === null) {
    const currentLevel = current.services_end[index]
    return {
      service,
      startedDay: null,
      targetLevel: currentLevel,
      lowLevel: currentLevel,
      currentLevel,
      completionDay: current.day,
      durationDays: 0,
      progress: 1,
      stage: 'complete',
    }
  }
  const start = trajectory[startedIndex]
  const targetLevel = start.services_before[index]
  const lowLevel = start.services_after_shock[index]
  let completionIndex: number | null = null
  for (let cursor = startedIndex; cursor < trajectory.length; cursor += 1) {
    if (trajectory[cursor].services_end[index] >= targetLevel - 1e-7) {
      completionIndex = cursor
      break
    }
  }
  const currentLevel = current.services_end[index]
  const progress = targetLevel <= lowLevel + 1e-8
    ? 1
    : clamp((currentLevel - lowLevel) / (targetLevel - lowLevel))
  let stage: RecoveryStage = 'active-work'
  if (dayIndex === startedIndex) stage = 'assessment'
  else if (progress < 0.22) stage = 'debris'
  else if (progress < 0.48) stage = 'frame'
  else if (progress < 0.72) stage = 'wrap'
  else if (progress < 0.995) stage = 'active-work'
  else stage = 'complete'
  return {
    service,
    startedDay: start.day,
    targetLevel,
    lowLevel,
    currentLevel,
    completionDay: completionIndex === null ? null : trajectory[completionIndex].day,
    durationDays: (completionIndex ?? trajectory.length - 1) - startedIndex + 1,
    progress,
    stage,
  }
}

export function incidentPhaseForDay({
  result,
  dayIndex,
  telegraph,
  impact,
  postImpactPhase,
}: {
  result: CompareResponse
  dayIndex: number
  telegraph: boolean
  impact: boolean
  postImpactPhase?: 'assessment' | 'response' | null
}): IncidentPhase {
  if (telegraph) return 'TELEGRAPH'
  if (impact) return 'IMPACT'
  if (postImpactPhase === 'assessment') return 'ASSESSMENT'
  if (postImpactPhase === 'response') return 'RESPONSE'
  const day = result.candidate.trajectory[dayIndex]
  if (day.shock.type) return 'ASSESSMENT'
  const latest = latestIncident(result, dayIndex, 4)
  if (latest?.ageDays === 1) return 'RESPONSE'
  if (latest && day.services_end.some((value, index) => value > (result.candidate.trajectory[dayIndex - 1]?.services_end[index] ?? value) + 0.002)) return 'RECOVERY'
  return 'CLEAR'
}

export function decisionRationale(
  day: DayResult,
  services: readonly Service[],
  priorities: readonly number[] = [1, 1, 1, 1, 1],
): string {
  const assignments = services.map((service, index) => ({
    service,
    weightedDeficit: Math.max(0.001, priorities[index] ?? 1)
      * Math.max(0.001, 1 - day.services_after_shock[index]),
    allocation: day.allocation[index],
  }))
  const primary = [...assignments].sort((left, right) => right.allocation - left.allocation)[0]
  const deferred = [...assignments]
    .filter((entry) => entry.service !== primary.service)
    .sort((left, right) => left.allocation - right.allocation)[0] ?? primary
  const ratio = primary.weightedDeficit / Math.max(deferred.weightedDeficit, 0.001)
  const arrivalsReduced = Boolean(
    day.shock.type
    && day.shock.severity > 0
    && day.shock.budget_factor > 0,
  )
  const record = `RECORDED TRIAGE: ${SERVICE_LABELS[deferred.service].toUpperCase()} ${deferred.allocation.toFixed(1)} UNITS; ${SERVICE_LABELS[primary.service].toUpperCase()} ${primary.allocation.toFixed(1)} UNITS. WEIGHTED DEFICIT RATIO ${SERVICE_LABELS[primary.service].toUpperCase()}/${SERVICE_LABELS[deferred.service].toUpperCase()} ${ratio.toFixed(1)}×.`
  if (arrivalsReduced) {
    return `ARRIVALS REDUCED — RESEQUENCING. ${record}`
  }
  return record
}

export function dawnSitrep(day: DayResult, services: readonly Service[]): string {
  const primaryIndex = day.allocation.reduce((best, value, index, values) => value > values[best] ? index : best, 0)
  const primary = services[primaryIndex]
  const arrivals = day.shock.type && day.shock.severity > 0 && day.shock.budget_factor > 0
    ? 'REDUCED ARRIVALS'
    : 'FULL ARRIVALS'
  return `DAY ${day.day} SITREP — ${arrivals}; ${SERVICE_LABELS[primary].toUpperCase()} POINT OF DISTRIBUTION LEADS AT ${day.allocation[primaryIndex].toFixed(1)} UNITS.`
}

export function recoveryMilestones(
  result: CompareResponse,
  observedDays = result.candidate.trajectory.length,
): RecoveryMilestone[] {
  const trajectory = result.candidate.trajectory.slice(0, Math.max(0, observedDays))
  const milestones: RecoveryMilestone[] = []
  const thresholds: Array<{ service: Service; level: number; label: string }> = [
    { service: 'food', level: 0.58, label: 'Market distribution restored' },
    { service: 'transport', level: 0.55, label: 'Transit loop resumed' },
    { service: 'public_services', level: 0.60, label: 'Civic coordination restored' },
  ]
  thresholds.forEach(({ service, level, label }) => {
    const index = result.services.indexOf(service)
    const day = trajectory.find((entry, dayIndex) => (
      entry.services_end[index] >= level
      && (dayIndex === 0 || trajectory[dayIndex - 1].services_end[index] < level)
    ))
    if (day) milestones.push({ day: day.day, service, label, source: `services_end crossed ${level.toFixed(2)}` })
  })
  const housing = result.services.indexOf('housing')
  const transport = result.services.indexOf('transport')
  const returnDay = trajectory.find((entry, dayIndex) => (
    entry.services_end[housing] >= 0.65
    && entry.services_end[transport] >= 0.65
    && (dayIndex === 0 || trajectory[dayIndex - 1].services_end[housing] < 0.65 || trajectory[dayIndex - 1].services_end[transport] < 0.65)
  ))
  if (returnDay) milestones.push({
    day: returnDay.day,
    service: 'city',
    label: 'Return traffic and regular pickup loop resumed',
    source: 'housing and transport services_end crossed 0.65',
  })
  return milestones.sort((left, right) => left.day - right.day)
}

export function shockAdjustedArrivalShortfall(result: CompareResponse, observedDays = result.candidate.trajectory.length): number {
  const boundedDays = Math.max(0, Math.min(observedDays, result.candidate.trajectory.length))
  const calmTotal = result.scenario.daily_budget * boundedDays
  const actual = result.candidate.trajectory.slice(0, boundedDays).reduce((sum, day) => sum + day.available_budget, 0)
  return Number(Math.max(0, calmTotal - actual).toFixed(8))
}

export function afterActionReports(
  result: CompareResponse,
  observedDays = result.candidate.trajectory.length,
): DisasterAfterAction[] {
  const trajectory = result.candidate.trajectory.slice(0, Math.max(0, observedDays))
  return result.shock_schedule.flatMap((shock, shockIndex) => {
    if (!shock.type || shockIndex >= trajectory.length) return []
    const type = shock.type as ShockType
    const strongestService = strongestShockService(shock, result.services)
    const day = trajectory[shockIndex]
    if (!day) return []
    // Keep every debrief logistics count identical to the manifest selectors.
    const emergencyWaveVehicles = Math.min(4, 1 + Math.round(shock.severity * 8))
    const dispatchCounts = vehicleDispatchCountsForDay(day, trajectory[shockIndex - 1], result.services)
    const recoveryDays: Partial<Record<Service, number | null>> = {}
    result.services.forEach((service, serviceOffset) => {
      if ((shock.impact[serviceOffset] ?? 0) <= 0.05) return
      const target = day.services_before[serviceOffset]
      const recovered = trajectory
        .slice(shockIndex)
        .find((entry) => entry.services_end[serviceOffset] >= target - 1e-7)
      recoveryDays[service] = recovered ? recovered.day - shock.day : null
    })
    return [{
      day: shock.day,
      type,
      severity: shock.severity,
      strongestService,
      emergencyWaveVehicles,
      ...dispatchCounts,
      logisticsRecorded: Boolean(day.logistics),
      recoveryDays,
    }]
  })
}

export function deterministicUnit(seed: number, day: number, salt: number): number {
  let value = (seed ^ Math.imul(day + 1, 0x9e3779b1) ^ Math.imul(salt + 1, 0x85ebca6b)) >>> 0
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d)
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b)
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295
}

export function disasterWind(result: CompareResponse, day: number): readonly [number, number] {
  const angle = deterministicUnit(result.seed, day, 49) * Math.PI * 2
  return [Math.cos(angle), Math.sin(angle)] as const
}

export function buildingPlacard(
  level: number,
  buildingIndex: number,
): Placard {
  return placardForDamage(damageStateFor(level, buildingIndex))
}

/**
 * A current-day repair presentation needs both a realized service gain and, for
 * schema-v3 results, recorded usable repair supply. Legacy results retain only
 * their explicitly disclosed service-gain presentation because no depot ledger
 * exists to inspect.
 */
export function hasCurrentDayRepairWork(
  day: DayResult,
  previous: DayResult | undefined,
  service: Service,
  services: readonly Service[] = ['transport', 'housing', 'food', 'healthcare', 'public_services'],
): boolean {
  const index = services.indexOf(service)
  if (index < 0) return false
  const priorLevel = previous?.services_end[index] ?? day.services_after_shock[index]
  const realizedGain = (day.services_end[index] ?? priorLevel) - priorLevel
  if (realizedGain <= 0.002) return false
  if (!day.logistics) return true
  return (day.logistics.repair_supply[index] ?? 0) > 0.001
}

export function activeSiteCount(day: DayResult, previous: DayResult | undefined, services: readonly Service[]): number {
  return services.reduce((count, service) => (
    hasCurrentDayRepairWork(day, previous, service, services)
      ? count + rebuildingCohortForDay(day, previous, service).length
      : count
  ), 0)
}

export function throughputVehiclesPerDay(
  day: DayResult,
  previous?: DayResult,
  services: readonly Service[] = ['transport', 'housing', 'food', 'healthcare', 'public_services'],
): number {
  const counts = vehicleDispatchCountsForDay(day, previous, services)
  return counts.lineHaulHeavyTrucks + counts.lastMileVehicles + counts.mutualAidVehicles
}

function buildingRepairActivationIndex(
  result: CompareResponse,
  dayIndex: number,
  service: Service,
  buildingIndex: number,
): number | null {
  const arc = recoveryArcForService(result, dayIndex, service)
  if (arc.startedDay === null) return null
  const trajectory = result.candidate.trajectory
  const startIndex = trajectory.findIndex((entry) => entry.day === arc.startedDay)
  if (startIndex < 0) return null
  for (let index = startIndex; index <= dayIndex; index += 1) {
    const day = trajectory[index]
    const previous = trajectory[index - 1]
    const dayArc = recoveryArcForService(result, index, service)
    if (
      dayArc.progress < 0.995
      &&
      hasCurrentDayRepairWork(day, previous, service, result.services)
      && rebuildingCohortForDay(day, previous, service).includes(buildingIndex)
    ) return index
  }
  return null
}

export function buildingRepairStarted(
  result: CompareResponse,
  dayIndex: number,
  service: Service,
  buildingIndex: number,
): boolean {
  return buildingRepairActivationIndex(result, dayIndex, service, buildingIndex) !== null
}

export function repairProgressForBuilding(
  result: CompareResponse,
  dayIndex: number,
  service: Service,
  buildingIndex: number,
): number {
  const serviceOffset = result.services.indexOf(service)
  const current = result.candidate.trajectory[dayIndex]
  if (serviceOffset < 0 || !current) return 1
  const activationIndex = buildingRepairActivationIndex(result, dayIndex, service, buildingIndex)
  if (activationIndex === null) {
    return damageStateFor(current.services_end[serviceOffset], buildingIndex) === 'intact' ? 1 : 0
  }
  const arc = recoveryArcForService(result, dayIndex, service)
  const activationArc = recoveryArcForService(result, activationIndex, service)
  const remainingArc = Math.max(1e-7, 1 - activationArc.progress)
  const trajectoryProgress = clamp((arc.progress - activationArc.progress) / remainingArc)
  const initialTier = damageStateFor(arc.lowLevel, buildingIndex)
  // A slight repair resolves earlier within the same recorded service arc; rubble
  // consumes the full arc. No presentation progress advances when the trajectory
  // itself stalls.
  const tierDuration = initialTier === 'slight' ? 0.52 : initialTier === 'moderate' ? 0.76 : 1
  return clamp(trajectoryProgress / tierDuration)
}

/** Fresh material weathers for four recorded day ticks after this site's own completion. */
export function repairFreshnessForBuilding(
  result: CompareResponse,
  dayIndex: number,
  service: Service,
  buildingIndex: number,
): number {
  const activationIndex = buildingRepairActivationIndex(result, dayIndex, service, buildingIndex)
  if (activationIndex === null) return 0
  let completionIndex: number | null = null
  for (let index = activationIndex; index <= dayIndex; index += 1) {
    if (repairProgressForBuilding(result, index, service, buildingIndex) >= 0.995) {
      completionIndex = index
      break
    }
  }
  return completionIndex === null ? 0 : clamp(1 - (dayIndex - completionIndex) / 4)
}

export function repairStageForBuilding(
  result: CompareResponse,
  dayIndex: number,
  service: Service,
  buildingIndex: number,
): RecoveryStage {
  const progress = repairProgressForBuilding(result, dayIndex, service, buildingIndex)
  if (progress <= 0.02) return 'assessment'
  if (progress < 0.28) return 'debris'
  if (progress < 0.52) return 'frame'
  if (progress < 0.76) return 'wrap'
  if (progress < 0.995) return 'active-work'
  return 'complete'
}

function presentationEase(value: number): number {
  const normalized = clamp(value)
  return normalized * normalized * (3 - 2 * normalized)
}

/**
 * A deterministic within-day activation offset for a site's recorded repair
 * cohort. The offset changes only presentation cadence: the endpoints remain
 * the exact repair values selected from adjacent returned daily records.
 */
export function buildingRepairPresentationOffset(
  result: CompareResponse,
  dayIndex: number,
  service: Service,
  buildingIndex: number,
): number {
  const serviceOffset = result.services.indexOf(service)
  const day = result.candidate.trajectory[dayIndex]
  if (serviceOffset < 0 || !day) return 0
  const activationIndex = buildingRepairActivationIndex(result, dayIndex, service, buildingIndex)
  if (activationIndex === null || activationIndex > dayIndex) return 0
  return 0.06 + deterministicUnit(
    result.seed,
    day.day,
    1_700 + serviceOffset * CITY_BUILDINGS_PER_DISTRICT + buildingIndex,
  ) * 0.56
}

/**
 * Visual-only repair progress between two exact day endpoints. Newly activated
 * buildings receive stable per-site offsets so a district repairs as a cohort,
 * never as one day-boundary swap. No simulator sub-day dynamics are implied.
 */
export function repairProgressForBuildingAt(
  result: CompareResponse,
  dayIndex: number,
  dayProgress: number,
  service: Service,
  buildingIndex: number,
): number {
  const current = result.candidate.trajectory[dayIndex]
  if (!current) return 1
  const previous = result.candidate.trajectory[dayIndex - 1]
  const serviceOffset = result.services.indexOf(service)
  const shockReset = serviceOffset >= 0
    && (current.services_after_shock[serviceOffset] ?? 0) < (current.services_before[serviceOffset] ?? 0) - 1e-8
  const start = shockReset || !previous
    ? 0
    : repairProgressForBuilding(result, dayIndex - 1, service, buildingIndex)
  const end = repairProgressForBuilding(result, dayIndex, service, buildingIndex)
  if (Math.abs(end - start) <= 1e-9) return end
  const offset = end > start
    ? buildingRepairPresentationOffset(result, dayIndex, service, buildingIndex)
    : 0
  const local = presentationEase((clamp(dayProgress) - offset) / Math.max(1e-7, 1 - offset))
  return clamp(start + (end - start) * local)
}

export function buildingRepairStartedAt(
  result: CompareResponse,
  dayIndex: number,
  dayProgress: number,
  service: Service,
  buildingIndex: number,
): boolean {
  const activationIndex = buildingRepairActivationIndex(result, dayIndex, service, buildingIndex)
  if (activationIndex === null || activationIndex > dayIndex) return false
  if (activationIndex < dayIndex) return true
  return clamp(dayProgress) >= buildingRepairPresentationOffset(
    result,
    dayIndex,
    service,
    buildingIndex,
  )
}

export function repairFreshnessForBuildingAt(
  result: CompareResponse,
  dayIndex: number,
  dayProgress: number,
  service: Service,
  buildingIndex: number,
): number {
  const end = repairFreshnessForBuilding(result, dayIndex, service, buildingIndex)
  const start = dayIndex > 0
    ? repairFreshnessForBuilding(result, dayIndex - 1, service, buildingIndex)
    : 0
  return clamp(start + (end - start) * presentationEase(dayProgress))
}

export function repairStageForBuildingAt(
  result: CompareResponse,
  dayIndex: number,
  dayProgress: number,
  service: Service,
  buildingIndex: number,
): RecoveryStage {
  const progress = repairProgressForBuildingAt(result, dayIndex, dayProgress, service, buildingIndex)
  if (progress <= 0.02) return 'assessment'
  if (progress < 0.28) return 'debris'
  if (progress < 0.52) return 'frame'
  if (progress < 0.76) return 'wrap'
  if (progress < 0.995) return 'active-work'
  return 'complete'
}

export function serviceAllocation(day: DayResult, service: Service): number {
  return day.allocation[serviceIndex(service)] ?? 0
}
