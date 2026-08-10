import type { CompareResponse, DayResult, Service, Shock, ShockType } from '../types'
import {
  DISTRICTS,
  SERVICE_LABELS,
  damageStateFor,
  rebuildingCohortForDay,
  serviceIndex,
  type DamageState,
} from './model'

export const HEAVY_TRUCK_CAPACITY = 12
export const LAST_MILE_CAPACITY = 4

export type Placard = 'GREEN' | 'YELLOW' | 'RED'
export type IncidentPhase = 'CLEAR' | 'TELEGRAPH' | 'IMPACT' | 'ASSESSMENT' | 'RESPONSE' | 'RECOVERY'
export type RecoveryStage = 'assessment' | 'debris' | 'frame' | 'wrap' | 'active-work' | 'complete'

export type DepotStatus = {
  service: Service
  district: string
  damage: DamageState
  placard: Placard
  serviceLevel: number
  allocationUnits: number
  palletUnits: number
  stockCapacity: number
  pendingUnits: number
  spoilageUnits: number
  throughputSignal: number
  dispatchedVehicles: number
  lastMileVehicles: number
  dockQueue: number
  reroutedFrom: Service | null
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
    queuedNextDayUnits: number
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
  wave: 'inbound' | 'line-haul' | 'mutual-aid' | 'last-mile' | 'emergency' | 'maintenance' | 'civilian'
  dispatchRank: number
}

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
  recoveryWaveVehicles: number
  recoveryDays: Partial<Record<Service, number | null>>
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

export function depotDamageState(serviceLevel: number): DamageState {
  if (serviceLevel >= 0.72) return 'intact'
  if (serviceLevel >= 0.48) return 'slight'
  if (serviceLevel >= 0.24) return 'moderate'
  return 'rubble'
}

export function depotDamageStateFromFactor(damageFactor: number): DamageState {
  // The canonical depot function retains a deliberate 0.30 floor.
  // These four bands cover the full runtime throughput range.
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

function nearestHealthyDepot(
  service: Service,
  day: DayResult,
  services: readonly Service[],
): Service | null {
  const target = DISTRICTS.find((district) => district.service === service)
  if (!target) return null
  return DISTRICTS
    .filter((district) => (
      district.service !== service
      && depotDamageState(levelFor(day, district.service, services)) !== 'rubble'
    ))
    .sort((left, right) => (
      Math.hypot(left.center[0] - target.center[0], left.center[2] - target.center[2])
      - Math.hypot(right.center[0] - target.center[0], right.center[2] - target.center[2])
    ))[0]?.service ?? null
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

export function depotStatusesForDay(
  day: DayResult,
  services: readonly Service[],
): DepotStatus[] {
  const dispatchOrder = services
    .map((service, index) => ({ service, allocation: day.allocation[index] ?? 0 }))
    .sort((left, right) => right.allocation - left.allocation)
  return DISTRICTS.map((district) => {
    const index = services.indexOf(district.service)
    const ledger = day.logistics
    const serviceLevel = day.services_end[index] ?? 0
    const allocationUnits = Math.max(0, day.allocation[index] ?? 0)
    const damageFactor = ledger?.depot_damage_factor[index]
    const damage = damageFactor === undefined
      ? depotDamageState(serviceLevel)
      : depotDamageStateFromFactor(damageFactor)
    const throughputSignal = clamp(ledger?.throughput_factor[index] ?? (0.18 + serviceLevel * 0.82))
    const pendingUnits = Math.max(0, ledger?.pending_next_day[index] ?? 0)
    const capacityHeldUnits = Math.max(0, ledger?.capacity_overflow[index] ?? 0)
    const deliveredUnits = ledger
      ? Math.max(0, (ledger.pending_arrivals_landed[index] ?? 0) + (ledger.same_day_delivery_landed[index] ?? 0))
      : allocationUnits
    const repairSupply = Math.max(0, ledger?.repair_supply[index] ?? allocationUnits)
    const dispatchedVehicles = splitExactCargo(deliveredUnits, HEAVY_TRUCK_CAPACITY).length
    const lastMileVehicles = splitExactCargo(repairSupply, LAST_MILE_CAPACITY).length
    const rank = dispatchOrder.findIndex((entry) => entry.service === district.service)
    return {
      service: district.service,
      district: district.label,
      damage,
      placard: placardForDamage(damage),
      serviceLevel,
      allocationUnits,
      palletUnits: Math.max(0, ledger?.depot_stock_end[index] ?? allocationUnits),
      stockCapacity: Math.max(0, ledger?.depot_capacity[index] ?? allocationUnits),
      pendingUnits,
      spoilageUnits: Math.max(0, ledger?.spoilage[index] ?? 0),
      throughputSignal,
      dispatchedVehicles,
      lastMileVehicles,
      dockQueue: ledger
        ? splitExactCargo(pendingUnits, HEAVY_TRUCK_CAPACITY).length
        : damage === 'intact' ? 0 : Math.ceil(dispatchedVehicles * (1 - throughputSignal)),
      reroutedFrom: ledger
        ? mutualAidDonor(day, index, services)
        : damage === 'rubble' ? nearestHealthyDepot(district.service, day, services) : null,
      inboundWindow: ledger
        ? pendingUnits > 1e-7
          ? `${pendingUnits.toFixed(1)} units queued for day ${day.day + 1}${capacityHeldUnits > 1e-7 ? ` · ${capacityHeldUnits.toFixed(1)} capacity-held` : ''}`
          : 'no next-day freight queued'
        : rank <= 0 ? 'first recovery wave' : rank === 1 ? 'early recovery wave' : 'later recovery wave',
      source: ledger
        ? 'candidate.trajectory[day].logistics — recorded runtime stock, throughput, queue, and transfer state'
        : 'runtime logistics unavailable',
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
        throughputFactor: throughputSignal,
        queuedNextDayUnits: pendingUnits,
        spoilageUnits: Math.max(0, ledger.spoilage[index] ?? 0),
        mutualAidNetUnits: ledger.mutual_aid_net[index] ?? 0,
      } : null,
    }
  })
}

function serviceDestination(service: Service, buildingIndex: number): string {
  const district = DISTRICTS.find((item) => item.service === service)
  return `${district?.shortLabel ?? SERVICE_LABELS[service]} site ${buildingIndex + 1}`
}

export function vehicleManifestsForDay(
  day: DayResult,
  previous: DayResult | undefined,
  services: readonly Service[],
): VehicleManifest[] {
  const allocationRank = services
    .map((service, index) => ({ service, allocation: day.allocation[index] ?? 0 }))
    .sort((left, right) => right.allocation - left.allocation)
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
        destination: `${depot?.shortLabel ?? service} point of distribution`,
        cargoUnits: cargo,
        cargoPallets: cargo,
        returnLeg: 'returns empty to hub staging',
        wave: 'line-haul',
        dispatchRank: rank,
      })
    })
    const buildingSites = rebuildingCohortForDay(day, previous, service)
    const destinations = buildingSites.length ? buildingSites : [rank * 7 % 36]
    splitExactCargo(lastMileUnits, LAST_MILE_CAPACITY).forEach((cargo, index) => {
      const buildingIndex = destinations[index % destinations.length]
      manifests.push({
        id: `d${day.day}-${service}-last-${index}`,
        service,
        fleet: FLEET_LABELS[service],
        origin: `${depot?.shortLabel ?? service} point of distribution`,
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

export function decisionRationale(day: DayResult, services: readonly Service[]): string {
  const weightedDeficits = services.map((service, index) => ({
    service,
    score: Math.max(0.001, 1 - day.services_after_shock[index]) * Math.max(0.001, day.allocation[index]),
    allocation: day.allocation[index],
  })).sort((left, right) => right.score - left.score)
  const primary = weightedDeficits[0]
  const deferred = [...weightedDeficits]
    .filter((entry) => entry.service !== primary.service)
    .sort((left, right) => left.allocation - right.allocation)[0] ?? primary
  const ratio = primary.score / Math.max(deferred.score, 0.001)
  const arrivalsReduced = Boolean(
    day.shock.type
    && day.shock.severity > 0
    && day.shock.budget_factor > 0,
  )
  if (arrivalsReduced) {
    return `ARRIVALS REDUCED — RESEQUENCING. ${SERVICE_LABELS[deferred.service].toUpperCase()} SEQUENCED LATER — ${SERVICE_LABELS[primary.service].toUpperCase()} DEFICIT SIGNAL ${ratio.toFixed(1)}× WEIGHTED.`
  }
  return `${SERVICE_LABELS[deferred.service].toUpperCase()} SEQUENCED LATER — ${SERVICE_LABELS[primary.service].toUpperCase()} DEFICIT SIGNAL ${ratio.toFixed(1)}× WEIGHTED.`
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

export function shockTaxedArrivalShortfall(result: CompareResponse, observedDays = result.candidate.trajectory.length): number {
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
    // Keep the debrief count identical to the rendered fleet selectors. The first
    // wave is the bounded rapid-assessment fleet in VehicleFleet; the heavy wave is
    // the sum of allocation-backed line-haul loads, not the edge-to-hub arrival count.
    const emergencyWaveVehicles = Math.min(4, 1 + Math.round(shock.severity * 8))
    const recoveryWaveVehicles = throughputVehiclesPerDay(day)
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
      recoveryWaveVehicles,
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

export function activeSiteCount(day: DayResult, previous: DayResult | undefined, services: readonly Service[]): number {
  return services.reduce((count, service) => count + rebuildingCohortForDay(day, previous, service).length, 0)
}

export function throughputVehiclesPerDay(day: DayResult): number {
  const ledger = day.logistics
  if (!ledger) {
    return day.allocation.reduce(
      (count, allocation) => count + splitExactCargo(allocation, HEAVY_TRUCK_CAPACITY).length,
      0,
    )
  }
  const landed = ledger.pending_arrivals_landed.map(
    (units, index) => Math.max(0, units + (ledger.same_day_delivery_landed[index] ?? 0)),
  )
  const lineHaul = landed.reduce(
    (count, units) => count + splitExactCargo(units, HEAVY_TRUCK_CAPACITY).length,
    0,
  )
  const lastMile = ledger.repair_supply.reduce(
    (count, units) => count + splitExactCargo(units, LAST_MILE_CAPACITY).length,
    0,
  )
  const mutualAid = ledger.mutual_aid_transfers.reduce(
    (count, transfer) => count + splitExactCargo(transfer.units, HEAVY_TRUCK_CAPACITY).length,
    0,
  )
  return lineHaul + lastMile + mutualAid
}

export function repairProgressForBuilding(
  result: CompareResponse,
  dayIndex: number,
  service: Service,
  buildingIndex: number,
): number {
  const arc = recoveryArcForService(result, dayIndex, service)
  if (arc.startedDay === null) return 1
  const stagger = (buildingIndex % 12) / 60
  return clamp((arc.progress - stagger) / Math.max(0.25, 1 - stagger))
}

export function serviceAllocation(day: DayResult, service: Service): number {
  return day.allocation[serviceIndex(service)] ?? 0
}
