import { services as defaultServices, type CompareResponse, type DayResult, type Service } from '../types'

export const CRITICAL_FLOOR = 0.12

export const ESSENTIAL_SERVICES = ['food', 'healthcare'] as const satisfies readonly Service[]

export type EssentialFallCause = {
  kind: 'essential'
  services: Service[]
  consecutiveDays: 4
}

export type CascadeFallCause = {
  kind: 'cascade'
  services: Service[]
  consecutiveDays: 2
}

export type FallCause = EssentialFallCause | CascadeFallCause

export type FallEvent = {
  day: number
  causes: FallCause[]
}

export type CityDayCondition = {
  day: number
  stumble: boolean
  belowFloor: Service[]
  darkServices: Service[]
  underFloorStreaks: Record<Service, number>
  cascadeStreak: number
  fall: FallEvent | null
}

export type DarkPeriod = {
  service: Service
  startedDay: number
  endedDay: number | null
  recoveredDay: number | null
}

export type RecoveryEvent = {
  service: Service
  day: number
}

export type WorstMoment = {
  day: number
  wellbeing: number
  weakestService: Service
  weakestLevel: number
  belowFloor: Service[]
}

export type CityOutcome = {
  conditions: CityDayCondition[]
  firstStumbleDay: number | null
  darkPeriods: DarkPeriod[]
  fall: FallEvent | null
  survived: boolean
  terminalDay: number | null
  worstMoment: WorstMoment | null
  recoveryEvents: RecoveryEvent[]
  recoveryCount: number
  finalWellbeing: number | null
  resilienceAuc: number | null
}

export type DisasterTally = {
  ambient: number
  player: number
  total: number
}

export type RunDebrief = {
  candidate: CityOutcome
  baseline: CityOutcome
  disasters: DisasterTally
  conventionalCounterfactual: string
}

const SERVICE_LABELS: Record<Service, string> = {
  transport: 'Transport',
  housing: 'Housing',
  food: 'Food',
  healthcare: 'Healthcare',
  public_services: 'Public services',
}

function emptyStreaks(): Record<Service, number> {
  return {
    transport: 0,
    housing: 0,
    food: 0,
    healthcare: 0,
    public_services: 0,
  }
}

function levelFor(day: DayResult, serviceOrder: readonly Service[], service: Service): number {
  const index = serviceOrder.indexOf(service)
  return index < 0 ? Number.POSITIVE_INFINITY : day.services_end[index]
}

function mean(values: readonly number[]): number | null {
  if (values.length === 0) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

/**
 * Applies the game-layer condition rules to end-of-day service values. The scan stops at
 * the first fall because a fallen run ends on that day; no later precomputed sim days are
 * allowed to create recoveries or change debrief metrics.
 */
export function deriveCityOutcome(
  trajectory: readonly DayResult[],
  serviceOrder: readonly Service[] = defaultServices,
  criticalFloor = CRITICAL_FLOOR,
): CityOutcome {
  const streaks = emptyStreaks()
  const wasBelow = new Set<Service>()
  const darkPeriods: DarkPeriod[] = []
  const openDarkPeriods = new Map<Service, DarkPeriod>()
  const recoveryEvents: RecoveryEvent[] = []
  const conditions: CityDayCondition[] = []
  const observedDays: DayResult[] = []
  let cascadeStreak = 0
  let firstStumbleDay: number | null = null
  let fall: FallEvent | null = null
  let worstMoment: WorstMoment | null = null
  let previousDayNumber: number | null = null

  for (const day of trajectory) {
    const isConsecutive = previousDayNumber === null || day.day === previousDayNumber + 1
    if (!isConsecutive) {
      for (const service of defaultServices) streaks[service] = 0
      cascadeStreak = 0
    }

    const belowFloor: Service[] = []
    for (const service of serviceOrder) {
      const level = levelFor(day, serviceOrder, service)
      if (level < criticalFloor) {
        belowFloor.push(service)
        streaks[service] = (isConsecutive ? streaks[service] : 0) + 1
      } else {
        if (wasBelow.has(service)) recoveryEvents.push({ service, day: day.day })
        streaks[service] = 0
        const openPeriod = openDarkPeriods.get(service)
        if (openPeriod) {
          openPeriod.endedDay = previousDayNumber
          openPeriod.recoveredDay = day.day
          openDarkPeriods.delete(service)
        }
      }
    }

    if (belowFloor.length > 0 && firstStumbleDay === null) firstStumbleDay = day.day

    for (const service of serviceOrder) {
      if (streaks[service] === 3 && !openDarkPeriods.has(service)) {
        const period: DarkPeriod = {
          service,
          startedDay: day.day,
          endedDay: null,
          recoveredDay: null,
        }
        darkPeriods.push(period)
        openDarkPeriods.set(service, period)
      }
    }

    cascadeStreak = belowFloor.length >= 2 ? cascadeStreak + 1 : 0
    const essentialServices = ESSENTIAL_SERVICES.filter(
      (service) => streaks[service] >= 4,
    )
    const causes: FallCause[] = []
    if (essentialServices.length > 0) {
      causes.push({
        kind: 'essential',
        services: [...essentialServices],
        consecutiveDays: 4,
      })
    }
    if (cascadeStreak >= 2) {
      causes.push({
        kind: 'cascade',
        services: [...belowFloor],
        consecutiveDays: 2,
      })
    }

    const dayFall: FallEvent | null = causes.length > 0 ? { day: day.day, causes } : null
    const darkServices = serviceOrder.filter((service) => openDarkPeriods.has(service))
    conditions.push({
      day: day.day,
      stumble: belowFloor.length > 0,
      belowFloor: [...belowFloor],
      darkServices,
      underFloorStreaks: { ...streaks },
      cascadeStreak,
      fall: dayFall,
    })
    observedDays.push(day)

    let weakestService = serviceOrder[0] ?? defaultServices[0]
    let weakestLevel = levelFor(day, serviceOrder, weakestService)
    for (const service of serviceOrder.slice(1)) {
      const level = levelFor(day, serviceOrder, service)
      if (level < weakestLevel) {
        weakestService = service
        weakestLevel = level
      }
    }
    if (worstMoment === null || day.resilience < worstMoment.wellbeing) {
      worstMoment = {
        day: day.day,
        wellbeing: day.resilience,
        weakestService,
        weakestLevel,
        belowFloor: [...belowFloor],
      }
    }

    wasBelow.clear()
    for (const service of belowFloor) wasBelow.add(service)
    previousDayNumber = day.day

    if (dayFall) {
      fall = dayFall
      break
    }
  }

  const terminalCondition = conditions.at(-1)
  const terminalDayResult = observedDays.at(-1) ?? null
  const activeResilience = observedDays.map((day) => day.resilience)

  return {
    conditions,
    firstStumbleDay,
    darkPeriods,
    fall,
    survived: fall === null,
    terminalDay: terminalCondition?.day ?? null,
    worstMoment,
    recoveryEvents,
    recoveryCount: recoveryEvents.length,
    finalWellbeing: terminalDayResult?.resilience ?? null,
    resilienceAuc: mean(activeResilience),
  }
}

function formatPercent(value: number | null): string {
  return value === null ? 'an unavailable' : `${Math.round(value * 100)}%`
}

/** A deterministic, factual baseline sentence for the end-of-run debrief. */
export function describeConventionalPlanner(outcome: CityOutcome): string {
  const darkPeriod = outcome.darkPeriods[0]
  let darkClause = ' without any district going dark'
  if (darkPeriod?.recoveredDay !== null && darkPeriod?.recoveredDay !== undefined) {
    darkClause = ` after ${SERVICE_LABELS[darkPeriod.service]} went dark on day ${darkPeriod.startedDay} and recovered on day ${darkPeriod.recoveredDay}`
  } else if (darkPeriod) {
    darkClause = `, with ${SERVICE_LABELS[darkPeriod.service]} dark from day ${darkPeriod.startedDay}`
  }

  if (outcome.fall) {
    return `A conventional rule-based planner falls on day ${outcome.fall.day} in this same run at ${formatPercent(outcome.finalWellbeing)} weighted wellbeing${darkClause}.`
  }

  return `A conventional rule-based planner ends this same run at ${formatPercent(outcome.finalWellbeing)} weighted wellbeing${darkClause}.`
}

export function deriveRunDebrief(result: CompareResponse): RunDebrief {
  const candidate = deriveCityOutcome(result.candidate.trajectory, result.services)
  const baseline = deriveCityOutcome(result.baseline.trajectory, result.services)
  const terminalDay = candidate.terminalDay ?? Number.POSITIVE_INFINITY
  const endured = result.shock_schedule.filter(
    (shock) => shock.type !== null && shock.day <= terminalDay,
  )
  // `forced` also covers the scenario's legacy authored singular shock. Only days present
  // in the additive kick list are player disasters. A day is counted once even if several
  // appended entries target it, because the returned shock schedule is the event endured.
  const playerShockDays = new Set((result.scenario.forced_shocks ?? []).map((shock) => shock.day))
  const player = endured.filter((shock) => shock.forced && playerShockDays.has(shock.day)).length
  const disasters = {
    ambient: endured.length - player,
    player,
    total: endured.length,
  }

  return {
    candidate,
    baseline,
    disasters,
    conventionalCounterfactual: describeConventionalPlanner(baseline),
  }
}
