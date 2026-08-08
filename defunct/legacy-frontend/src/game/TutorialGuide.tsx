import { BookOpen, Check } from 'lucide-react'
import { shockDisplayName } from '../shockPresentation'
import type { CompareResponse, Service } from '../types'
import { SERVICE_LABELS } from './model'
import { strongestShockService, type IncidentPhase } from './realism'

export const TUTORIAL_PHASES = ['TELEGRAPH', 'IMPACT', 'ASSESSMENT', 'RESPONSE', 'RECOVERY'] as const
export type TutorialPhase = (typeof TUTORIAL_PHASES)[number]

export type TutorialLesson = {
  phase: TutorialPhase
  heading: string
  explanation: string
  evidence: string
  source: string
}

function largestIndex(values: readonly number[]): number {
  return values.reduce((best, value, index) => value > (values[best] ?? Number.NEGATIVE_INFINITY) ? index : best, 0)
}

function serviceLabel(service: Service | undefined): string {
  return service ? SERVICE_LABELS[service] : 'Unrecorded service'
}

export function tutorialPhaseForDay(dayIndex: number, incidentPhase: IncidentPhase): TutorialPhase {
  if (dayIndex <= 0) return 'TELEGRAPH'
  if (dayIndex === 1 && (incidentPhase === 'IMPACT' || incidentPhase === 'ASSESSMENT' || incidentPhase === 'RESPONSE')) {
    return incidentPhase
  }
  if (dayIndex <= 2) return 'RESPONSE'
  return 'RECOVERY'
}

/**
 * Every number in the tutorial comes from the returned comparison. The instructional
 * prose names presentation phases only; it never introduces an unmodelled intervention.
 */
export function tutorialLessonFor(
  result: CompareResponse,
  dayIndex: number,
  incidentPhase: IncidentPhase,
): TutorialLesson {
  const phase = tutorialPhaseForDay(dayIndex, incidentPhase)
  const incidentIndex = result.shock_schedule.findIndex((shock) => shock.type !== null)
  const safeIncidentIndex = Math.max(0, incidentIndex)
  const incident = result.shock_schedule[safeIncidentIndex]
  const incidentDay = result.candidate.trajectory[safeIncidentIndex]
  const current = result.candidate.trajectory[dayIndex] ?? incidentDay
  const previous = result.candidate.trajectory[dayIndex - 1]
  const strongestService = incident?.type
    ? strongestShockService(incident, result.services)
    : result.services[0]
  const strongestIndex = result.services.indexOf(strongestService)
  const strongestImpact = incident?.impact[strongestIndex] ?? 0
  const immediateLoss = incidentDay
    ? Math.max(0, (incidentDay.services_before[strongestIndex] ?? 0) - (incidentDay.services_after_shock[strongestIndex] ?? 0))
    : 0

  if (phase === 'TELEGRAPH') {
    return {
      phase,
      heading: 'Read the recorded arrival before it lands.',
      explanation: 'The view is staging the next returned shock record. No forecast or extra event has been invented.',
      evidence: incident?.type
        ? `Returned day ${incident.day}: ${shockDisplayName(incident.type)} at raw ${incident.severity.toFixed(2)}; strongest typed footprint ${serviceLabel(strongestService)} at ${strongestImpact.toFixed(2)}.`
        : 'The returned shock tape contains no incident to stage.',
      source: `shock_schedule[${safeIncidentIndex}]`,
    }
  }

  if (phase === 'IMPACT') {
    return {
      phase,
      heading: 'Separate the typed strike from the response.',
      explanation: 'The impact footprint changes service condition first; allocation and logistics follow from the same returned day.',
      evidence: incidentDay
        ? `${serviceLabel(strongestService)} moved from ${(incidentDay.services_before[strongestIndex] * 100).toFixed(1)}% to ${(incidentDay.services_after_shock[strongestIndex] * 100).toFixed(1)}% before recovery allocation, an immediate ${(immediateLoss * 100).toFixed(1)}-point loss.`
        : 'No returned incident day is available.',
      source: `candidate.trajectory[${safeIncidentIndex}].services_before / services_after_shock`,
    }
  }

  if (phase === 'ASSESSMENT') {
    const lossByService = incidentDay?.services_before.map((value, index) => (
      Math.max(0, value - incidentDay.services_after_shock[index])
    )) ?? []
    const assessedIndex = largestIndex(lossByService)
    return {
      phase,
      heading: 'Assessment reads the returned service losses.',
      explanation: 'RELAY identifies the largest immediate drop from the trajectory; the view does not infer damage counts or casualties.',
      evidence: incidentDay
        ? `${serviceLabel(result.services[assessedIndex])} has the largest immediate recorded drop: ${((lossByService[assessedIndex] ?? 0) * 100).toFixed(1)} points. Available arrivals are ${incidentDay.available_budget.toFixed(1)} of ${result.scenario.daily_budget.toFixed(1)} units.`
        : 'No returned assessment record is available.',
      source: `candidate.trajectory[${safeIncidentIndex}].services_before / services_after_shock / available_budget`,
    }
  }

  if (phase === 'RESPONSE') {
    const allocationIndex = largestIndex(current?.allocation ?? [])
    const allocated = current?.allocation[allocationIndex] ?? 0
    const repairSupply = current?.logistics?.repair_supply[allocationIndex]
    const allocationTotal = current?.allocation.reduce((sum, value) => sum + value, 0) ?? 0
    return {
      phase,
      heading: 'Response follows the returned allocation and depot ledger.',
      explanation: 'The planner assigns every available arrival unit. Visible freight and repair movement use these same quantities.',
      evidence: current
        ? `${serviceLabel(result.services[allocationIndex])} leads at ${allocated.toFixed(1)} units; ${allocationTotal.toFixed(1)} of ${current.available_budget.toFixed(1)} available units are assigned.${repairSupply === undefined ? ' This restored legacy result has no recorded depot ledger.' : ` Its recorded effective repair supply is ${repairSupply.toFixed(1)} units.`}`
        : 'No returned response day is available.',
      source: `candidate.trajectory[${dayIndex}].allocation / available_budget / logistics.repair_supply`,
    }
  }

  const incidentTarget = incidentDay?.services_before[strongestIndex]
  const crossingIndex = incidentTarget === undefined
    ? -1
    : result.candidate.trajectory.findIndex((entry, index) => (
        index >= safeIncidentIndex
        && index <= dayIndex
        && (entry.services_end[strongestIndex] ?? Number.NEGATIVE_INFINITY) >= incidentTarget - 1e-7
      ))
  if (current && incidentTarget !== undefined && crossingIndex < 0) {
    const incidentCurrent = current.services_end[strongestIndex] ?? incidentTarget
    const incidentPrevious = previous?.services_end[strongestIndex]
      ?? current.services_after_shock[strongestIndex]
      ?? incidentCurrent
    const incidentChange = incidentCurrent - incidentPrevious
    const repairSupply = current.logistics?.repair_supply[strongestIndex]
    return {
      phase,
      heading: 'Follow the incident service only to its returned target.',
      explanation: `${serviceLabel(strongestService)} remains below its pre-event returned level, so this is still the same incident recovery arc. The guide will stop that claim as soon as the trajectory reaches the target.`,
      evidence: `${serviceLabel(strongestService)} changes ${incidentChange >= 0 ? '+' : ''}${(incidentChange * 100).toFixed(1)} points on day ${current.day}, reaching ${(incidentCurrent * 100).toFixed(1)}% against its ${(incidentTarget * 100).toFixed(1)}% pre-event target.${repairSupply === undefined ? ' Depot state was not recorded in this legacy result.' : ` Recorded effective repair supply is ${repairSupply.toFixed(1)} units.`}`,
      source: `candidate.trajectory[${safeIncidentIndex}].services_before / candidate.trajectory[${dayIndex}].services_end${current.logistics ? ' / logistics.repair_supply' : ''}`,
    }
  }

  const change = current?.services_end.map((value, index) => (
    value - (previous?.services_end[index] ?? current.services_after_shock[index])
  )) ?? []
  const improvementIndex = largestIndex(change)
  const improvement = change[improvementIndex] ?? 0
  const repairSupply = current?.logistics?.repair_supply[improvementIndex]
  const crossingDay = crossingIndex < 0 ? null : result.candidate.trajectory[crossingIndex]?.day ?? null
  return {
    phase,
    heading: 'The incident target is restored; widen the view.',
    explanation: `${serviceLabel(strongestService)} crossed its pre-event target${crossingDay === null ? '' : ` on returned day ${crossingDay}`}. From here, RELAY follows the broader city's current returned change and does not extend that incident recovery arc.`,
    evidence: current
      ? improvement > 1e-7
        ? `${serviceLabel(result.services[improvementIndex])} has the largest current-day city improvement: +${(improvement * 100).toFixed(1)} points on day ${current.day}.${repairSupply === undefined ? ' Depot state was not recorded in this legacy result.' : ` Recorded effective repair supply is ${repairSupply.toFixed(1)} units.`}`
        : `No service records a positive current-day improvement on day ${current.day}; the guide makes no continuing recovery claim.`
      : 'No returned recovery day is available.',
    source: `candidate.trajectory[${safeIncidentIndex}].services_before / candidate.trajectory[${dayIndex}].services_end${current?.logistics ? ' / logistics.repair_supply' : ''}`,
  }
}

export function TutorialGuide({
  result,
  dayIndex,
  incidentPhase,
}: {
  result: CompareResponse
  dayIndex: number
  incidentPhase: IncidentPhase
}) {
  const lesson = tutorialLessonFor(result, dayIndex, incidentPhase)
  const activeIndex = TUTORIAL_PHASES.indexOf(lesson.phase)
  return (
    <aside className="tutorial-guide" aria-labelledby="tutorial-guide-heading" aria-live="polite">
      <header>
        <BookOpen size={16} strokeWidth={1.8} aria-hidden="true" />
        <div><span>RELAY guide · Day {dayIndex + 1} of 8</span><b id="tutorial-guide-heading">{lesson.phase}</b></div>
      </header>
      <ol aria-label="Tutorial incident phases">
        {TUTORIAL_PHASES.map((phase, index) => (
          <li key={phase} data-active={phase === lesson.phase} data-complete={index < activeIndex}>
            {index < activeIndex ? <Check size={10} aria-hidden="true" /> : <i aria-hidden="true" />}
            <span>{phase}</span>
          </li>
        ))}
      </ol>
      <h2>{lesson.heading}</h2>
      <p>{lesson.explanation}</p>
      <strong>{lesson.evidence}</strong>
      <small>Source: {lesson.source}</small>
    </aside>
  )
}
