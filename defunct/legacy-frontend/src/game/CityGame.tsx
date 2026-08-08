import { Canvas } from '@react-three/fiber'
import { BarChart3, Gauge, Pause, Play, Rotate3D, Volume2, VolumeX, X } from 'lucide-react'
import { Component, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ComponentProps, type DragEvent, type ErrorInfo, type ReactNode } from 'react'
import * as THREE from 'three'
import { ComparisonError, runComparison } from '../api'
import { defaultScenario, defaultSeed } from '../scenarios'
import { shockDisplayName } from '../shockPresentation'
import { shockTypes, type CompareResponse, type Service, type ShockType } from '../types'
import { CityScene } from './CityScene'
import { DISASTER_STRIKE_SETTLE_END_FRACTION } from './DisasterEffects'
import { DisasterTray } from './DisasterTray'
import { QualityMonitor } from './QualityMonitor'
import { DISTRICTS, appendForcedShock, closestDistrict, rebuildingCohortForDay, relayNarration, type CityImpactEvent, type DistrictDefinition } from './model'
import type { PlaybackSpeed } from './pacing'
import { presentationIncidentStage, sampleRunPresentation } from './presentation'
import { createPresentationClock, type PresentationClockStore } from './presentationClock'
import { PresentationEvidenceBridge } from './PresentationEvidenceBridge'
import { CollapseScreen, RunDebriefScreen } from './RunOutcome'
import {
  applyAuthoredScenarioPreset,
  canUseDisaster,
  createGameSession,
  createTutorialScenario,
  createTutorialSession,
  recordDisaster,
  remainingDisasters,
  SCENARIO_PRESETS,
  storedForcedShocksBeforePlayer,
  TUTORIAL_SEED,
  TUTORIAL_SHOCK,
  type GameSessionState,
  type SessionSelection,
} from './session'
import { StartScreen } from './StartScreen'
import { deriveCityOutcome, deriveRunDebrief } from './stakes'
import { TutorialGuide, tutorialLessonFor } from './TutorialGuide'
import { useCityAudio } from './useCityAudio'
import { CITY_CAMERA_LAYOUT, CITY_PLATE, CITY_PLATE_CAMERA_BOUNDS } from './worldLayout'
import type { VehicleDockDwell } from './VehicleFleet'
import {
  activeSiteCount,
  disasterWind,
  depotStatusesForDay,
  incidentRecoveryProgress,
  incidentPhaseForDay,
  intensityBand,
  LEGACY_V1_DEPOT_DISCLOSURE,
  latestIncident,
  latestIncidentOfType,
  recoveryArcForService,
  strongestShockService,
} from './realism'
import {
  advanceAdaptiveQuality,
  applyQualityCeiling,
  createAdaptiveQualityState,
  renderQualityProfile,
  viewportQualityPolicy,
  type ViewportMetrics,
} from './renderQuality'
import './game.css'

type CityGameProps = {
  initialResult?: CompareResponse | null
  onOpenToolbox: () => void
  onResult: (result: CompareResponse) => void
}

function impactEventKey(seed: number, day: number, type: ShockType, severity: number): string {
  return `${seed}:${day}:${type}:${severity.toFixed(6)}`
}

function deterministicImpactId(seed: number, day: number, type: ShockType, severity: number): number {
  const value = impactEventKey(seed, day, type, severity)
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

function impactEventForResultDay(
  result: CompareResponse,
  dayIndex: number,
  retainedTargets: ReadonlyMap<string, Pick<CityImpactEvent, 'id' | 'point' | 'service'>>,
): CityImpactEvent | null {
  const day = result.candidate.trajectory[dayIndex]
  if (!day?.shock.type) return null
  const type = day.shock.type as ShockType
  const key = impactEventKey(result.seed, day.day, type, day.shock.severity)
  const retainedTarget = retainedTargets.get(key)
  const strongest = strongestShockService(day.shock, result.services)
  const district = DISTRICTS.find((entry) => entry.service === strongest) ?? DISTRICTS[0]
  return {
    id: retainedTarget?.id ?? deterministicImpactId(result.seed, day.day, type, day.shock.severity),
    type,
    severity: day.shock.severity,
    day: day.day,
    point: retainedTarget?.point ?? [
      district.center[0],
      CITY_PLATE.studPositionY + CITY_PLATE.studHeight / 2 + 0.02,
      district.center[2],
    ],
    service: retainedTarget?.service ?? strongest,
    impact: [...day.shock.impact],
    wind: type === 'weather' ? disasterWind(result, day.day) : undefined,
  }
}

type GamePhase = 'setup' | 'loading' | 'playing' | 'collapse' | 'debrief'

const INCIDENT_PHASE_COPY = {
  TELEGRAPH: 'Arrival staged for the next day boundary',
  IMPACT: 'Recorded footprint is crossing Relay City',
  ASSESSMENT: 'Damage assessment leads the response',
  RESPONSE: 'Emergency wave is handing off to recovery logistics',
  RECOVERY: 'Crews remain on the recorded multi-day recovery arc',
  CLEAR: 'No active incident sequence',
} as const

type SceneBoundaryProps = { children: ReactNode; onOpenToolbox: () => void }
type SceneBoundaryState = { failed: boolean }

class SceneBoundary extends Component<SceneBoundaryProps, SceneBoundaryState> {
  state: SceneBoundaryState = { failed: false }

  static getDerivedStateFromError(): SceneBoundaryState {
    return { failed: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('The city view could not initialize.', error, info)
  }

  render() {
    if (this.state.failed) return <WebGLFallback onOpenToolbox={this.props.onOpenToolbox} />
    return this.props.children
  }
}

function browserHasWebGL(): boolean {
  if (typeof document === 'undefined') return false
  try {
    const canvas = document.createElement('canvas')
    return Boolean(canvas.getContext('webgl2') ?? canvas.getContext('webgl'))
  } catch {
    return false
  }
}

function browserViewportMetrics(): ViewportMetrics {
  if (typeof window === 'undefined') {
    return { width: 1440, height: 900, devicePixelRatio: 1 }
  }
  return {
    width: window.innerWidth || 1440,
    height: window.innerHeight || 900,
    devicePixelRatio: window.devicePixelRatio || 1,
  }
}

function WebGLFallback({ onOpenToolbox }: { onOpenToolbox: () => void }) {
  return (
    <div className="webgl-fallback" role="alert">
      <div className="fallback-orb" aria-hidden="true" />
      <p>3D Relay City view unavailable</p>
      <h2>This browser could not start WebGL.</h2>
      <span>The complete Relay City simulation remains available in the Analyst Toolbox.</span>
      <button type="button" onClick={onOpenToolbox}>Open Analyst Toolbox</button>
    </div>
  )
}

function LoadingCity() {
  return (
    <div className="game-loading" role="status">
      <div className="loading-plate" aria-hidden="true"><i /><i /><i /><i /><i /><i /></div>
      <p>Building Relay City&rsquo;s shared shock trajectory</p>
      <span>RELAY is resolving the full deterministic Relay City trajectory under shared conditions.</span>
    </div>
  )
}

function usePresentationClockSnapshot(clock: PresentationClockStore) {
  return useSyncExternalStore(clock.subscribe, clock.getSnapshot, clock.getSnapshot)
}

function PresentedCityScene({ presentationClock, result, ...props }: {
  presentationClock: PresentationClockStore
  result: CompareResponse
} & Omit<ComponentProps<typeof CityScene>, 'result' | 'presentation'>) {
  const cursor = usePresentationClockSnapshot(presentationClock)
  const presentation = useMemo(
    () => sampleRunPresentation(result, cursor),
    [cursor, result],
  )
  return <CityScene {...props} result={result} presentation={presentation} />
}

function CityPresentationHud({
  result,
  presentationClock,
  darkServices,
}: {
  result: CompareResponse
  presentationClock: PresentationClockStore
  darkServices: readonly Service[]
}) {
  const cursor = usePresentationClockSnapshot(presentationClock)
  const sample = useMemo(() => sampleRunPresentation(result, cursor), [cursor, result])
  const wellbeing = sample.wellbeing * 100
  const shockReduced = sample.recordedDay.available_budget < result.scenario.daily_budget - 0.001
  const serviceReadings = DISTRICTS.map((district) => {
    const index = result.services.indexOf(district.service)
    return { ...district, value: sample.services[index] ?? 0 }
  })
  return (
    <>
      <div
        className="city-hud"
        aria-label={`Day ${sample.dayNumber} Relay City condition`}
        data-presentation-day-index={sample.dayIndex}
        data-presentation-progress={sample.progress.toFixed(6)}
        data-presentation-eased={sample.easedProgress.toFixed(6)}
      >
        <div className="day-readout">
          <span>Day</span><strong>{String(sample.dayNumber).padStart(2, '0')}</strong><small>of {result.scenario.horizon_days}</small>
        </div>
        <div className="wellbeing-readout">
          <span>Relay City condition</span><strong>{wellbeing.toFixed(1)}%</strong>
          <i><b style={{ width: `${wellbeing}%` }} /></i>
        </div>
        <div className="budget-readout">
          <span>Supply today</span>
          <strong>{sample.availableBudget.toFixed(1)} <small>/ {result.scenario.daily_budget.toFixed(0)}</small></strong>
          <em>{shockReduced ? 'shock-adjusted arrival · visual interpolation' : 'full daily arrival · visual interpolation'}</em>
        </div>
        {!sample.recordedDay.logistics ? <p className="legacy-v1-disclosure" role="note">{LEGACY_V1_DEPOT_DISCLOSURE}</p> : null}
      </div>
      <div className="service-strip" aria-label="Service condition by district">
        <small className="service-strip-note">Condition · visual interpolation · exact days in Toolbox</small>
        {serviceReadings.map((service) => (
          <div
            key={service.service}
            className={darkServices.includes(service.service) ? 'is-dark' : ''}
            style={{ '--service-accent': service.accent } as React.CSSProperties}
          >
            <span>{service.shortLabel}</span><strong>{(service.value * 100).toFixed(1)}%</strong>
            <i><b style={{ width: `${service.value * 100}%` }} /></i>
          </div>
        ))}
      </div>
    </>
  )
}

function PresentationServiceValue({
  result,
  presentationClock,
  service,
}: {
  result: CompareResponse
  presentationClock: PresentationClockStore
  service: Service
}) {
  const cursor = usePresentationClockSnapshot(presentationClock)
  const sample = useMemo(() => sampleRunPresentation(result, cursor), [cursor, result])
  const index = result.services.indexOf(service)
  return <>{((sample.services[index] ?? 0) * 100).toFixed(1)}% <small>visual interpolation</small></>
}

function PresentationDepotStockValue({
  result,
  presentationClock,
  service,
  capacity,
}: {
  result: CompareResponse
  presentationClock: PresentationClockStore
  service: Service
  capacity: number
}) {
  const cursor = usePresentationClockSnapshot(presentationClock)
  const sample = useMemo(() => sampleRunPresentation(result, cursor), [cursor, result])
  const index = result.services.indexOf(service)
  const stock = sample.logistics?.depotStock[index] ?? 0
  return <>{stock.toFixed(1)} / {capacity.toFixed(0)} units <small>visual interpolation</small></>
}

export function CityGame({ initialResult, onOpenToolbox, onResult }: CityGameProps) {
  const [phase, setPhase] = useState<GamePhase>(initialResult ? 'playing' : 'setup')
  const [result, setResult] = useState<CompareResponse | null>(initialResult ?? null)
  const [session, setSession] = useState<GameSessionState | null>(() => (
    initialResult ? createGameSession({ mode: 'sandbox', difficulty: 'moderate', preset: null }) : null
  ))
  const [sessionSource, setSessionSource] = useState<'start-screen' | 'toolbox'>(initialResult ? 'toolbox' : 'start-screen')
  const [error, setError] = useState<string | null>(null)
  const [dayIndex, setDayIndex] = useState(0)
  const [paused, setPaused] = useState(false)
  const [speed, setSpeed] = useState<PlaybackSpeed>(1)
  const [severity, setSeverity] = useState(0.26)
  const [aimingType, setAimingType] = useState<ShockType | null>(null)
  const [aimedDistrict, setAimedDistrict] = useState<DistrictDefinition | null>(null)
  const [recomputing, setRecomputing] = useState(false)
  const [kickError, setKickError] = useState<string | null>(null)
  const [pendingImpact, setPendingImpact] = useState<CityImpactEvent | null>(null)
  const [activeImpact, setActiveImpact] = useState<CityImpactEvent | null>(null)
  const [selectedDistrict, setSelectedDistrict] = useState<DistrictDefinition | null>(null)
  const [postImpactPhase, setPostImpactPhase] = useState<'assessment' | 'response' | null>(null)
  const [dockDwell, setDockDwell] = useState<VehicleDockDwell>({ id: 'none', active: false, strength: 0 })
  const [webglAvailable] = useState(browserHasWebGL)
  const [viewport, setViewport] = useState<ViewportMetrics>(browserViewportMetrics)
  const qualityPolicy = useMemo(() => viewportQualityPolicy(viewport), [viewport])
  const [qualityState, setQualityState] = useState(() => createAdaptiveQualityState(
    viewportQualityPolicy(browserViewportMetrics()),
  ))
  const renderQuality = useMemo(
    () => renderQualityProfile(qualityState.tier, viewport),
    [qualityState.tier, viewport],
  )
  const cameraRef = useRef<THREE.Camera | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const raycaster = useRef(new THREE.Raycaster())
  const platePlane = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0))
  const impactSequence = useRef(0)
  const impactTargets = useRef(new Map<string, Pick<CityImpactEvent, 'id' | 'point' | 'service'>>())
  const startController = useRef<AbortController | null>(null)
  const requestGeneration = useRef(0)
  const [presentationClock] = useState(() => createPresentationClock({
    dayCount: Math.max(1, initialResult?.candidate.trajectory.length ?? 1),
    initialPaused: false,
  }))
  const [presentationEvidenceEnabled] = useState(() => (
    typeof window !== 'undefined' && window.__RELAY_PRESENTATION_EVIDENCE__?.enabled === true
  ))
  const evidenceSeekRestore = useRef<{ dayIndex: number; progress: number; paused: boolean } | null>(null)

  const alignIncidentStateToCursor = useCallback((nextDayIndex: number, progress: number) => {
    setPendingImpact(null)
    if (!result) {
      setActiveImpact(null)
      setPostImpactPhase(null)
      return
    }
    const event = impactEventForResultDay(result, nextDayIndex, impactTargets.current)
    if (!event) {
      setActiveImpact(null)
      setPostImpactPhase(null)
      return
    }
    const stage = presentationIncidentStage(progress)
    const settling = stage === 'response' && progress < DISASTER_STRIKE_SETTLE_END_FRACTION
    setActiveImpact(stage === 'impact' || stage === 'assessment' || settling ? event : null)
    setPostImpactPhase(stage === 'impact' ? null : stage)
  }, [result])

  const seekPresentationForEvidence = useCallback((nextDayIndex: number, progress: number) => {
    if (!evidenceSeekRestore.current) {
      const current = presentationClock.getSnapshot()
      evidenceSeekRestore.current = {
        dayIndex: current.dayIndex,
        progress: current.progress,
        paused: current.paused,
      }
    }
    presentationClock.setPaused(true)
    presentationClock.seek({ dayIndex: nextDayIndex, progress })
    alignIncidentStateToCursor(nextDayIndex, progress)
    setDayIndex(presentationClock.getSnapshot().dayIndex)
    setPaused(true)
  }, [alignIncidentStateToCursor, presentationClock])

  const clearPresentationEvidenceSeek = useCallback(() => {
    const restore = evidenceSeekRestore.current
    if (!restore) return
    evidenceSeekRestore.current = null
    presentationClock.seek({ dayIndex: restore.dayIndex, progress: restore.progress })
    alignIncidentStateToCursor(restore.dayIndex, restore.progress)
    presentationClock.setPaused(restore.paused)
    setDayIndex(presentationClock.getSnapshot().dayIndex)
    setPaused(restore.paused)
  }, [alignIncidentStateToCursor, presentationClock])

  useEffect(() => {
    const updateViewport = () => setViewport(browserViewportMetrics())
    window.addEventListener('resize', updateViewport)
    return () => window.removeEventListener('resize', updateViewport)
  }, [])

  useEffect(() => {
    setQualityState((current) => current.ceiling === qualityPolicy.ceiling
      ? current
      : applyQualityCeiling(current, qualityPolicy.ceiling))
  }, [qualityPolicy.ceiling])

  const recordQualityWindow = useCallback((medianFps: number) => {
    setQualityState((current) => advanceAdaptiveQuality(current, medianFps))
  }, [])

  useEffect(() => {
    return () => {
      requestGeneration.current += 1
      startController.current?.abort()
      presentationClock.destroy()
    }
  }, [presentationClock])

  const clearTransientState = () => {
    presentationClock.seek({ dayIndex: 0, progress: 0 })
    setDayIndex(0)
    setPaused(false)
    setSpeed(1)
    setSeverity(0.26)
    setAimingType(null)
    setAimedDistrict(null)
    setRecomputing(false)
    setKickError(null)
    setPendingImpact(null)
    setActiveImpact(null)
    setSelectedDistrict(null)
    setPostImpactPhase(null)
    impactTargets.current.clear()
  }

  const startPresetSession = async (selection: SessionSelection) => {
    startController.current?.abort()
    const controller = new AbortController()
    startController.current = controller
    const generation = requestGeneration.current + 1
    requestGeneration.current = generation
    const presetId = selection.preset ?? 'fault-line'
    const nextSession = createGameSession({ ...selection, preset: presetId })
    const scenario = applyAuthoredScenarioPreset(defaultScenario, selection.difficulty, presetId)
    clearTransientState()
    setSession(nextSession)
    setSessionSource('start-screen')
    setResult(null)
    setError(null)
    setPhase('loading')
    try {
      const response = await runComparison(defaultSeed, scenario, controller.signal)
      if (generation !== requestGeneration.current) return
      setResult(response)
      onResult(response)
      setPhase('playing')
    } catch (caught) {
      if (controller.signal.aborted || generation !== requestGeneration.current) return
      setError(caught instanceof ComparisonError ? caught.message : 'The deterministic comparison could not be completed.')
    }
  }

  const startTutorialSession = async () => {
    startController.current?.abort()
    const controller = new AbortController()
    startController.current = controller
    const generation = requestGeneration.current + 1
    requestGeneration.current = generation
    const scenario = createTutorialScenario(defaultScenario)
    clearTransientState()
    setSession(createTutorialSession())
    setSessionSource('start-screen')
    setResult(null)
    setError(null)
    setPhase('loading')
    try {
      const response = await runComparison(TUTORIAL_SEED, scenario, controller.signal)
      if (generation !== requestGeneration.current) return
      const actualShock = response.shock_schedule[TUTORIAL_SHOCK.day - 1]
      if (
        !actualShock
        || actualShock.day !== TUTORIAL_SHOCK.day
        || actualShock.type !== TUTORIAL_SHOCK.type
        || !actualShock.forced
        || Math.abs(actualShock.severity - TUTORIAL_SHOCK.severity) > 1e-9
      ) {
        throw new ComparisonError('TUTORIAL_SCHEDULE_MISMATCH', 'The returned tutorial trajectory did not contain its authored Weather incident.')
      }
      const service = strongestShockService(actualShock, response.services)
      const district = DISTRICTS.find((entry) => entry.service === service) ?? DISTRICTS[0]
      setResult(response)
      onResult(response)
      const tutorialImpact: CityImpactEvent = {
        id: ++impactSequence.current,
        type: TUTORIAL_SHOCK.type,
        severity: actualShock.severity,
        day: actualShock.day,
        point: [district.center[0], CITY_PLATE.studPositionY + CITY_PLATE.studHeight / 2 + 0.02, district.center[2]],
        service,
        impact: actualShock.impact,
        wind: disasterWind(response, actualShock.day),
      }
      impactTargets.current.set(
        impactEventKey(response.seed, actualShock.day, TUTORIAL_SHOCK.type, actualShock.severity),
        { id: tutorialImpact.id, point: tutorialImpact.point, service: tutorialImpact.service },
      )
      setPendingImpact(tutorialImpact)
      setPhase('playing')
    } catch (caught) {
      if (controller.signal.aborted || generation !== requestGeneration.current) return
      setError(caught instanceof ComparisonError ? caught.message : 'The deterministic tutorial could not be completed.')
    }
  }

  const returnToSetup = () => {
    requestGeneration.current += 1
    startController.current?.abort()
    clearTransientState()
    setSession(null)
    setSessionSource('start-screen')
    setResult(null)
    setError(null)
    setPhase('setup')
  }

  const candidateOutcome = useMemo(
    () => result ? deriveCityOutcome(result.candidate.trajectory, result.services) : null,
    [result],
  )
  const debrief = useMemo(() => {
    if (!result) return null
    if (!session) return deriveRunDebrief(result)
    if (sessionSource === 'toolbox') {
      return deriveRunDebrief(result, {
        authoredShocks: storedForcedShocksBeforePlayer(result.scenario, session.playerShocks),
        playerShocks: session.playerShocks,
      })
    }
    const authoredShocks = session.tutorial
      ? [TUTORIAL_SHOCK]
      : session.scenarioPreset
        ? SCENARIO_PRESETS[session.scenarioPreset].forcedShocks
        : []
    const authoredLabel = session.tutorial
      ? 'Guided tutorial'
      : session.scenarioPreset
        ? `Authored preset · ${SCENARIO_PRESETS[session.scenarioPreset].label}`
        : 'Authored scenario'
    return deriveRunDebrief(result, {
      authoredShocks,
      authoredLabel,
      playerShocks: session.playerShocks,
    })
  }, [result, session, sessionSource])
  const terminalIndex = result && candidateOutcome
    ? Math.max(0, Math.min(result.candidate.trajectory.length - 1, candidateOutcome.conditions.length - 1))
    : 0

  useEffect(() => {
    presentationClock.replaceRun(terminalIndex + 1)
  }, [presentationClock, result, terminalIndex])

  useEffect(() => {
    presentationClock.setCallbacks({
      onDayChange: ({ toDayIndex }) => {
        alignIncidentStateToCursor(toDayIndex, 0)
        setDayIndex(toDayIndex)
      },
      onTerminal: () => {
        setPhase((current) => current === 'playing'
          ? (candidateOutcome?.fall ? 'collapse' : 'debrief')
          : current)
      },
    })
    return () => presentationClock.setCallbacks({})
  }, [alignIncidentStateToCursor, candidateOutcome, presentationClock])

  useEffect(() => {
    presentationClock.setPaused(paused)
  }, [paused, presentationClock])

  useEffect(() => {
    presentationClock.setSpeed(speed)
  }, [presentationClock, speed])

  useEffect(() => {
    presentationClock.setAiming(Boolean(aimingType))
  }, [aimingType, presentationClock])

  useEffect(() => {
    presentationClock.setBlocked(
      phase !== 'playing' || !result || !candidateOutcome || recomputing,
    )
  }, [candidateOutcome, phase, presentationClock, recomputing, result])

  const day = result?.candidate.trajectory[dayIndex]
  const dayImpactEvent = useMemo<CityImpactEvent | null>(() => (
    result ? impactEventForResultDay(result, dayIndex, impactTargets.current) : null
  ), [dayIndex, result])
  const currentDayHasShock = Boolean(day?.shock.type)
  const incidentPhase = result ? incidentPhaseForDay({
    result,
    dayIndex,
    telegraph: Boolean(pendingImpact && day && pendingImpact.day > day.day),
    impact: currentDayHasShock && postImpactPhase === null,
    postImpactPhase: currentDayHasShock ? postImpactPhase : null,
  }) : 'CLEAR'
  const responseEnabled = incidentPhase === 'CLEAR'
    || incidentPhase === 'RESPONSE'
    || incidentPhase === 'RECOVERY'
  const dayCondition = candidateOutcome?.conditions[dayIndex] ?? null
  const currentActiveSites = useMemo(() => {
    if (!result || !day || !responseEnabled) return 0
    return activeSiteCount(day, result.candidate.trajectory[dayIndex - 1], result.services)
  }, [day, dayIndex, responseEnabled, result])
  const currentDepotStatuses = useMemo(
    () => result && day ? depotStatusesForDay(day, result.services) : [],
    [day, result],
  )
  const weatherIncident = result ? latestIncidentOfType(result, dayIndex, 'weather') : null
  const weatherRecovery = result && weatherIncident
    ? incidentRecoveryProgress(
        result,
        weatherIncident.dayIndex,
        dayIndex,
        strongestShockService(weatherIncident.shock, result.services),
      )
    : 1
  const weatherActivity = weatherIncident && weatherRecovery < 1
    ? weatherIncident.shock.severity * (1 - 0.72 * weatherRecovery)
    : 0
  const audioSnapshot = useMemo(() => {
    if (!result || !day) return null
    return {
      day: day.day,
      shockType: day.shock.type,
      shockSeverity: day.shock.severity,
      narration: relayNarration(result, dayIndex),
      darkServices: dayCondition?.darkServices ?? [],
      cuesEnabled: phase === 'playing' && !paused,
      trafficActivity: day.resilience,
      constructionActivity: responseEnabled ? Math.min(1, currentActiveSites / 18) : 0,
      weatherActivity,
      emergencyWave: {
        id: `${result.seed}:${day.day}:${day.shock.type ?? 'clear'}:emergency`,
        active: incidentPhase === 'ASSESSMENT' && Boolean(day.shock.type),
        strength: day.shock.severity,
      },
      dockDwell: {
        id: dockDwell.id,
        active: phase === 'playing' && !paused && responseEnabled && dockDwell.active,
        strength: dockDwell.strength,
      },
      fallen: phase === 'collapse',
    }
  }, [currentActiveSites, day, dayCondition, dayIndex, dockDwell, incidentPhase, paused, phase, responseEnabled, result, weatherActivity])
  const { muted, toggleMuted } = useCityAudio(audioSnapshot)
  const disasterRemaining = session ? remainingDisasters(session) : null
  const canThrow = Boolean(
    phase === 'playing'
    && session
    && result
    && day
    && dayIndex < terminalIndex
    && day.day < result.scenario.horizon_days
    && !recomputing
    && !pendingImpact
    && !activeImpact
    && canUseDisaster(session),
  )

  const cycleSpeed = () => setSpeed((current) => current === 0.5 ? 1 : current === 1 ? 2 : 0.5)

  const pointOnPlate = (clientX: number, clientY: number): [number, number, number] | null => {
    const camera = cameraRef.current
    const canvas = canvasRef.current
    if (!camera || !canvas) return null
    const rect = canvas.getBoundingClientRect()
    const pointer = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    )
    raycaster.current.setFromCamera(pointer, camera)
    const point = new THREE.Vector3()
    if (!raycaster.current.ray.intersectPlane(platePlane.current, point)) return null
    if (
      Math.abs(point.x) > CITY_PLATE_CAMERA_BOUNDS.halfWidth
      || Math.abs(point.z) > CITY_PLATE_CAMERA_BOUNDS.halfDepth
    ) return null
    return [point.x, CITY_PLATE.studPositionY + CITY_PLATE.studHeight / 2 + 0.02, point.z]
  }

  const handleDragOver = (event: DragEvent<HTMLElement>) => {
    if (!aimingType || !canThrow) return
    event.preventDefault()
    const point = pointOnPlate(event.clientX, event.clientY)
    event.dataTransfer.dropEffect = point ? 'copy' : 'none'
    setAimedDistrict(point ? closestDistrict(point[0], point[2]) : null)
  }

  const applyKick = async (
    type: ShockType,
    district: DistrictDefinition,
    point: [number, number, number],
  ) => {
    if (!result || !day || !session || !canThrow) return
    const targetDay = day.day + 1
    const throwDayIndex = dayIndex
    const nextScenario = appendForcedShock(result.scenario, { day: targetDay, type, severity })
    const generation = requestGeneration.current + 1
    requestGeneration.current = generation
    setAimingType(null)
    setAimedDistrict(null)
    setRecomputing(true)
    setKickError(null)
    try {
      const response = await runComparison(result.seed, nextScenario)
      if (generation !== requestGeneration.current) return
      const actualShock = response.shock_schedule[targetDay - 1]
      if (
        !actualShock
        || actualShock.day !== targetDay
        || actualShock.type !== type
        || !actualShock.forced
        || Math.abs(actualShock.severity - severity) > 1e-9
        || actualShock.impact.length !== response.services.length
      ) {
        throw new ComparisonError('SHOCK_SCHEDULE_MISMATCH', 'The returned trajectory did not contain the appended forced shock.')
      }
      const impact: CityImpactEvent = {
        id: ++impactSequence.current,
        type: actualShock.type,
        severity: actualShock.severity,
        day: targetDay,
        point,
        service: district.service,
        impact: actualShock.impact,
        wind: type === 'weather' ? disasterWind(response, targetDay) : undefined,
      }
      impactTargets.current.set(
        impactEventKey(response.seed, actualShock.day, actualShock.type as ShockType, actualShock.severity),
        { id: impact.id, point: impact.point, service: impact.service },
      )
      setResult(response)
      onResult(response)
      setSession((current) => current ? recordDisaster(current, {
        day: actualShock.day,
        type: actualShock.type as ShockType,
        severity: actualShock.severity,
      }) : current)
      setDayIndex(throwDayIndex)
      setPendingImpact(impact)
      setPaused(false)
    } catch (caught) {
      setKickError(caught instanceof Error ? caught.message : 'The forced shock could not be applied.')
    } finally {
      setRecomputing(false)
    }
  }

  const handleKick = async (event: DragEvent<HTMLElement>) => {
    event.preventDefault()
    if (!canThrow) return
    const encodedType = event.dataTransfer.getData('application/x-civic-shock')
    const type = shockTypes.includes(encodedType as ShockType) ? encodedType as ShockType : aimingType
    const point = pointOnPlate(event.clientX, event.clientY)
    if (!type || !point) return
    await applyKick(type, closestDistrict(point[0], point[2]), point)
  }

  const confirmDistrictKick = () => {
    if (!aimingType || !aimedDistrict || !canThrow) return
    const [x, , z] = aimedDistrict.center
    void applyKick(
      aimingType,
      aimedDistrict,
      [x, CITY_PLATE.studPositionY + CITY_PLATE.studHeight / 2 + 0.02, z],
    )
  }

  useEffect(() => {
    if (!pendingImpact || !day || day.day < pendingImpact.day) return
    setPendingImpact(null)
  }, [day, pendingImpact])

  useEffect(() => {
    const synchronize = () => {
      const cursor = presentationClock.getSnapshot()
      if (cursor.dayIndex !== dayIndex) return
      if (!dayImpactEvent) {
        setActiveImpact((current) => current === null ? current : null)
        setPostImpactPhase((current) => current === null ? current : null)
        return
      }
      const stage = presentationIncidentStage(cursor.progress)
      if (stage === 'impact') {
        setActiveImpact((current) => current?.id === dayImpactEvent.id ? current : dayImpactEvent)
        setPostImpactPhase((current) => current === null ? current : null)
        return
      }
      const next = stage === 'assessment' ? 'assessment' : 'response'
      // Keep the deterministic strike tree mounted through ASSESSMENT so its
      // rain, haze, crumble, and footprint can then dissolve from the shared
      // cursor during early RESPONSE instead of cutting off at its boundary.
      const settling = stage === 'response'
        && cursor.progress < DISASTER_STRIKE_SETTLE_END_FRACTION
      setActiveImpact((current) => stage === 'assessment' || settling
        ? (current?.id === dayImpactEvent.id ? current : dayImpactEvent)
        : (current === null ? current : null))
      setPostImpactPhase((current) => current === next ? current : next)
    }
    synchronize()
    return presentationClock.subscribe(synchronize)
  }, [dayImpactEvent, dayIndex, presentationClock])

  const completeImpact = () => {
    const cursor = presentationClock.getSnapshot()
    if (cursor.dayIndex !== dayIndex) return
    const stage = presentationIncidentStage(cursor.progress)
    if (stage === 'impact') return
    if (stage === 'response' && cursor.progress >= DISASTER_STRIKE_SETTLE_END_FRACTION) {
      setActiveImpact(null)
    }
    setPostImpactPhase(stage === 'assessment' ? 'assessment' : 'response')
  }

  const aimLabel = pendingImpact
      ? `${shockDisplayName(pendingImpact.type)} is gathering now and strikes at the day ${pendingImpact.day} boundary.`
      : null

  const tutorialLesson = session?.tutorial && result
    ? tutorialLessonFor(result, dayIndex, incidentPhase)
    : null
  const selectedDepot = selectedDistrict
    ? currentDepotStatuses.find((status) => status.service === selectedDistrict.service) ?? null
    : null
  const selectedArc = result && selectedDistrict
    ? recoveryArcForService(result, dayIndex, selectedDistrict.service)
    : null
  const selectedSites = responseEnabled && result && day && selectedDistrict
    ? rebuildingCohortForDay(
        day,
        result.candidate.trajectory[dayIndex - 1],
        selectedDistrict.service,
      ).length
    : 0
  const bannerEvent = pendingImpact ?? activeImpact
  const recentIncident = result ? latestIncident(result, dayIndex, 4) : null
  const bannerShockType = bannerEvent?.type
    ?? (day?.shock.type as ShockType | null | undefined)
    ?? recentIncident?.type
  const bannerSeverity = bannerEvent?.severity
    ?? (day?.shock.type ? day.shock.severity : recentIncident?.shock.severity)
    ?? 0

  if (phase === 'setup') {
    return (
      <StartScreen
        onStart={(selection) => void startPresetSession(selection)}
        onStartTutorial={() => void startTutorialSession()}
        onOpenToolbox={onOpenToolbox}
      />
    )
  }

  return (
    <main className="game-shell">
      <header
        className="game-rail"
        inert={phase === 'collapse' || phase === 'debrief' ? true : undefined}
        aria-hidden={phase === 'collapse' || phase === 'debrief' ? true : undefined}
      >
        <div className="game-brand">
          <span className="relay-mark" aria-hidden="true"><i /><i /><i /></span>
          <div><b>RELAY</b><small>Relay City autonomous recovery</small></div>
        </div>
        <div className="game-rail-actions">
          <span className="local-chip" title="Simulation auto-execution never actuates real infrastructure">
            <i />Simulation-only auto-execute
          </span>
          <button
            className={`sound-toggle ${muted ? 'is-muted' : ''}`}
            type="button"
            aria-label={muted ? 'Unmute Relay City sound' : 'Mute Relay City sound'}
            aria-pressed={muted}
            onClick={toggleMuted}
          >
            {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
            <span>Sound {muted ? 'off' : 'on'}</span>
          </button>
          <button className="toolbox-switch" type="button" onClick={onOpenToolbox}>
            <BarChart3 size={16} />Analyst Toolbox
          </button>
        </div>
      </header>

      <section
        className={`city-stage ${aimingType ? 'aiming-disaster' : ''} ${dayCondition?.stumble ? 'is-stumbling' : ''}`}
        data-quality-tier={renderQuality.tier}
        data-quality-dpr={renderQuality.dpr}
        aria-label="Interactive Relay City recovery diorama"
        inert={phase === 'collapse' || phase === 'debrief' ? true : undefined}
        aria-hidden={phase === 'collapse' || phase === 'debrief' ? true : undefined}
        onDragOver={handleDragOver}
        onDrop={(event) => void handleKick(event)}
      >
        {!result && !error ? <LoadingCity /> : null}
        {error ? (
          <div className="game-error" role="alert">
            <p>Comparison blocked</p><h2>{error}</h2>
            <div className="game-error-actions">
              <button type="button" onClick={returnToSetup}>Return to run setup</button>
              <button type="button" onClick={onOpenToolbox}>Inspect in Analyst Toolbox</button>
            </div>
          </div>
        ) : null}
        {result && day ? (
          <>
            {presentationEvidenceEnabled ? (
              <PresentationEvidenceBridge
                result={result}
                clock={presentationClock}
                incidentPhase={incidentPhase}
                pendingShock={pendingImpact ? { type: pendingImpact.type, day: pendingImpact.day } : null}
                onSeek={seekPresentationForEvidence}
                onClearSeek={clearPresentationEvidenceSeek}
              />
            ) : null}
            {webglAvailable ? (
              <SceneBoundary onOpenToolbox={onOpenToolbox}>
                <Canvas
                  className="city-canvas"
                  shadows={renderQuality.shadowsEnabled ? 'basic' : false}
                  dpr={renderQuality.dpr}
                  camera={{
                    position: [...CITY_CAMERA_LAYOUT.defaultPosition],
                    fov: 36,
                    near: 0.1,
                    far: CITY_CAMERA_LAYOUT.fogFar + CITY_CAMERA_LAYOUT.farClearance,
                  }}
                  gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
                  fallback={<WebGLFallback onOpenToolbox={onOpenToolbox} />}
                  onCreated={({ camera, gl }) => {
                    cameraRef.current = camera
                    canvasRef.current = gl.domElement
                  }}
                >
                  <QualityMonitor onWindow={recordQualityWindow} />
                  <PresentedCityScene
                      presentationClock={presentationClock}
                      result={result}
                      dayIndex={dayIndex}
                      aimedService={aimedDistrict?.service ?? null}
                      pendingImpact={pendingImpact}
                      activeImpact={activeImpact}
                      selectedService={selectedDistrict?.service ?? null}
                      onSelectService={(service) => {
                        const district = DISTRICTS.find((entry) => entry.service === service) ?? null
                        if (aimingType) setAimedDistrict(district)
                        else setSelectedDistrict((current) => current?.service === service ? null : district)
                      }}
                      onImpactComplete={completeImpact}
                      darkServices={dayCondition?.darkServices ?? []}
                      previousDarkServices={candidateOutcome?.conditions[dayIndex - 1]?.darkServices ?? []}
                      stumble={dayCondition?.stumble ?? false}
                      previousStumble={candidateOutcome?.conditions[dayIndex - 1]?.stumble ?? false}
                      fallen={Boolean(dayCondition?.fall)}
                      responseEnabled={responseEnabled}
                      incidentPhase={incidentPhase}
                      vehicleMode={incidentPhase === 'ASSESSMENT' ? 'assessment' : 'full'}
                      onDockDwellChange={setDockDwell}
                      narrationOverride={tutorialLesson ? `GUIDE / ${tutorialLesson.phase} — ${tutorialLesson.heading}` : undefined}
                      quality={renderQuality}
                  />
                </Canvas>
              </SceneBoundary>
            ) : <WebGLFallback onOpenToolbox={onOpenToolbox} />}

            <CityPresentationHud
              result={result}
              presentationClock={presentationClock}
              darkServices={dayCondition?.darkServices ?? []}
            />

            {incidentPhase !== 'CLEAR' && bannerShockType ? (
              <div className="incident-phase-banner" role={session?.tutorial ? undefined : 'status'} data-phase={incidentPhase.toLowerCase()}>
                <span>Incident phase</span>
                <b>{incidentPhase}</b>
                <em>{shockDisplayName(bannerShockType)} · {intensityBand(bannerShockType, bannerSeverity)} · raw {bannerSeverity.toFixed(2)}</em>
                <small>{INCIDENT_PHASE_COPY[incidentPhase]}</small>
              </div>
            ) : null}

            {session?.tutorial ? (
              <TutorialGuide result={result} dayIndex={dayIndex} incidentPhase={incidentPhase} />
            ) : null}

            {selectedDistrict && selectedDepot && selectedArc ? (
              <aside className="district-inspector" aria-label={`${selectedDistrict.shortLabel} district inspector`}>
                <header>
                  <div><span>District inspector</span><b>{selectedDistrict.label}</b></div>
                  <button type="button" onClick={() => setSelectedDistrict(null)} aria-label="Close district inspector"><X size={15} /></button>
                </header>
                <div className="district-inspector-grid">
                  <div><span>Service state</span><strong><PresentationServiceValue result={result} presentationClock={presentationClock} service={selectedDepot.service} /></strong></div>
                  {responseEnabled ? <div><span>Today's allocation</span><strong>{selectedDepot.allocationUnits.toFixed(1)} <small>units</small></strong></div> : null}
                  {responseEnabled ? <div><span>Active sites</span><strong>{selectedSites}</strong></div> : null}
                </div>
                {responseEnabled ? (
                  <>
                    <dl>
                      <div><dt>Point of distribution</dt><dd>{selectedDepot.recorded ? `${selectedDepot.placard} · ${selectedDepot.damage}` : 'Condition unavailable'}</dd></div>
                      {selectedDepot.recorded && selectedDepot.stockCapacity !== null ? <div><dt>Depot stock</dt><dd><PresentationDepotStockValue result={result} presentationClock={presentationClock} service={selectedDepot.service} capacity={selectedDepot.stockCapacity} /></dd></div> : null}
                      {selectedDepot.recorded ? <div><dt>Landed today</dt><dd>{selectedDepot.recorded.landedUnits.toFixed(1)} units</dd></div> : null}
                      {selectedDepot.recorded ? <div><dt>Repair supply</dt><dd>{selectedDepot.recorded.repairSupplyUnits.toFixed(1)} effective units</dd></div> : null}
                      {selectedDepot.recorded && selectedDepot.throughputSignal !== null ? <div><dt>Effective throughput</dt><dd>{Math.round(selectedDepot.throughputSignal * 100)}%</dd></div> : null}
                      {selectedDepot.recorded && selectedDepot.recorded.damagePenalty > 1e-7 ? <div><dt>Depot damage</dt><dd>{selectedDepot.recorded.damagePenalty.toFixed(2)} · {selectedDepot.recorded.damageDaysRemaining} days left</dd></div> : null}
                      <div><dt>Inbound window</dt><dd>{selectedDepot.recorded ? selectedDepot.inboundWindow : 'Unavailable in legacy result'}</dd></div>
                      <div><dt>{selectedDepot.recorded ? 'Quantity-derived dock flow' : 'Allocation-backed dispatch presentation'}</dt><dd>{selectedDepot.dispatchedVehicles} line-haul / {selectedDepot.lastMileVehicles} last-mile load equivalents</dd></div>
                      {selectedDepot.recorded && selectedDepot.dockQueue !== null ? <div><dt>Constrained dock queue</dt><dd>{selectedDepot.dockQueue} loads · {(selectedDepot.dockQueueUnits ?? 0).toFixed(1)} held units</dd></div> : null}
                      <div><dt>Recovery arc</dt><dd>{selectedArc.stage.replace('-', ' ')} · {Math.round(selectedArc.progress * 100)}%</dd></div>
                      {selectedDepot.reroutedFrom ? <div><dt>Distribution route</dt><dd>Nearest-healthy presentation route from {DISTRICTS.find((entry) => entry.service === selectedDepot.reroutedFrom)?.shortLabel}; triggered by recorded local depot rubble</dd></div> : null}
                      {selectedDepot.mutualAidFrom ? <div><dt>Mutual aid</dt><dd>Recorded transfer from {DISTRICTS.find((entry) => entry.service === selectedDepot.mutualAidFrom)?.shortLabel}</dd></div> : null}
                    </dl>
                    <p>{selectedDepot.source}</p>
                  </>
                ) : (
                  <p>Damage assessment in progress. End-of-day allocation, depot ledger, dispatch, and repair records appear when RESPONSE begins.</p>
                )}
              </aside>
            ) : null}

            <div className="playback-controls" aria-label="Playback controls">
              <button type="button" onClick={() => setPaused((current) => !current)} aria-label={paused ? 'Resume playback' : 'Pause playback'}>
                {paused ? <Play size={17} fill="currentColor" /> : <Pause size={17} fill="currentColor" />}
              </button>
              <button type="button" onClick={cycleSpeed} aria-label={`Playback speed ${speed} times`}><Gauge size={16} />{speed}×</button>
              <label>
                <span className="sr-only">Simulation day</span>
                <input
                  type="range"
                  min="0"
                  max={terminalIndex}
                  value={dayIndex}
                  onChange={(event) => {
                    const nextDayIndex = Number(event.target.value)
                    presentationClock.seek({ dayIndex: nextDayIndex, progress: 0 })
                    alignIncidentStateToCursor(nextDayIndex, 0)
                    setDayIndex(nextDayIndex)
                    setPaused(true)
                  }}
                />
              </label>
            </div>

            <div className="camera-hint"><Rotate3D size={15} /><span>Drag the plate to orbit · scroll to zoom</span></div>
            {day.shock.type ? (
              <div className="shock-ribbon"><span>Current-day record</span><b>{shockDisplayName(day.shock.type)}</b><em>{intensityBand(day.shock.type as ShockType, day.shock.severity)} · raw {day.shock.severity.toFixed(2)}</em></div>
            ) : null}
            {dayCondition?.stumble ? (
              <div className="critical-ribbon" role="status">
                <span>Critical floor breached</span>
                <b>{dayCondition.belowFloor.map((service) => DISTRICTS.find((district) => district.service === service)?.shortLabel ?? service).join(' · ')}</b>
              </div>
            ) : null}
            {!session?.tutorial ? <DisasterTray
              severity={severity}
              aimingType={aimingType}
              disabled={!canThrow}
              remaining={disasterRemaining}
              mode={session?.mode ?? 'sandbox'}
              aimedDistrict={aimedDistrict}
              targetLabel={aimLabel}
              onSeverity={setSeverity}
              onArm={(type) => {
                if (!canThrow) return
                setAimingType((current) => current === type ? null : type)
                setAimedDistrict(null)
                setKickError(null)
              }}
              onAimStart={(type, event) => {
                if (!canThrow) return
                event.dataTransfer.effectAllowed = 'copy'
                event.dataTransfer.setData('application/x-civic-shock', type)
                setAimingType(type)
                setAimedDistrict(null)
                setKickError(null)
              }}
              onAimEnd={() => { setAimingType(null); setAimedDistrict(null) }}
              onDistrictSelect={(district) => setAimedDistrict(district)}
              onConfirm={confirmDistrictKick}
              onCancel={() => { setAimingType(null); setAimedDistrict(null) }}
            /> : null}
            {aimingType ? <div className="aim-vignette" aria-hidden="true"><span>Choose a district or release over the plate</span></div> : null}
            {recomputing ? <span className="sr-only" role="status">RELAY is resolving the appended shock trajectory.</span> : null}
            {kickError ? <div className="kick-error" role="alert">{kickError}</div> : null}
          </>
        ) : null}
      </section>
      {phase === 'collapse' && candidateOutcome ? (
        <CollapseScreen outcome={candidateOutcome} onDebrief={() => setPhase('debrief')} />
      ) : null}
      {phase === 'debrief' && debrief && session ? (
        <RunDebriefScreen
          debrief={debrief}
          mode={session.mode}
          difficulty={sessionSource === 'toolbox' ? null : session.difficulty}
          runLabel={session.tutorial
            ? 'Guided incident'
            : session.scenarioPreset
              ? SCENARIO_PRESETS[session.scenarioPreset].label
              : undefined}
          playerKicks={session.disastersUsed}
          onOpenToolbox={onOpenToolbox}
          onRestart={returnToSetup}
        />
      ) : null}
    </main>
  )
}
