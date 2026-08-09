import { Canvas } from '@react-three/fiber'
import { BarChart3, Gauge, Pause, Play, Rotate3D, Volume2, VolumeX, X } from 'lucide-react'
import { Component, useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type ErrorInfo, type ReactNode } from 'react'
import * as THREE from 'three'
import { ComparisonError, runComparison } from '../api'
import { defaultScenario, defaultSeed } from '../scenarios'
import { shockDisplayName } from '../shockPresentation'
import { shockTypes, type CompareResponse, type ShockType } from '../types'
import { CityScene } from './CityScene'
import { DisasterTray } from './DisasterTray'
import { QualityMonitor } from './QualityMonitor'
import { DISTRICTS, appendForcedShock, closestDistrict, rebuildingCohortForDay, relayNarration, type CityImpactEvent, type DistrictDefinition } from './model'
import { gameDayDurationMs, continuousMotionRate, type PlaybackSpeed } from './pacing'
import { RunDebriefScreen } from './RunOutcome'
import { SceneMotionProvider } from './SceneEffects'
import {
  applyDifficultyPreset,
  canUseDisaster,
  createGameSession,
  recordDisaster,
  remainingDisasters,
  type GameSessionState,
  type SessionSelection,
} from './session'
import { StartScreen } from './StartScreen'
import { deriveCityOutcome } from './stakes'
import { useCityAudio } from './useCityAudio'
import { CITY_CAMERA_LAYOUT, CITY_PLATE, CITY_PLATE_CAMERA_BOUNDS } from './worldLayout'
import {
  activeSiteCount,
  disasterWind,
  depotStatusesForDay,
  incidentPhaseForDay,
  intensityBand,
  latestIncident,
  recoveryArcForService,
  strongestShockService,
  throughputVehiclesPerDay,
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

type GamePhase = 'setup' | 'loading' | 'playing' | 'debrief'

const INCIDENT_PHASE_COPY = {
  TELEGRAPH: 'Arrival staged for the next day boundary',
  IMPACT: 'Recorded footprint is crossing the city',
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
      <p>3D city view unavailable</p>
      <h2>This browser could not start WebGL.</h2>
      <span>The complete simulation remains available in the Analyst Toolbox.</span>
      <button type="button" onClick={onOpenToolbox}>Open Analyst Toolbox</button>
    </div>
  )
}

function LoadingCity() {
  return (
    <div className="game-loading" role="status">
      <div className="loading-plate" aria-hidden="true"><i /><i /><i /><i /><i /><i /></div>
      <p>Building the shared shock trajectory</p>
      <span>RELAY is resolving the full deterministic trajectory under shared conditions.</span>
    </div>
  )
}

export function CityGame({ initialResult, onOpenToolbox, onResult }: CityGameProps) {
  const [phase, setPhase] = useState<GamePhase>(initialResult ? 'playing' : 'setup')
  const [result, setResult] = useState<CompareResponse | null>(initialResult ?? null)
  const [session, setSession] = useState<GameSessionState | null>(() => (
    initialResult ? createGameSession({ mode: 'sandbox', difficulty: 'moderate' }) : null
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
  const [shiftChange, setShiftChange] = useState(false)
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
  const presentedShockKey = useRef<string | null>(null)
  const startController = useRef<AbortController | null>(null)
  const requestGeneration = useRef(0)

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
    }
  }, [])

  const clearTransientState = () => {
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
    presentedShockKey.current = null
  }

  const startPresetSession = async (selection: SessionSelection) => {
    startController.current?.abort()
    const controller = new AbortController()
    startController.current = controller
    const generation = requestGeneration.current + 1
    requestGeneration.current = generation
    const nextSession = createGameSession(selection)
    const scenario = applyDifficultyPreset(
      defaultScenario,
      selection.difficulty,
      'start-screen',
    )
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
  const terminalIndex = result ? Math.max(0, result.candidate.trajectory.length - 1) : 0

  useEffect(() => {
    if (phase !== 'playing' || !result || !candidateOutcome || paused || recomputing) return
    const delay = gameDayDurationMs(speed, Boolean(aimingType))
    if (dayIndex >= terminalIndex) {
      const timer = window.setTimeout(() => {
        setPhase('debrief')
      }, delay)
      return () => window.clearTimeout(timer)
    }
    const timer = window.setTimeout(() => {
      setDayIndex((current) => Math.min(current + 1, terminalIndex))
    }, delay)
    return () => window.clearTimeout(timer)
  }, [aimingType, candidateOutcome, dayIndex, paused, phase, recomputing, result, speed, terminalIndex])

  const day = result?.candidate.trajectory[dayIndex]
  const dayCondition = candidateOutcome?.conditions[dayIndex] ?? null
  const currentActiveSites = useMemo(() => {
    if (!result || !day) return 0
    return activeSiteCount(day, result.candidate.trajectory[dayIndex - 1], result.services)
  }, [day, dayIndex, result])
  const currentDepotStatuses = useMemo(
    () => result && day ? depotStatusesForDay(day, result.services) : [],
    [day, result],
  )
  const audioSnapshot = useMemo(() => {
    if (!result || !day) return null
    const dockActivity = throughputVehiclesPerDay(day)
    return {
      day: day.day,
      shockType: day.shock.type,
      shockSeverity: day.shock.severity,
      narration: relayNarration(result, dayIndex),
      darkServices: dayCondition?.darkServices ?? [],
      cuesEnabled: phase === 'playing' && !paused,
      trafficActivity: day.resilience,
      constructionActivity: Math.min(1, currentActiveSites / 18),
      weatherActivity: latestIncident(result, dayIndex, 2)?.type === 'weather'
        ? latestIncident(result, dayIndex, 2)?.shock.severity ?? 0
        : 0,
      emergencyWave: {
        id: `${result.seed}:${day.day}:${day.shock.type ?? 'clear'}:emergency`,
        active: Boolean(activeImpact && day.shock.type),
        strength: day.shock.severity,
      },
      dockDwell: {
        id: `${result.seed}:${day.day}:dock`,
        active: phase === 'playing' && !paused && dockActivity > 0,
        strength: Math.min(1, dockActivity / 24),
      },
      fallen: false,
    }
  }, [activeImpact, currentActiveSites, day, dayCondition, dayIndex, paused, phase, result])
  const { muted, toggleMuted } = useCityAudio(audioSnapshot)
  const shockReduced = result && day ? day.available_budget < result.scenario.daily_budget - 0.001 : false
  const wellbeing = day ? Math.round(day.resilience * 100) : 0
  const serviceReadings = useMemo(() => {
    if (!result || !day) return []
    return DISTRICTS.map((district) => {
      const index = result.services.indexOf(district.service)
      return { ...district, value: day.services_end[index] }
    })
  }, [day, result])
  const disasterRemaining = session ? remainingDisasters(session) : null
  const canThrow = Boolean(
    phase === 'playing'
    && session
    && result
    && day
    && dayIndex < terminalIndex
    && day.day + 1 < result.scenario.horizon_days - result.scenario.assessment_tail_days + 1
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
    const tailStart = result.scenario.horizon_days - result.scenario.assessment_tail_days + 1
    if (targetDay >= tailStart) {
      setKickError(`Assessment days ${tailStart}–${result.scenario.horizon_days} are shock-free by protocol.`)
      return
    }
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
      setResult(response)
      onResult(response)
      setSession((current) => current ? recordDisaster(current) : current)
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
    if (result) {
      presentedShockKey.current = `${result.seed}:${pendingImpact.day}:${pendingImpact.type}:${pendingImpact.severity.toFixed(6)}`
    }
    setActiveImpact(pendingImpact)
    setPendingImpact(null)
  }, [day, pendingImpact, result])

  useEffect(() => {
    setPostImpactPhase(null)
  }, [day?.day])

  useEffect(() => {
    if (phase !== 'playing' || !day) return undefined
    setShiftChange(true)
    const timer = window.setTimeout(() => setShiftChange(false), 650)
    return () => window.clearTimeout(timer)
  }, [day?.day, phase])

  useEffect(() => {
    if (!result || !day || !day.shock.type || pendingImpact || activeImpact) return
    const key = `${result.seed}:${day.day}:${day.shock.type}:${day.shock.severity.toFixed(6)}`
    if (presentedShockKey.current === key) return
    presentedShockKey.current = key
    const shockType = day.shock.type as ShockType
    const service = strongestShockService(day.shock, result.services)
    const district = DISTRICTS.find((entry) => entry.service === service) ?? DISTRICTS[0]
    setActiveImpact({
      id: ++impactSequence.current,
      type: shockType,
      severity: day.shock.severity,
      day: day.day,
      point: [district.center[0], CITY_PLATE.studPositionY + CITY_PLATE.studHeight / 2 + 0.02, district.center[2]],
      service,
      impact: day.shock.impact,
      wind: shockType === 'weather' ? disasterWind(result, day.day) : undefined,
    })
  }, [activeImpact, day, pendingImpact, result])

  useEffect(() => {
    if (!postImpactPhase) return
    if (postImpactPhase === 'assessment') {
      const timer = window.setTimeout(() => setPostImpactPhase('response'), 1_250)
      return () => window.clearTimeout(timer)
    }
    return undefined
  }, [postImpactPhase])

  const completeImpact = () => {
    setActiveImpact(null)
    setPostImpactPhase('assessment')
  }

  const aimLabel = pendingImpact
      ? `${shockDisplayName(pendingImpact.type)} is gathering now and strikes at the day ${pendingImpact.day} boundary.`
      : null

  const worldMotionRate = continuousMotionRate(speed, paused)
  const incidentPhase = result ? incidentPhaseForDay({
    result,
    dayIndex,
    telegraph: Boolean(pendingImpact),
    impact: Boolean(activeImpact),
    postImpactPhase,
  }) : 'CLEAR'
  const selectedDepot = selectedDistrict
    ? currentDepotStatuses.find((status) => status.service === selectedDistrict.service) ?? null
    : null
  const selectedArc = result && selectedDistrict
    ? recoveryArcForService(result, dayIndex, selectedDistrict.service)
    : null
  const selectedSites = result && day && selectedDistrict
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
    return <StartScreen onStart={(selection) => void startPresetSession(selection)} onOpenToolbox={onOpenToolbox} />
  }

  return (
    <main className="game-shell">
      <header
        className="game-rail"
        inert={phase === 'debrief' ? true : undefined}
        aria-hidden={phase === 'debrief' ? true : undefined}
      >
        <div className="game-brand">
          <span className="relay-mark" aria-hidden="true"><i /><i /><i /></span>
          <div><b>RELAY</b><small>The city you can't knock over</small></div>
        </div>
        <div className="game-rail-actions">
          <span className="local-chip"><i />Local simulation</span>
          <button
            className={`sound-toggle ${muted ? 'is-muted' : ''}`}
            type="button"
            aria-label={muted ? 'Unmute city sound' : 'Mute city sound'}
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
        aria-label="Interactive recovery city"
        inert={phase === 'debrief' ? true : undefined}
        aria-hidden={phase === 'debrief' ? true : undefined}
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
                  <SceneMotionProvider motionRate={worldMotionRate}>
                    <CityScene
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
                      paused={paused}
                      playbackSpeed={speed}
                      motionRate={worldMotionRate}
                      onImpactComplete={completeImpact}
                      criticalServices={dayCondition?.belowFloor ?? []}
                      darkServices={dayCondition?.darkServices ?? []}
                      stumble={dayCondition?.stumble ?? false}
                      fallen={false}
                      shiftChange={shiftChange}
                      quality={renderQuality}
                    />
                  </SceneMotionProvider>
                </Canvas>
              </SceneBoundary>
            ) : <WebGLFallback onOpenToolbox={onOpenToolbox} />}

            <div className="city-hud" aria-label={`Day ${day.day} city condition`}>
              <div className="day-readout">
                <span>Day</span><strong>{String(day.day).padStart(2, '0')}</strong><small>of {result.scenario.horizon_days}</small>
              </div>
              <div className="wellbeing-readout">
                <span>City condition</span><strong>{wellbeing}%</strong>
                <i><b style={{ width: `${wellbeing}%` }} /></i>
              </div>
              <div className="budget-readout">
                <span>Budget today</span>
                <strong>{Math.round(day.available_budget)} <small>/ {Math.round(result.scenario.daily_budget)}</small></strong>
                <em>{shockReduced ? 'shock-adjusted arrival' : 'full daily arrival'}</em>
              </div>
            </div>

            {incidentPhase !== 'CLEAR' && bannerShockType ? (
              <div className="incident-phase-banner" role="status" data-phase={incidentPhase.toLowerCase()}>
                <span>Incident phase</span>
                <b>{incidentPhase}</b>
                <em>{shockDisplayName(bannerShockType)} · {intensityBand(bannerShockType, bannerSeverity)} · raw {bannerSeverity.toFixed(2)}</em>
                <small>{INCIDENT_PHASE_COPY[incidentPhase]}</small>
              </div>
            ) : null}

            {selectedDistrict && selectedDepot && selectedArc ? (
              <aside className="district-inspector" aria-label={`${selectedDistrict.shortLabel} district inspector`}>
                <header>
                  <div><span>District inspector</span><b>{selectedDistrict.label}</b></div>
                  <button type="button" onClick={() => setSelectedDistrict(null)} aria-label="Close district inspector"><X size={15} /></button>
                </header>
                <div className="district-inspector-grid">
                  <div><span>Service state</span><strong>{Math.round(selectedDepot.serviceLevel * 100)}%</strong></div>
                  <div><span>Today's allocation</span><strong>{selectedDepot.allocationUnits.toFixed(1)} <small>units</small></strong></div>
                  <div><span>Active sites</span><strong>{selectedSites}</strong></div>
                </div>
                <dl>
                  <div><dt>Point of distribution</dt><dd>{selectedDepot.placard} · {selectedDepot.damage}</dd></div>
                  {selectedDepot.recorded ? <div><dt>Depot stock</dt><dd>{selectedDepot.recorded.stockEndUnits.toFixed(1)} / {selectedDepot.stockCapacity.toFixed(0)} units</dd></div> : null}
                  {selectedDepot.recorded ? <div><dt>Landed today</dt><dd>{selectedDepot.recorded.landedUnits.toFixed(1)} units</dd></div> : null}
                  {selectedDepot.recorded ? <div><dt>Repair supply</dt><dd>{selectedDepot.recorded.repairSupplyUnits.toFixed(1)} effective units</dd></div> : null}
                  {selectedDepot.recorded ? <div><dt>Effective throughput</dt><dd>{Math.round(selectedDepot.throughputSignal * 100)}%</dd></div> : null}
                  {selectedDepot.recorded && selectedDepot.recorded.damagePenalty > 1e-7 ? <div><dt>Depot damage</dt><dd>{selectedDepot.recorded.damagePenalty.toFixed(2)} · {selectedDepot.recorded.damageDaysRemaining} days left</dd></div> : null}
                  <div><dt>{selectedDepot.recorded ? 'Inbound window' : 'Presentation wave'}</dt><dd>{selectedDepot.inboundWindow}</dd></div>
                  <div><dt>Dock flow</dt><dd>{selectedDepot.dispatchedVehicles} line-haul / {selectedDepot.lastMileVehicles} last-mile vehicles</dd></div>
                  {selectedDepot.recorded ? <div><dt>Next-day queue</dt><dd>{selectedDepot.dockQueue} loads · {selectedDepot.recorded.capacityHeldUnits.toFixed(1)} units capacity-held</dd></div> : null}
                  <div><dt>Recovery arc</dt><dd>{selectedArc.stage.replace('-', ' ')} · {Math.round(selectedArc.progress * 100)}%</dd></div>
                  {selectedDepot.reroutedFrom ? <div><dt>Mutual aid</dt><dd>Rerouted from {DISTRICTS.find((entry) => entry.service === selectedDepot.reroutedFrom)?.shortLabel}</dd></div> : null}
                </dl>
                <p>{selectedDepot.source}</p>
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
                  onChange={(event) => { setDayIndex(Number(event.target.value)); setPaused(true) }}
                />
              </label>
            </div>

            <div className="service-strip" aria-label="Service condition by district">
              {serviceReadings.map((service) => (
                <div
                  key={service.service}
                  className={dayCondition?.darkServices.includes(service.service) ? 'is-dark' : ''}
                  style={{ '--service-accent': service.accent } as React.CSSProperties}
                >
                  <span>{service.shortLabel}</span><strong>{Math.round(service.value * 100)}</strong>
                  <i><b style={{ width: `${service.value * 100}%` }} /></i>
                </div>
              ))}
            </div>

            <div className="camera-hint"><Rotate3D size={15} /><span>Drag the plate to orbit · scroll to zoom</span></div>
            {day.shock.type ? (
              <div className="shock-ribbon"><span>Recorded event</span><b>{shockDisplayName(day.shock.type)}</b><em>{intensityBand(day.shock.type as ShockType, day.shock.severity)} · raw {day.shock.severity.toFixed(2)}</em></div>
            ) : null}
            {dayCondition?.stumble ? (
              <div className="critical-ribbon" role="status">
                <span>Critical floor breached</span>
                <b>{dayCondition.belowFloor.map((service) => serviceReadings.find((reading) => reading.service === service)?.shortLabel ?? service).join(' · ')}</b>
              </div>
            ) : null}
            <DisasterTray
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
            />
            {aimingType ? <div className="aim-vignette" aria-hidden="true"><span>Choose a district or release over the plate</span></div> : null}
            {recomputing ? <span className="sr-only" role="status">RELAY is resolving the appended shock trajectory.</span> : null}
            {kickError ? <div className="kick-error" role="alert">{kickError}</div> : null}
          </>
        ) : null}
      </section>
      {phase === 'debrief' && result && session ? (
        <RunDebriefScreen
          result={result}
          mode={session.mode}
          difficulty={sessionSource === 'toolbox' ? null : session.difficulty}
          playerKicks={session.disastersUsed}
          onOpenToolbox={onOpenToolbox}
          onRestart={returnToSetup}
        />
      ) : null}
    </main>
  )
}
