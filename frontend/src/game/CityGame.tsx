import { Canvas } from '@react-three/fiber'
import { BarChart3, Gauge, Pause, Play, Rotate3D } from 'lucide-react'
import { Component, useEffect, useMemo, useRef, useState, type DragEvent, type ErrorInfo, type ReactNode } from 'react'
import * as THREE from 'three'
import { ComparisonError, runComparison } from '../api'
import { defaultScenario, defaultSeed } from '../scenarios'
import { shockTypes, type CompareResponse, type ShockType } from '../types'
import { CityScene } from './CityScene'
import { DisasterTray } from './DisasterTray'
import { DISTRICTS, appendForcedShock, closestDistrict, shockImpactFor, type CityImpactEvent, type DistrictDefinition } from './model'
import { CollapseScreen, RunDebriefScreen } from './RunOutcome'
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
import { deriveCityOutcome, deriveRunDebrief } from './stakes'
import './game.css'

type CityGameProps = {
  initialResult?: CompareResponse | null
  onOpenToolbox: () => void
  onResult: (result: CompareResponse) => void
}

type GamePhase = 'setup' | 'loading' | 'playing' | 'collapse' | 'debrief'

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
  const [speed, setSpeed] = useState(1)
  const [severity, setSeverity] = useState(0.26)
  const [aimingType, setAimingType] = useState<ShockType | null>(null)
  const [aimedDistrict, setAimedDistrict] = useState<DistrictDefinition | null>(null)
  const [recomputing, setRecomputing] = useState(false)
  const [kickError, setKickError] = useState<string | null>(null)
  const [pendingImpact, setPendingImpact] = useState<CityImpactEvent | null>(null)
  const [activeImpact, setActiveImpact] = useState<CityImpactEvent | null>(null)
  const [webglAvailable] = useState(browserHasWebGL)
  const cameraRef = useRef<THREE.Camera | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const raycaster = useRef(new THREE.Raycaster())
  const platePlane = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0))
  const impactSequence = useRef(0)
  const startController = useRef<AbortController | null>(null)
  const requestGeneration = useRef(0)

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
  const debrief = useMemo(() => result ? deriveRunDebrief(result) : null, [result])
  const terminalIndex = result && candidateOutcome
    ? Math.max(0, Math.min(result.candidate.trajectory.length - 1, candidateOutcome.conditions.length - 1))
    : 0

  useEffect(() => {
    if (phase !== 'playing' || !result || !candidateOutcome || paused || recomputing) return
    const delay = (2000 / speed) * (aimingType ? 4 : 1)
    if (dayIndex >= terminalIndex) {
      const timer = window.setTimeout(() => {
        setPhase(candidateOutcome.fall ? 'collapse' : 'debrief')
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
  const shockTaxed = result && day ? day.available_budget < result.scenario.daily_budget - 0.001 : false
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
    && day.day < result.scenario.horizon_days
    && !recomputing
    && !pendingImpact
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
    if (Math.abs(point.x) > 12.3 || Math.abs(point.z) > 11.3) return null
    return [point.x, 0.24, point.z]
  }

  const handleDragOver = (event: DragEvent<HTMLElement>) => {
    if (!aimingType || !canThrow) return
    event.preventDefault()
    const point = pointOnPlate(event.clientX, event.clientY)
    event.dataTransfer.dropEffect = point ? 'copy' : 'none'
    setAimedDistrict(point ? closestDistrict(point[0], point[2]) : null)
  }

  const handleKick = async (event: DragEvent<HTMLElement>) => {
    event.preventDefault()
    if (!result || !day || !session || !canThrow) return
    const encodedType = event.dataTransfer.getData('application/x-civic-shock')
    const type = shockTypes.includes(encodedType as ShockType) ? encodedType as ShockType : aimingType
    const point = pointOnPlate(event.clientX, event.clientY)
    if (!type || !point) return
    const district = closestDistrict(point[0], point[2])
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

  useEffect(() => {
    if (!pendingImpact || !day || day.day < pendingImpact.day) return
    setActiveImpact(pendingImpact)
    setPendingImpact(null)
  }, [day, pendingImpact])

  useEffect(() => {
    if (!activeImpact) return
    const timer = window.setTimeout(() => setActiveImpact(null), 1900)
    return () => window.clearTimeout(timer)
  }, [activeImpact])

  const aimLabel = aimingType && aimedDistrict
    ? `${aimedDistrict.shortLabel}: ${Math.round(shockImpactFor(aimingType, aimedDistrict.service) * 100)}% typed impact`
    : pendingImpact
      ? `${pendingImpact.type} strikes overnight on day ${pendingImpact.day}.`
      : null

  if (phase === 'setup') {
    return <StartScreen onStart={(selection) => void startPresetSession(selection)} onOpenToolbox={onOpenToolbox} />
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
          <div><b>Civic Relay</b><small>The city you can't knock over</small></div>
        </div>
        <div className="game-rail-actions">
          <span className="local-chip"><i />Local simulation</span>
          <button className="toolbox-switch" type="button" onClick={onOpenToolbox}>
            <BarChart3 size={16} />Analyst Toolbox
          </button>
        </div>
      </header>

      <section
        className={`city-stage ${aimingType ? 'aiming-disaster' : ''} ${dayCondition?.stumble ? 'is-stumbling' : ''}`}
        aria-label="Interactive recovery city"
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
            {webglAvailable ? (
              <SceneBoundary onOpenToolbox={onOpenToolbox}>
                <Canvas
                  className="city-canvas"
                  shadows="basic"
                  dpr={[1, 1.65]}
                  camera={{ position: [19, 18, 23], fov: 36, near: 0.1, far: 100 }}
                  gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
                  fallback={<WebGLFallback onOpenToolbox={onOpenToolbox} />}
                  onCreated={({ camera, gl }) => {
                    cameraRef.current = camera
                    canvasRef.current = gl.domElement
                  }}
                >
                  <CityScene
                    result={result}
                    dayIndex={dayIndex}
                    aimedService={aimedDistrict?.service ?? null}
                    activeImpact={activeImpact}
                    criticalServices={dayCondition?.belowFloor ?? []}
                    darkServices={dayCondition?.darkServices ?? []}
                    stumble={dayCondition?.stumble ?? false}
                  />
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
                <em>{shockTaxed ? 'shock tax' : 'full daily arrival'}</em>
              </div>
            </div>

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
              <div className="shock-ribbon"><span>Overnight event</span><b>{day.shock.type}</b><em>{day.shock.severity.toFixed(2)} severity</em></div>
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
              targetLabel={aimLabel}
              onSeverity={setSeverity}
              onAimStart={(type, event) => {
                if (!canThrow) return
                event.dataTransfer.effectAllowed = 'copy'
                event.dataTransfer.setData('application/x-civic-shock', type)
                setAimingType(type)
                setAimedDistrict(null)
                setKickError(null)
              }}
              onAimEnd={() => { setAimingType(null); setAimedDistrict(null) }}
            />
            {aimingType ? <div className="aim-vignette" aria-hidden="true"><span>Release over the plate</span></div> : null}
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
          playerKicks={session.disastersUsed}
          onOpenToolbox={onOpenToolbox}
          onRestart={returnToSetup}
        />
      ) : null}
    </main>
  )
}
