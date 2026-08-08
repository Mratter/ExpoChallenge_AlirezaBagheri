import { useEffect, useRef, useState } from 'react'
import { BarChart3, Check, Cpu, Play, RotateCcw, SlidersHorizontal } from 'lucide-react'
import { evaluateToolbox, ToolboxEvaluationError } from '../api'
import { humanizeToken } from '../format'
import type {
  MeasuredWorkbenchBenchmark,
  ToolboxEvaluationResponse,
  ToolboxStructuredInput,
} from '../types'

const SERVICES = [
  { id: 'transport', label: 'Transport' },
  { id: 'housing', label: 'Housing' },
  { id: 'food', label: 'Food' },
  { id: 'healthcare', label: 'Healthcare' },
  { id: 'public_services', label: 'Public services' },
] as const

const REGIMES = [
  { value: 0, label: 'Regime A', detail: 'Steady public telemetry' },
  { value: 1, label: 'Regime B', detail: 'Shifting public telemetry' },
  { value: 2, label: 'Regime C', detail: 'Cross-service telemetry' },
  { value: 3, label: 'Regime D', detail: 'Reversal telemetry' },
] as const

export const DEFAULT_TOOLBOX_INPUT: ToolboxStructuredInput = {
  public_forecast_signal: [1.25, 0.1, -0.45, 0.35, -0.2],
  visible_service_need: [0.25, 1.1, 0.2, 0.15, 0.1],
  public_regime: 2,
  current_service_health: [0.68, 0.71, 0.64, 0.73, 0.69],
  phase_window: 1,
}

type ToolboxPreset = {
  id: string
  label: string
  detail: string
  input: ToolboxStructuredInput
}

export const TOOLBOX_PRESETS: ToolboxPreset[] = [
  {
    id: 'cross-service',
    label: 'Cross-service lead',
    detail: 'A strong forecast appears away from the largest visible need.',
    input: DEFAULT_TOOLBOX_INPUT,
  },
  {
    id: 'transport-pressure',
    label: 'Transport pressure',
    detail: 'Low transport health with a concentrated visible need.',
    input: {
      public_forecast_signal: [1.6, 0.15, -0.2, 0.4, -0.1],
      visible_service_need: [1.1, 0.25, 0.35, 0.4, 0.2],
      public_regime: 0,
      current_service_health: [0.48, 0.7, 0.64, 0.62, 0.68],
      phase_window: 3,
    },
  },
  {
    id: 'health-pressure',
    label: 'Healthcare pressure',
    detail: 'A late-window health signal with reduced service health.',
    input: {
      public_forecast_signal: [0.1, -0.2, 0.35, 1.55, 0.25],
      visible_service_need: [0.2, 0.3, 0.35, 1.2, 0.25],
      public_regime: 3,
      current_service_health: [0.72, 0.68, 0.63, 0.46, 0.65],
      phase_window: 7,
    },
  },
  {
    id: 'mixed-pressure',
    label: 'Mixed pressure',
    detail: 'Several public signals compete near the end of the scenario.',
    input: {
      public_forecast_signal: [-0.35, 0.8, 1.25, 0.45, 0.9],
      visible_service_need: [0.35, 0.65, 1.05, 0.4, 0.7],
      public_regime: 1,
      current_service_health: [0.62, 0.55, 0.5, 0.68, 0.58],
      phase_window: 10,
    },
  },
]

type FiveValueField = 'public_forecast_signal' | 'visible_service_need' | 'current_service_health'

function cloneInput(input: ToolboxStructuredInput): ToolboxStructuredInput {
  return {
    public_forecast_signal: [...input.public_forecast_signal],
    visible_service_need: [...input.visible_service_need],
    public_regime: input.public_regime,
    current_service_health: [...input.current_service_health],
    phase_window: input.phase_window,
  }
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function SignalGroup({
  legend,
  description,
  field,
  values,
  minimum,
  maximum,
  step,
  onChange,
}: {
  legend: string
  description: string
  field: FiveValueField
  values: ToolboxStructuredInput[FiveValueField]
  minimum: number
  maximum: number
  step: number
  onChange: (field: FiveValueField, index: number, value: number) => void
}) {
  return (
    <fieldset className="toolbox-group">
      <legend>{legend}</legend>
      <p>{description}</p>
      <div className="toolbox-sliders">
        {SERVICES.map((service, index) => {
          const inputId = `toolbox-${field}-${service.id}`
          return (
            <label className="toolbox-signal" htmlFor={inputId} key={service.id}>
              <span>{service.label}</span>
              <input
                id={inputId}
                type="range"
                min={minimum}
                max={maximum}
                step={step}
                value={values[index]}
                onChange={(event) => onChange(field, index, Number(event.currentTarget.value))}
              />
              <output htmlFor={inputId}>{values[index].toFixed(2)}</output>
            </label>
          )
        })}
      </div>
    </fieldset>
  )
}

function ResultPanel({ result }: { result: ToolboxEvaluationResponse }) {
  return (
    <div className="toolbox-result-content">
      <div className="toolbox-runtime">
        <Cpu size={16} aria-hidden="true" />
        <span>REAL MODEL INFERENCE</span>
        <b>{result.runtime.engine} / {result.runtime.execution_provider}</b>
        <code title={result.runtime.onnx_sha256}>{result.runtime.onnx_sha256.slice(0, 12)}…</code>
      </div>

      <div className="toolbox-decisions" aria-label="Model and heuristic decisions">
        <article className="toolbox-model-decision">
          <span>300K MODEL ACTION</span>
          <strong>{SERVICES[result.model.action_index].label}</strong>
          <small>Action {result.model.action_index + 1} · {percentage(result.model.confidence)} confidence</small>
        </article>
        <article className="toolbox-heuristic-decision">
          <span>STATIC HEURISTIC ACTION</span>
          <strong>{SERVICES[result.heuristic.action_index].label}</strong>
          <small>Largest visible-need signal</small>
        </article>
      </div>

      <p className={`toolbox-agreement ${result.comparison.same_action ? 'same' : 'different'}`}>
        {result.comparison.same_action ? <Check size={15} aria-hidden="true" /> : <BarChart3 size={15} aria-hidden="true" />}
        {result.comparison.same_action
          ? 'Both policies choose the same action for this public state.'
          : 'The learned model and static heuristic choose different actions.'}
      </p>

      <section className="toolbox-scores" aria-labelledby="toolbox-scores-title">
        <div className="toolbox-output-heading">
          <div>
            <span>MODEL OUTPUT LAYER</span>
            <h3 id="toolbox-scores-title">Action probabilities</h3>
          </div>
          <p>Margin {percentage(result.model.probability_margin)}</p>
        </div>
        <div className="toolbox-score-list">
          {SERVICES.map((service, index) => {
            const probability = result.model.probabilities[index]
            const selected = index === result.model.action_index
            return (
              <div className={`toolbox-score-row ${selected ? 'selected' : ''}`} key={service.id}>
                <div><span>{service.label}</span><small>logit {result.model.logits[index].toFixed(3)}</small></div>
                <div className="toolbox-probability" aria-hidden="true"><i style={{ width: percentage(probability) }} /></div>
                <strong>{percentage(probability)}</strong>
              </div>
            )
          })}
        </div>
      </section>

      <details className="toolbox-vector">
        <summary>Inspect the 21 raw features sent to the model</summary>
        <ol>
          {result.input.feature_order.map((feature, index) => (
            <li key={feature}><span>{humanizeToken(feature)}</span><code>{result.input.vector[index].toFixed(4)}</code></li>
          ))}
        </ol>
        <p>Raw public values are normalized inside the sealed ONNX graph.</p>
      </details>
    </div>
  )
}

export function ModelToolbox({ benchmark }: { benchmark: MeasuredWorkbenchBenchmark }) {
  const [input, setInput] = useState<ToolboxStructuredInput>(() => cloneInput(DEFAULT_TOOLBOX_INPUT))
  const [activePreset, setActivePreset] = useState(TOOLBOX_PRESETS[0].id)
  const [result, setResult] = useState<ToolboxEvaluationResponse | null>(null)
  const [status, setStatus] = useState<'idle' | 'running' | 'success' | 'error'>('idle')
  const [error, setError] = useState<ToolboxEvaluationError | null>(null)
  const controller = useRef<AbortController | null>(null)

  useEffect(() => () => controller.current?.abort(), [])

  const updateFiveValue = (field: FiveValueField, index: number, value: number) => {
    setActivePreset('custom')
    setInput((current) => {
      const next = [...current[field]] as ToolboxStructuredInput[FiveValueField]
      next[index] = value
      return { ...current, [field]: next }
    })
  }

  const choosePreset = (preset: ToolboxPreset) => {
    controller.current?.abort()
    setInput(cloneInput(preset.input))
    setActivePreset(preset.id)
    setResult(null)
    setError(null)
    setStatus('idle')
  }

  const reset = () => choosePreset(TOOLBOX_PRESETS[0])

  const runModel = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    controller.current?.abort()
    const requestController = new AbortController()
    controller.current = requestController
    setStatus('running')
    setError(null)
    setResult(null)
    try {
      const response = await evaluateToolbox(input, requestController.signal)
      if (response.benchmark_id !== benchmark.benchmark_id) {
        throw new ToolboxEvaluationError('Model response was bound to a different benchmark.', 200)
      }
      setResult(response)
      setStatus('success')
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      const toolboxError = caught instanceof ToolboxEvaluationError
        ? caught
        : new ToolboxEvaluationError(caught instanceof Error ? caught.message : 'Real model inference failed.', 0)
      setError(toolboxError)
      setStatus('error')
    }
  }

  const phaseRadians = 2 * Math.PI * (input.phase_window - 1) / 12
  const { objective, head_to_head: headToHead, scenario_total: total } = benchmark

  return (
    <section id="toolbox" className="toolbox-section" aria-labelledby="toolbox-title">
      <header className="toolbox-heading">
        <div>
          <span className="index-label">LIVE MODEL TOOLBOX / 21 PUBLIC SIGNALS</span>
          <h2 id="toolbox-title">Change the city signals. Run the real model.</h2>
          <p>Build one public decision state, execute the sealed 300,113-parameter ONNX policy, and compare its action with the fixed visible-need heuristic.</p>
        </div>
        <span className="toolbox-seal"><Cpu size={15} aria-hidden="true" /> ONNX MODEL</span>
      </header>

      <div className="toolbox-benchmark-strip" aria-label="Sealed benchmark summary">
        <div><span>MODEL OBJECTIVE</span><strong>{objective.learned_policy.passes} / {total}</strong></div>
        <div><span>HEURISTIC OBJECTIVE</span><strong>{objective.static_heuristic.passes} / {total}</strong></div>
        <div>
          <span>DIRECT MATCH</span>
          <strong>{headToHead.learned_wins}–{headToHead.heuristic_wins}–{headToHead.ties}</strong>
          <small>model · heuristic · ties</small>
        </div>
        <p>{benchmark.synthetic_disclosure}</p>
      </div>

      <div className="toolbox-layout">
        <form className="toolbox-controls" onSubmit={(event) => void runModel(event)}>
          <div className="toolbox-toolbar">
            <div>
              <SlidersHorizontal size={17} aria-hidden="true" />
              <span>INPUT PATCH BAY</span>
            </div>
            <button className="toolbox-reset" type="button" onClick={reset} disabled={status === 'running'}>
              <RotateCcw size={14} aria-hidden="true" /> Reset
            </button>
          </div>

          <div className="toolbox-presets" role="group" aria-label="Signal presets">
            {TOOLBOX_PRESETS.map((preset) => (
              <button
                type="button"
                key={preset.id}
                aria-pressed={activePreset === preset.id}
                title={preset.detail}
                onClick={() => choosePreset(preset)}
                disabled={status === 'running'}
              >
                {preset.label}
              </button>
            ))}
          </div>

          <SignalGroup
            legend="Public forecast"
            description="Early-warning telemetry by service. Negative values indicate easing pressure; positive values indicate rising pressure."
            field="public_forecast_signal"
            values={input.public_forecast_signal}
            minimum={-8}
            maximum={8}
            step={0.05}
            onChange={updateFiveValue}
          />
          <SignalGroup
            legend="Visible service need"
            description="The current demand signal available to both the model and heuristic."
            field="visible_service_need"
            values={input.visible_service_need}
            minimum={0}
            maximum={1.5}
            step={0.01}
            onChange={updateFiveValue}
          />

          <fieldset className="toolbox-group toolbox-regime-group">
            <legend>Public regime telemetry</legend>
            <p>Select one valid public regime; the workbench expands it into four one-hot model features.</p>
            <div className="toolbox-regimes">
              {REGIMES.map((regime) => (
                <label className={input.public_regime === regime.value ? 'selected' : ''} key={regime.value}>
                  <input
                    type="radio"
                    name="toolbox-public-regime"
                    value={regime.value}
                    checked={input.public_regime === regime.value}
                    onChange={() => {
                      setActivePreset('custom')
                      setInput((current) => ({ ...current, public_regime: regime.value }))
                    }}
                  />
                  <span>{regime.label}</span>
                  <small>{regime.detail}</small>
                </label>
              ))}
            </div>
          </fieldset>

          <SignalGroup
            legend="Current service health"
            description="Normalized public service condition, from unavailable (0.00) to fully healthy (1.00)."
            field="current_service_health"
            values={input.current_service_health}
            minimum={0}
            maximum={1}
            step={0.01}
            onChange={updateFiveValue}
          />

          <fieldset className="toolbox-group toolbox-phase-group">
            <legend>Scenario phase</legend>
            <p>Choose one of the 12 cascade windows; this produces the final sine and cosine inputs.</p>
            <label className="toolbox-phase" htmlFor="toolbox-phase-window">
              <span>Window</span>
              <input
                id="toolbox-phase-window"
                type="range"
                min={1}
                max={12}
                step={1}
                value={input.phase_window}
                onChange={(event) => {
                  const phaseWindow = Number(event.currentTarget.value)
                  setActivePreset('custom')
                  setInput((current) => ({ ...current, phase_window: phaseWindow }))
                }}
              />
              <output htmlFor="toolbox-phase-window">{input.phase_window} / 12</output>
            </label>
            <div className="toolbox-phase-vector" aria-label="Derived phase features">
              <span>phase sin <b>{Math.sin(phaseRadians).toFixed(3)}</b></span>
              <span>phase cos <b>{Math.cos(phaseRadians).toFixed(3)}</b></span>
            </div>
          </fieldset>

          <button className="toolbox-run" type="submit" disabled={status === 'running'}>
            <Play size={16} fill="currentColor" aria-hidden="true" />
            {status === 'running' ? 'Running real ONNX model…' : 'Run real model'}
          </button>
        </form>

        <aside className="toolbox-output" aria-live="polite" aria-busy={status === 'running'}>
          {status === 'idle' && (
            <div className="toolbox-empty">
              <Cpu size={30} aria-hidden="true" />
              <span>AWAITING INPUT</span>
              <h3>The artifact is ready.</h3>
              <p>Adjust the public signals or choose a preset, then run one real ONNX inference.</p>
            </div>
          )}
          {status === 'running' && (
            <div className="toolbox-empty toolbox-running">
              <Cpu size={30} aria-hidden="true" />
              <span>ONNX RUNTIME</span>
              <h3>Evaluating 21 public features…</h3>
              <p>Normalization and the 300k-parameter forward pass execute inside the sealed model.</p>
            </div>
          )}
          {status === 'error' && error && (
            <div className="toolbox-error" role="alert">
              <span>INFERENCE BLOCKED</span>
              <h3>The model did not run.</h3>
              <p>{error.message}</p>
              {error.details.length > 0 && (
                <ul>{error.details.map((detail) => <li key={`${detail.field}-${detail.type}`}>{detail.field}: {detail.message}</li>)}</ul>
              )}
            </div>
          )}
          {status === 'success' && result && <ResultPanel result={result} />}
        </aside>
      </div>
    </section>
  )
}
