import { ArrowRight, BarChart3, BookOpen, Info } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import {
  DIFFICULTY_DETAILS,
  MODE_DETAILS,
  SCENARIO_PRESETS,
  difficulties,
  gameModes,
  scenarioPresetIds,
  type Difficulty,
  type GameMode,
  type ScenarioPresetId,
  type SessionSelection,
} from './session'
import './start-screen.css'

export type StartScreenProps = {
  onStart: (selection: SessionSelection) => void
  onStartTutorial?: () => void
  onOpenToolbox?: () => void
  initialMode?: GameMode
  initialDifficulty?: Difficulty
  initialPreset?: ScenarioPresetId
}

function RelayGlyph() {
  return (
    <span className="session-relay-glyph" aria-hidden="true">
      <i /><i /><i /><i /><i />
    </span>
  )
}

function IntensityMark({ level }: { level: number }) {
  return (
    <span className="session-intensity" aria-hidden="true">
      {[1, 2, 3].map((mark) => <i key={mark} data-active={mark <= level} />)}
    </span>
  )
}

export function StartScreen({
  onStart,
  onStartTutorial,
  onOpenToolbox,
  initialMode = 'stress',
  initialDifficulty = 'moderate',
  initialPreset = 'fault-line',
}: StartScreenProps) {
  const [mode, setMode] = useState<GameMode>(initialMode)
  const [difficulty, setDifficulty] = useState<Difficulty>(initialDifficulty)
  const [preset, setPreset] = useState<ScenarioPresetId>(initialPreset)
  const isStressTest = mode === 'stress'

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    onStart({ mode, difficulty, preset })
  }

  return (
    <main className="session-screen">
      <header className="session-rail">
        <a className="session-brand" href="#/game" aria-label="Relay City recovery planner home">
          <RelayGlyph />
          <span><b>RELAY</b><small>Autonomous recovery planner</small></span>
        </a>
        {onOpenToolbox && (
          <button className="session-toolbox-link" type="button" onClick={onOpenToolbox}>
            <BarChart3 size={15} strokeWidth={1.8} aria-hidden="true" />
            Analyst Toolbox
          </button>
        )}
      </header>

      <div className="session-tabletop" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /><i /></div>

      <form className="session-docket" onSubmit={handleSubmit} aria-labelledby="session-heading">
        <div className="session-intro">
          <RelayGlyph />
          <div>
            <p>Recovery run · ready</p>
            <h1 id="session-heading">Run Relay City through a recovery scenario.</h1>
            <span>Choose a disclosed authored mix, run format, and operating envelope. Once Relay City is live, you control player disasters and the clock.</span>
          </div>
        </div>

        <fieldset className="session-fieldset session-mode-fieldset">
          <legend>Run format</legend>
          <div className="session-option-grid session-mode-grid">
            {gameModes.map((value) => {
              const details = MODE_DETAILS[value]
              return (
                <label className="session-option" data-selected={mode === value} key={value}>
                  <input
                    type="radio"
                    name="game-mode"
                    value={value}
                    checked={mode === value}
                    onChange={() => setMode(value)}
                  />
                  <span className="session-option-body">
                    <span className="session-option-topline">
                      <b>{details.label}</b>
                      <em>{value === 'stress' ? '6 disasters' : 'Unlimited'}</em>
                    </span>
                    <small>{details.summary}</small>
                  </span>
                </label>
              )
            })}
          </div>
        </fieldset>

        <section className="session-arsenal" aria-live="polite" data-mode={mode}>
          <div>
            <p>Disaster allowance</p>
            <strong>{isStressTest ? 'Six for this run' : 'Unlimited in Sandbox'}</strong>
            <span>{isStressTest
              ? 'Difficulty changes Relay City conditions, not the six-disaster allowance.'
              : 'Throw as many disasters as you choose. Relay City condition-collapse rules remain active.'}</span>
          </div>
          {isStressTest ? (
            <span className="session-arsenal-rack" role="img" aria-label="6 disasters available">
              {Array.from({ length: 6 }, (_, index) => <i key={index} />)}
            </span>
          ) : (
            <span className="session-infinity" role="img" aria-label="Unlimited disasters">∞</span>
          )}
        </section>

        <fieldset className="session-fieldset session-scenario-fieldset">
          <legend>Authored scenario</legend>
          <div className="session-option-grid session-scenario-grid">
            {scenarioPresetIds.map((value) => {
              const details = SCENARIO_PRESETS[value]
              return (
                <label className="session-option session-scenario-option" data-selected={preset === value} key={value}>
                  <input
                    type="radio"
                    name="scenario-preset"
                    value={value}
                    checked={preset === value}
                    onChange={() => setPreset(value)}
                  />
                  <span className="session-option-body">
                    <span className="session-option-topline"><b>{details.label}</b><em>{details.forcedShocks.length} authored</em></span>
                    <small>{details.summary}</small>
                    <span className="session-scenario-mix">{details.disclosedMix}</span>
                  </span>
                </label>
              )
            })}
          </div>
          <p className="session-schedule-note">The authored mix is disclosed now. Events are identified live as they occur; the complete ordered schedule is disclosed in the end-of-run debrief.</p>
        </fieldset>

        <fieldset className="session-fieldset session-difficulty-fieldset">
          <legend>Relay City conditions</legend>
          <div className="session-option-grid session-difficulty-grid">
            {difficulties.map((value, index) => {
              const details = DIFFICULTY_DETAILS[value]
              return (
                <label className="session-option" data-selected={difficulty === value} key={value}>
                  <input
                    type="radio"
                    name="difficulty"
                    value={value}
                    checked={difficulty === value}
                    onChange={() => setDifficulty(value)}
                  />
                  <span className="session-option-body">
                    <span className="session-option-topline">
                      <b>{details.label}</b>
                      <span className="session-option-tools">
                        <button
                          type="button"
                          className="session-parameter-tip"
                          aria-label={`Show ${details.label} parameter changes`}
                          aria-describedby={`difficulty-${value}-parameters`}
                          onClick={(event) => event.preventDefault()}
                        >
                          <Info size={13} aria-hidden="true" />
                          <span id={`difficulty-${value}-parameters`} role="tooltip">{details.parameters}</span>
                        </button>
                        <IntensityMark level={index + 1} />
                      </span>
                    </span>
                    <small>{details.summary}</small>
                  </span>
                </label>
              )
            })}
          </div>
        </fieldset>

        {onStartTutorial ? (
          <section className="session-tutorial-callout" aria-labelledby="tutorial-callout-heading">
            <BookOpen size={20} strokeWidth={1.7} aria-hidden="true" />
            <div>
              <p>First run</p>
              <h2 id="tutorial-callout-heading">Guided eight-day incident</h2>
              <span>RELAY explains TELEGRAPH → IMPACT → ASSESSMENT → RESPONSE → RECOVERY from one returned Weather trajectory. No player disasters.</span>
            </div>
            <button type="button" onClick={onStartTutorial}>Start guided run</button>
          </section>
        ) : null}

        <footer className="session-actions">
          <p><span>Selected</span><b>{SCENARIO_PRESETS[preset].label} · {MODE_DETAILS[mode].label} · {DIFFICULTY_DETAILS[difficulty].label}</b></p>
          <button className="session-start-button" type="submit">
            Start {MODE_DETAILS[mode].label}
            <ArrowRight size={17} strokeWidth={2} aria-hidden="true" />
          </button>
        </footer>
      </form>
    </main>
  )
}
