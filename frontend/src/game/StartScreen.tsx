import { ArrowRight, BarChart3 } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import {
  DIFFICULTY_DETAILS,
  MODE_DETAILS,
  difficulties,
  gameModes,
  type Difficulty,
  type GameMode,
  type SessionSelection,
} from './session'
import './start-screen.css'

export type StartScreenProps = {
  onStart: (selection: SessionSelection) => void
  onOpenToolbox?: () => void
  initialMode?: GameMode
  initialDifficulty?: Difficulty
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
  onOpenToolbox,
  initialMode = 'stress',
  initialDifficulty = 'moderate',
}: StartScreenProps) {
  const [mode, setMode] = useState<GameMode>(initialMode)
  const [difficulty, setDifficulty] = useState<Difficulty>(initialDifficulty)
  const isStressTest = mode === 'stress'

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    onStart({ mode, difficulty })
  }

  return (
    <main className="session-screen">
      <header className="session-rail">
        <a className="session-brand" href="#/game" aria-label="RELAY recovery planner home">
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
            <h1 id="session-heading">Put the city through a recovery run.</h1>
            <span>Choose the run format and the conditions RELAY will face. Once the city is live, you control the disasters and the clock.</span>
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
              ? 'Difficulty changes city conditions, not the six-disaster allowance.'
              : 'Throw as many disasters as you choose. City-condition collapse rules remain active.'}</span>
          </div>
          {isStressTest ? (
            <span className="session-arsenal-rack" role="img" aria-label="6 disasters available">
              {Array.from({ length: 6 }, (_, index) => <i key={index} />)}
            </span>
          ) : (
            <span className="session-infinity" role="img" aria-label="Unlimited disasters">∞</span>
          )}
        </section>

        <fieldset className="session-fieldset session-difficulty-fieldset">
          <legend>City conditions</legend>
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
                      <IntensityMark level={index + 1} />
                    </span>
                    <small>{details.summary}</small>
                  </span>
                </label>
              )
            })}
          </div>
        </fieldset>

        <footer className="session-actions">
          <p><span>Selected</span><b>{MODE_DETAILS[mode].label} · {DIFFICULTY_DETAILS[difficulty].label}</b></p>
          <button className="session-start-button" type="submit">
            Start {MODE_DETAILS[mode].label}
            <ArrowRight size={17} strokeWidth={2} aria-hidden="true" />
          </button>
        </footer>
      </form>
    </main>
  )
}
