import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useEffect, useRef, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { CompareResponse, DayResult, Shock } from '../src/types'

vi.mock('@react-three/fiber', async () => {
  const THREE = await import('three')
  return {
    Canvas: ({ onCreated, children }: {
      onCreated?: (state: { camera: InstanceType<typeof THREE.PerspectiveCamera>; gl: { domElement: HTMLCanvasElement } }) => void
      children?: ReactNode
    }) => {
      const canvas = useRef<HTMLCanvasElement>(null)
      useEffect(() => {
        if (!canvas.current || !onCreated) return
        const camera = new THREE.PerspectiveCamera(36, 1440 / 840, 0.1, 100)
        camera.position.set(19, 18, 23)
        camera.lookAt(0, 0, 0)
        camera.updateProjectionMatrix()
        camera.updateMatrixWorld()
        canvas.current.getBoundingClientRect = () => ({
          x: 0, y: 60, left: 0, top: 60, right: 1440, bottom: 900,
          width: 1440, height: 840, toJSON: () => ({}),
        })
        onCreated({ camera, gl: { domElement: canvas.current } })
      }, [onCreated])
      return <div><canvas ref={canvas} />{children}</div>
    },
  }
})

vi.mock('../src/game/CityScene', () => ({ CityScene: () => null }))

import { CityGame } from '../src/game/CityGame'

const SERVICES = ['transport', 'housing', 'food', 'healthcare', 'public_services'] as const

function trajectoryDay(day: number, shock?: Shock): DayResult {
  return {
    day,
    shock: shock ?? { day, type: null, severity: 0, impact: [0, 0, 0, 0, 0], budget_factor: 0, forced: false },
    available_budget: 180,
    services_before: [0.4, 0.4, 0.4, 0.4, 0.4],
    services_after_shock: [0.4, 0.4, 0.4, 0.4, 0.4],
    raw_action: [0, 0, 0, 0, 0],
    raw_proposal: [36, 36, 36, 36, 36],
    lower_bounds: [0, 0, 0, 0, 0],
    upper_bounds: [90, 90, 90, 90, 90],
    allocation: [36, 36, 36, 36, 36],
    projection: {
      distance: 0, sum: 180, constraint_violations: 0,
      violation_breakdown: { sum_violations: 0, budget_violations: 0, lower_violations: 0, upper_violations: 0, total: 0 },
      bindings: [],
    },
    planner_evidence: null,
    support: [0.7, 0.7, 0.7, 0.7, 0.7],
    gain: [0.01, 0.01, 0.01, 0.01, 0.01],
    strain: [0, 0, 0, 0, 0],
    services_end: [0.4, 0.4, 0.4, 0.4, 0.4],
    resilience: 0.4,
    reward: 0.4,
  }
}

function resultFixture(forced?: Shock): CompareResponse {
  const trajectory = Array.from({ length: 14 }, (_, index) => trajectoryDay(index + 1, index === 1 && forced ? forced : undefined))
  const schedule = trajectory.map((day) => day.shock)
  return {
    schema_version: '2.1.0',
    result_id: 'a'.repeat(64),
    seed: 424242,
    scenario: {
      name: 'Kick fixture', horizon_days: 14, daily_budget: 180,
      initial_services: [0.4, 0.4, 0.4, 0.4, 0.4], priorities: [1, 1, 1, 1, 1],
      shock_probability: 0.2, severity_min: 0.1, severity_max: 0.28,
      forced_shock: { day: 5, type: 'utility', severity: 0.26 }, forced_shocks: forced ? [{ day: 2, type: 'aftershock', severity: 0.26 }] : [],
    },
    services: [...SERVICES],
    shock_schedule: schedule,
    candidate: { trajectory },
    baseline: { trajectory },
  } as CompareResponse
}

function dragTransfer() {
  const values = new Map<string, string>()
  return {
    effectAllowed: 'none', dropEffect: 'none', files: [], items: [], types: [],
    setData: (type: string, value: string) => values.set(type, value),
    getData: (type: string) => values.get(type) ?? '',
    clearData: () => values.clear(),
    setDragImage: () => undefined,
  } as unknown as DataTransfer
}

function dispatchDrag(element: Element, type: string, transfer: DataTransfer, clientX = 720, clientY = 450) {
  const event = new Event(type, { bubbles: true, cancelable: true })
  Object.defineProperties(event, {
    dataTransfer: { value: transfer },
    clientX: { value: clientX },
    clientY: { value: clientY },
  })
  fireEvent(element, event)
}

async function throwAftershock() {
  const transfer = dragTransfer()
  const card = screen.getByRole('button', { name: 'Aftershock' })
  const stage = screen.getByRole('region', { name: 'Interactive recovery city' })
  dispatchDrag(card, 'dragstart', transfer)
  await act(async () => { await Promise.resolve() })
  dispatchDrag(stage, 'dragover', transfer)
  await act(async () => { await Promise.resolve() })
  dispatchDrag(stage, 'drop', transfer)
  dispatchDrag(card, 'dragend', transfer)
  await act(async () => { await Promise.resolve() })
}

describe('city kick flow', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as never)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('freezes the current day, appends a next-boundary shock, then resumes after the rerun', async () => {
    let resolveFetch!: (response: { ok: boolean; json: () => Promise<CompareResponse> }) => void
    let capturedInit: RequestInit | undefined
    const fetchMock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
      capturedInit = init
      return new Promise<{ ok: boolean; json: () => Promise<CompareResponse> }>((resolve) => { resolveFetch = resolve })
    })
    vi.stubGlobal('fetch', fetchMock)
    const returnedShock: Shock = {
      day: 2, type: 'aftershock', severity: 0.26,
      impact: [0.65, 1, 0.2, 0.35, 0.45], budget_factor: 0.15, forced: true,
    }
    const rerun = resultFixture(returnedShock)
    const onResult = vi.fn()
    render(<CityGame initialResult={resultFixture()} onOpenToolbox={vi.fn()} onResult={onResult} />)
    await act(async () => { await Promise.resolve() })

    await throwAftershock()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const request = JSON.parse(capturedInit?.body as string)
    expect(request.scenario.forced_shock).toEqual({ day: 5, type: 'utility', severity: 0.26 })
    expect(request.scenario.forced_shocks).toEqual([{ day: 2, type: 'aftershock', severity: 0.26 }])

    await act(async () => { vi.advanceTimersByTime(8_000) })
    expect(screen.getByLabelText('Day 1 city condition')).toBeVisible()

    await act(async () => {
      resolveFetch({ ok: true, json: async () => rerun })
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(onResult).toHaveBeenCalledWith(rerun)
    expect(screen.getByLabelText('Day 1 city condition')).toBeVisible()

    await act(async () => { vi.advanceTimersByTime(2_000) })
    expect(screen.getByLabelText('Day 2 city condition')).toBeVisible()
    expect(screen.getByText('aftershock')).toBeVisible()
  })

  it('keeps a prominent accessible sound control on by default', async () => {
    render(<CityGame initialResult={resultFixture()} onOpenToolbox={vi.fn()} onResult={vi.fn()} />)
    const mute = screen.getByRole('button', { name: 'Mute city sound' })
    expect(mute).toHaveAttribute('aria-pressed', 'false')
    expect(mute).toHaveTextContent('Sound on')

    fireEvent.click(mute)

    const unmute = screen.getByRole('button', { name: 'Unmute city sound' })
    expect(unmute).toHaveAttribute('aria-pressed', 'true')
    expect(unmute).toHaveTextContent('Sound off')
  })

  it('offers a button-only district target flow and reports the strongest authored footprint', async () => {
    const returnedShock: Shock = {
      day: 2, type: 'utility', severity: 0.26,
      impact: [0.3, 0.35, 0.45, 0.7, 1], budget_factor: 0.3, forced: true,
    }
    let capturedInit: RequestInit | undefined
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      capturedInit = init
      return { ok: true, json: async () => resultFixture(returnedShock) }
    })
    vi.stubGlobal('fetch', fetchMock)
    const onResult = vi.fn()
    render(<CityGame initialResult={resultFixture()} onOpenToolbox={vi.fn()} onResult={onResult} />)

    fireEvent.click(screen.getByRole('button', { name: 'Utility' }))
    expect(screen.getByRole('group', { name: 'Choose a district target' })).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Housing' }))

    expect(screen.getByText('Housing is hit hardest by Aftershock · 100% footprint.')).toBeVisible()
    expect(screen.getByText('Selected Utility footprint · 35%.')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Aftershock' })).toHaveClass('is-hardest')

    fireEvent.click(screen.getByRole('button', { name: 'Strike Housing overnight' }))
    await act(async () => { await Promise.resolve(); await Promise.resolve() })

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const request = JSON.parse(capturedInit?.body as string)
    expect(request.scenario.forced_shocks).toEqual([{ day: 2, type: 'utility', severity: 0.26 }])
    expect(onResult).toHaveBeenCalledTimes(1)
  })

  it('rejects a response whose forced schedule does not match the thrown disaster', async () => {
    const mismatched: Shock = {
      day: 2, type: 'weather', severity: 0.26,
      impact: [0.75, 0.55, 0.5, 0.4, 0.6], budget_factor: 0.25, forced: true,
    }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => resultFixture(mismatched) }))
    const onResult = vi.fn()
    render(<CityGame initialResult={resultFixture()} onOpenToolbox={vi.fn()} onResult={onResult} />)
    await act(async () => { await Promise.resolve() })
    fireEvent.click(screen.getByRole('button', { name: 'Pause playback' }))

    await throwAftershock()
    await act(async () => { await Promise.resolve(); await Promise.resolve() })

    expect(screen.getByRole('alert')).toHaveTextContent('returned trajectory did not contain the appended forced shock')
    expect(onResult).not.toHaveBeenCalled()
    expect(screen.getByLabelText('Day 1 city condition')).toBeVisible()
  })

  it('consumes one Stress Test disaster only after a verified rerun and locks stacking until impact', async () => {
    const returnedShock: Shock = {
      day: 2, type: 'aftershock', severity: 0.26,
      impact: [0.65, 1, 0.2, 0.35, 0.45], budget_factor: 0.15, forced: true,
    }
    let call = 0
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async () => {
      call += 1
      return { ok: true, json: async () => call === 1 ? resultFixture() : resultFixture(returnedShock) }
    }))
    render(<CityGame onOpenToolbox={vi.fn()} onResult={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Start Stress Test' }))
    await act(async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve() })
    expect(screen.getByLabelText('6 disasters remaining')).toBeVisible()

    await throwAftershock()
    await act(async () => { await Promise.resolve(); await Promise.resolve() })
    expect(screen.getByLabelText('5 disasters remaining')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Aftershock' })).toBeDisabled()

    await act(async () => { vi.advanceTimersByTime(2_000) })
    expect(screen.getByLabelText('Day 2 city condition')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Aftershock' })).toBeEnabled()
  })

  it('stops at the earliest city-condition fall, shows the somber screen, then one debrief', async () => {
    const fallen = resultFixture()
    fallen.candidate.trajectory[0].services_end = [0.08, 0.08, 0.5, 0.5, 0.5]
    fallen.candidate.trajectory[0].resilience = 0.2
    fallen.candidate.trajectory[1].services_end = [0.08, 0.08, 0.5, 0.5, 0.5]
    fallen.candidate.trajectory[1].resilience = 0.18
    render(<CityGame initialResult={fallen} onOpenToolbox={vi.fn()} onResult={vi.fn()} />)

    await act(async () => { vi.advanceTimersByTime(2_000) })
    expect(screen.getByLabelText('Day 2 city condition')).toBeVisible()
    await act(async () => { vi.advanceTimersByTime(2_000) })

    expect(screen.getByRole('heading', { name: 'The city fell on day 2.' })).toHaveFocus()
    expect(screen.getByText(/two consecutive days/i)).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Review what happened' }))

    expect(screen.getByRole('heading', { name: 'Fell on day 2' })).toHaveFocus()
    expect(screen.getByRole('heading', { name: 'conventional rule-based planner' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Inspect this run in the Analyst Toolbox' })).toBeVisible()
  })
})
