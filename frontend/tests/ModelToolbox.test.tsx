import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_TOOLBOX_INPUT, ModelToolbox } from '../src/components/ModelToolbox'
import { toolboxBenchmarkFixture, toolboxResponseFixture } from './toolboxFixtures'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('real model toolbox', () => {
  it('presents the sealed benchmark and runs an edited state through the live endpoint', async () => {
    const response = toolboxResponseFixture()
    response.input.structured.visible_service_need[0] = 1.25
    response.input.vector[5] = 1.25
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => response })
    vi.stubGlobal('fetch', fetchMock)
    render(<ModelToolbox benchmark={toolboxBenchmarkFixture()} />)

    const summary = screen.getByLabelText('Sealed benchmark summary')
    expect(summary).toHaveTextContent(/MODEL OBJECTIVE\s*38 \/ 40/)
    expect(summary).toHaveTextContent(/HEURISTIC OBJECTIVE\s*20 \/ 40/)
    expect(summary).toHaveTextContent(/DIRECT MATCH\s*38–0–2/)

    const transportNeed = document.getElementById('toolbox-visible_service_need-transport')
    expect(transportNeed).toBeInstanceOf(HTMLInputElement)
    fireEvent.change(transportNeed as HTMLInputElement, { target: { value: '1.25' } })
    fireEvent.click(screen.getByRole('button', { name: 'Run real model' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    const request = JSON.parse(String(fetchMock.mock.calls[0][1]?.body)) as Record<string, unknown>
    expect(request).toMatchObject({
      visible_service_need: [1.25, 1.1, 0.2, 0.15, 0.1],
      public_regime: 2,
      phase_window: 1,
    })

    const decisions = await screen.findByLabelText('Model and heuristic decisions')
    expect(within(decisions).getByText('Healthcare')).toBeVisible()
    expect(within(decisions).getByText('Housing')).toBeVisible()
    expect(screen.getByText('The learned model and static heuristic choose different actions.')).toBeVisible()
    expect(screen.getByText('onnxruntime / CPUExecutionProvider')).toBeVisible()
    expect(screen.getByText(/69\.0% confidence/)).toBeVisible()
  })

  it('applies presets and resets to the registered default state', () => {
    vi.stubGlobal('fetch', vi.fn())
    render(<ModelToolbox benchmark={toolboxBenchmarkFixture()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Healthcare pressure' }))
    const healthcareForecast = document.getElementById('toolbox-public_forecast_signal-healthcare') as HTMLInputElement
    expect(healthcareForecast.value).toBe('1.55')
    expect(screen.getByRole('radio', { name: /Regime D/ })).toBeChecked()

    fireEvent.click(screen.getByRole('button', { name: 'Reset' }))
    expect(healthcareForecast.value).toBe(String(DEFAULT_TOOLBOX_INPUT.public_forecast_signal[3]))
    expect(screen.getByRole('radio', { name: /Regime C/ })).toBeChecked()
  })

  it('keeps the toolbox mounted when the scenario window slider changes', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => toolboxResponseFixture(),
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<ModelToolbox benchmark={toolboxBenchmarkFixture()} />)

    const phaseWindow = document.getElementById('toolbox-phase-window') as HTMLInputElement
    fireEvent.change(phaseWindow, { target: { value: '7' } })

    expect(phaseWindow.value).toBe('7')
    expect(screen.getByText('7 / 12')).toBeVisible()
    expect(screen.getByRole('heading', { name: 'Change the city signals. Run the real model.' })).toBeVisible()

    fireEvent.click(screen.getByRole('button', { name: 'Run real model' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    const request = JSON.parse(String(fetchMock.mock.calls[0][1]?.body)) as Record<string, unknown>
    expect(request.phase_window).toBe(7)
  })

  it('shows actionable backend validation and does not display an old decision', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({
        error: {
          code: 'INVALID_TOOLBOX_INPUT',
          message: 'Toolbox input validation failed.',
          details: [{ field: 'phase_window', message: 'Must be between 1 and 12.', type: 'range' }],
        },
      }),
    }))
    render(<ModelToolbox benchmark={toolboxBenchmarkFixture()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Run real model' }))

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Toolbox input validation failed.')
    expect(alert).toHaveTextContent('phase_window: Must be between 1 and 12.')
    expect(screen.queryByLabelText('Model and heuristic decisions')).not.toBeInTheDocument()
  })
})
