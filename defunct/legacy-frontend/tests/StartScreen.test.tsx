import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { StartScreen } from '../src/game/StartScreen'

afterEach(cleanup)

describe('game start screen', () => {
  it('defaults to a six-disaster Moderate Stress Test without exposing raw controls', () => {
    render(<StartScreen onStart={vi.fn()} />)

    expect(screen.getByRole('radio', { name: /Stress Test/i })).toBeChecked()
    expect(screen.getByRole('radio', { name: /Moderate/i })).toBeChecked()
    expect(screen.getByRole('radio', { name: /Fault-line city/i })).toBeChecked()
    expect(screen.getByRole('img', { name: '6 disasters available' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Start Stress Test' })).toBeVisible()
    expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument()
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
    expect(screen.getByRole('tooltip', { name: /ambient shock probability to 0.20/i })).toHaveTextContent('daily supply arrivals to 180 units')
    expect(screen.getByRole('button', { name: 'Show Moderate parameter changes' })).toHaveAccessibleDescription(/ambient shock probability to 0.20/i)
  })

  it('labels Sandbox as unlimited and starts the selected session', () => {
    const onStart = vi.fn()
    render(<StartScreen onStart={onStart} />)

    fireEvent.click(screen.getByRole('radio', { name: /Sandbox/i }))
    fireEvent.click(screen.getByRole('radio', { name: /Severe/i }))
    fireEvent.click(screen.getByRole('radio', { name: /Coastal storm season/i }))

    expect(screen.getByRole('img', { name: 'Unlimited disasters' })).toBeVisible()
    expect(screen.getByText('Unlimited in Sandbox')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Start Sandbox' }))
    expect(onStart).toHaveBeenCalledWith({ mode: 'sandbox', difficulty: 'severe', preset: 'coastal' })
  })

  it('discloses all authored mixes without revealing ordered days and launches the guided run separately', () => {
    const onStartTutorial = vi.fn()
    render(<StartScreen onStart={vi.fn()} onStartTutorial={onStartTutorial} />)

    expect(screen.getByText('Earthquake ×2 at raw 0.24 / 0.31 · Utility ×1 at raw 0.22')).toBeVisible()
    expect(screen.getByText('Weather ×2 at raw 0.24 / 0.34 · Supply ×1 at raw 0.18')).toBeVisible()
    expect(screen.queryByText(/Day 3 · Earthquake/i)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Start guided run' }))
    expect(onStartTutorial).toHaveBeenCalledOnce()
  })

  it('offers the Analyst Toolbox as a quiet secondary route when supplied', () => {
    const onOpenToolbox = vi.fn()
    render(<StartScreen onStart={vi.fn()} onOpenToolbox={onOpenToolbox} />)

    fireEvent.click(screen.getByRole('button', { name: 'Analyst Toolbox' }))
    expect(onOpenToolbox).toHaveBeenCalledOnce()
  })
})
