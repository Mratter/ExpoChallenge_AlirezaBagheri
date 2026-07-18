import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { StartScreen } from '../src/game/StartScreen'

afterEach(cleanup)

describe('game start screen', () => {
  it('defaults to a six-disaster Moderate Stress Test without exposing raw controls', () => {
    render(<StartScreen onStart={vi.fn()} />)

    expect(screen.getByRole('radio', { name: /Stress Test/i })).toBeChecked()
    expect(screen.getByRole('radio', { name: /Moderate/i })).toBeChecked()
    expect(screen.getByRole('img', { name: '6 disasters available' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Start Stress Test' })).toBeVisible()
    expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument()
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
  })

  it('labels Sandbox as unlimited and starts the selected session', () => {
    const onStart = vi.fn()
    render(<StartScreen onStart={onStart} />)

    fireEvent.click(screen.getByRole('radio', { name: /Sandbox/i }))
    fireEvent.click(screen.getByRole('radio', { name: /Severe/i }))

    expect(screen.getByRole('img', { name: 'Unlimited disasters' })).toBeVisible()
    expect(screen.getByText('Unlimited in Sandbox')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Start Sandbox' }))
    expect(onStart).toHaveBeenCalledWith({ mode: 'sandbox', difficulty: 'severe' })
  })

  it('offers the Analyst Toolbox as a quiet secondary route when supplied', () => {
    const onOpenToolbox = vi.fn()
    render(<StartScreen onStart={vi.fn()} onOpenToolbox={onOpenToolbox} />)

    fireEvent.click(screen.getByRole('button', { name: 'Analyst Toolbox' }))
    expect(onOpenToolbox).toHaveBeenCalledOnce()
  })
})
