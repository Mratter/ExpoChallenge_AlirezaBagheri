import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App'
import { measuredOverviewFixture, overviewFixture } from './fixtures'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('model workbench', () => {
  it('shows a claim-free loading state while the evidence API is pending', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => undefined)))
    render(<App />)

    expect(screen.getByRole('heading', { name: 'Binding claims to evidence…' })).toBeVisible()
    expect(screen.queryByText('32 / 40')).not.toBeInTheDocument()
  })

  it('fails closed when authoritative evidence is unavailable and retries cleanly', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 503, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => overviewFixture() })
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)

    expect(await screen.findByRole('heading', { name: 'Evidence unavailable' })).toBeVisible()
    expect(screen.queryByText('32 / 40')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Retry evidence connection' }))

    expect(await screen.findByRole('heading', { name: /The trained model is the policy/ })).toBeVisible()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('separates the trained v2 policy from the untrained R22 architecture', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => overviewFixture() }))
    render(<App />)

    expect((await screen.findAllByText('32 / 40'))[0]).toBeVisible()
    expect(screen.getByLabelText('R22 training status')).toHaveTextContent('0')
    expect(screen.getByLabelText('R22 training status')).toHaveTextContent('not trained-model performance')
    expect(screen.getAllByText('30,000')[0]).toBeVisible()

    fireEvent.click(screen.getByRole('button', { name: /Structured graph scheduler R22 \/ V10/ }))
    await waitFor(() => expect(screen.getAllByRole('heading', { name: 'Structured graph scheduler R22 / V10' })[0]).toBeVisible())
    expect(screen.getByLabelText('Structured graph scheduler R22 / V10 training receipt')).toHaveTextContent('Training volume0 transitions')
  })

  it('lets a presenter step through the model decision pipeline', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => overviewFixture() }))
    render(<App />)
    await screen.findByRole('heading', { name: /The trained model is the policy/ })

    fireEvent.click(screen.getByRole('button', { name: /Make feasible/ }))
    const details = screen.getByRole('region', { name: 'Make feasible stage details' })
    expect(details).toHaveTextContent('The solver enforces exact constraints.')
  })

  it('renders the 40-unit benchmark count matrix and unmeasured next benchmark', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => overviewFixture() }))
    render(<App />)
    await screen.findAllByText('32 / 40')

    expect(screen.getByRole('img', { name: '32 candidate wins and 8 baseline wins across 40 held-out scenario units' })).toBeVisible()
    expect(screen.getByText('not yet run')).toBeVisible()
    expect(screen.getByText(/has no measured result/)).toBeVisible()
  })

  it('renders the measured synthetic showcase without conflating independent counts', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => measuredOverviewFixture() }))
    render(<App />)
    await screen.findByRole('heading', { name: /The model learns the warning/ })

    expect(screen.getByRole('link', { name: /OPEN MODEL TOOLBOX/ })).toHaveAttribute('href', '#toolbox')
    expect(screen.getByLabelText('Active 300k model receipt')).toHaveTextContent('300,113')
    expect(screen.queryByRole('button', { name: /Structured graph scheduler R22/ })).not.toBeInTheDocument()
    expect(screen.getByLabelText('Model objective passes: 38 of 40')).toHaveTextContent(/MODEL OBJECTIVE PASSES\s*38\s*\/\s*40/)
    expect(screen.getByLabelText('Static heuristic objective passes: 20 of 40')).toHaveTextContent(/STATIC HEURISTIC OBJECTIVE PASSES\s*20\s*\/\s*40/)
    expect(screen.getByRole('img', { name: 'Direct matched head-to-head: learned model 38 wins, static heuristic 0 wins, and 2 ties across 40 scenarios' })).toBeVisible()
    expect(screen.getByText('A scenario passes when at least 10 of 12 hidden cascade windows are contained by the public action chosen for that window.')).toBeVisible()
    expect(screen.getByText('higher is better')).toBeVisible()
    expect(screen.getByText('exact equality in integer contained-window count')).toBeVisible()
    expect(screen.getByText('+2.250')).toBeVisible()
    expect(screen.getByText('[1.775, 2.750]')).toBeVisible()
    expect(screen.getByText('0.3038310907050')).toBeVisible()
    expect(screen.getByText('0.3365531942800')).toBeVisible()
    expect(screen.getByLabelText('Synthetic benchmark disclosure')).toHaveTextContent('Engineered synthetic benchmark of learnable observable patterns; not real-world validation.')
    expect(screen.getAllByText('artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/final/result.json')[0]).toBeVisible()
    expect(screen.queryByText(/adaptive-cascades-showcase-v1/)).not.toBeInTheDocument()
    expect(screen.getByLabelText('Adaptive Cascade MLP v2 (300k) training receipt')).toHaveTextContent('9,600 labeled windows')
  })
})
