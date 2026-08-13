import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { mariaRetrospective } from './generated/mariaRetrospective'
import { LandingPage } from './LandingPage'

describe('evidence landing page', () => {
  const markup = renderToStaticMarkup(<LandingPage />)

  it('uses exact application routes and presents both primary views', () => {
    expect(markup).toContain('href="#/toolbox"')
    expect(markup).toContain('href="#/game"')
    expect(markup).toContain('Open Analyst Toolbox')
    expect(markup).toContain('Explore the 3D city')
  })

  it('keeps historical reconstruction and simulation labels explicit', () => {
    expect(markup).toContain('project reconstruction from official records')
    expect(markup).toContain('The shipped v4 line is a simulated alternative')
    expect(markup).toContain('Derived recovery index')
    expect(markup).toContain('observed markers')
    expect(markup).toContain('simulation')
  })

  it('renders six accessible charts and three native evidence tables', () => {
    expect(markup.match(/role="img"/g)).toHaveLength(6)
    expect(markup.match(/<table/g)).toHaveLength(3)
    expect(markup.match(/<caption/g)).toHaveLength(3)
  })

  it('renders substantive numbers only through the generated display contract', () => {
    const { display, interface: tensorInterface } = mariaRetrospective
    expect(markup).toContain(`${display.dayZeroLabel}–${display.dayEndLabel}`)
    expect(markup).toContain(`Day ${display.horizonStart}–${display.dayEnd} · ${display.indexMin}–${display.indexMax}`)
    expect(markup).toContain(`${display.dayCount}</b> dated points`)
    expect(markup).toContain(`${mariaRetrospective.scenarioCount}</b> frozen scenario`)
    expect(markup).toContain(`${mariaRetrospective.syntheticBenchmarkCaseCount}-case final split`)
    expect(markup).toContain(`${tensorInterface.observationCount}-input, ${tensorInterface.actionCount}-action trace`)
    for (const day of display.milestoneDays) expect(markup).toContain(`Day ${day}`)
  })

  it('keeps the reactive heuristic out of the Maria evidence and in separate simulation tables', () => {
    const [retrospective, benchmark] = markup.split('<section class="benchmark-section"')
    expect(retrospective).not.toContain('163/200')
    expect(retrospective).not.toContain('89.6%')
    expect(retrospective).not.toContain('Reactive heuristic')
    expect(retrospective).not.toContain('landing-chart-reactive')
    expect(benchmark).toBeTruthy()
    expect(benchmark).toContain('Selected runtime comparison')
    expect(benchmark).toContain('163/200')
    expect(benchmark).toContain('72/200')
    expect(benchmark).toContain('81.5%')
    expect(benchmark).toContain('36.0%')
    expect(markup).toContain('data-table="maria-milestones"')
    expect(markup).toContain('data-table="runtime-comparison"')
    expect(markup).toContain('data-table="complete-benchmark"')
    expect(benchmark.match(/data-runtime-comparison=/g)).toHaveLength(2)
    expect(benchmark.match(/data-benchmark=/g)).toHaveLength(mariaRetrospective.benchmarkRows.length)
    for (const row of mariaRetrospective.benchmarkRows) expect(benchmark).toContain(row.label)
  })

  it('gives every chart a non-empty accessible title', () => {
    const titles = [...markup.matchAll(/<title id="[^"]+">([^<]+)<\/title>/g)]
    expect(titles).toHaveLength(6)
    for (const [, title] of titles) expect(title.trim()).not.toBe('')
  })
})
