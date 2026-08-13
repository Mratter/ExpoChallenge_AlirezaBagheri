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
    expect(markup).toContain(mariaRetrospective.caption)
    expect(markup).toContain('Derived recovery index')
    expect(markup).toContain('observed markers')
    expect(markup).toContain('simulation')
  })

  it('renders six accessible charts and two native evidence tables', () => {
    expect(markup.match(/role="img"/g)).toHaveLength(6)
    expect(markup.match(/<table/g)).toHaveLength(2)
    expect(markup.match(/<caption/g)).toHaveLength(2)
  })

  it('renders all benchmark rows only in the separate benchmark section', () => {
    const [retrospective, benchmark] = markup.split('Separate synthetic benchmark')
    expect(retrospective).not.toContain('163/200')
    expect(retrospective).not.toContain('89.6%')
    expect(benchmark).toBeTruthy()
    for (const row of mariaRetrospective.benchmarkRows) expect(benchmark).toContain(row.label)
  })
})
