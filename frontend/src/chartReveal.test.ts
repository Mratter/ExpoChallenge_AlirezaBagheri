import { describe, expect, it } from 'vitest'
import appSource from './App.tsx?raw'
import landingSource from './LandingPage.tsx?raw'
import landingCss from './landing.css?raw'
import appCss from './styles.css?raw'

/**
 * A chart line that carries `vector-effect: non-scaling-stroke` has its dash
 * pattern resolved in device space, while `pathLength` normalizes the same
 * pattern in user space. Revealing a line by animating `stroke-dashoffset`
 * across that pair truncates it at `1 / scale` of its length, and
 * `animation-fill-mode: forwards` freezes the truncation permanently — the
 * animation silently deletes evidence at every render scale above 1:1.
 *
 * Reveal animations must therefore be driven by a clip wipe, never by stroke
 * geometry. A static dash pattern (the pointer crosshair, shock guides) is
 * unaffected: nothing about those lines encodes a measured length.
 */
const stylesheets = [
  { name: 'landing.css', source: landingCss },
  { name: 'styles.css', source: appCss },
]
const chartSources = [
  { name: 'LandingPage.tsx', source: landingSource },
  { name: 'App.tsx', source: appSource },
]

function rules(source: string): { selector: string; body: string }[] {
  return [...source.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map(([, selector, body]) => ({
    selector: selector.trim().split('\n').at(-1)!.trim(),
    body,
  }))
}

describe('chart reveal animations', () => {
  for (const { name, source } of stylesheets) {
    it(`${name} never reveals a line by animating its stroke`, () => {
      expect(source).not.toContain('stroke-dashoffset')
      const animatedDashes = rules(source)
        .filter(({ body }) => /(?:^|;)\s*stroke-dasharray\s*:/.test(body) && /(?:^|;)\s*animation\s*:/.test(body))
        .map(({ selector }) => selector)
      expect(animatedDashes).toEqual([])
    })

    it(`${name} keeps the resting line whole when the reveal never runs`, () => {
      const reveal = rules(source).filter(({ body }) => /animation\s*:[^;]*line-reveal/.test(body))
      expect(reveal.length).toBeGreaterThan(0)
      for (const { selector, body } of reveal) {
        // A fill mode would hold the opening frame whenever the animation is
        // frozen — a backgrounded tab, a dropped frame — hiding the evidence.
        expect(/animation\s*:[^;]*\b(forwards|both)\b/.test(body)).toBe(false)
        // Nothing may clip the line outside the animation's own active phase.
        expect(rules(source).filter((rule) => rule.selector === selector)
          .some((rule) => /(?:^|;)\s*clip-path\s*:/.test(rule.body))).toBe(false)
      }
      const keyframes = /@keyframes [a-z-]*line-reveal \{([\s\S]*?)\n\}/.exec(source)
      expect(keyframes?.[1]).toContain('clip-path: inset(0 100% 0 0)')
      // No closing keyframe: the animation resolves to the element's own state.
      expect(keyframes?.[1]).not.toContain('to {')
    })
  }

  for (const { name, source } of chartSources) {
    it(`${name} draws chart lines at their true length`, () => {
      expect(source).not.toContain('pathLength')
      expect(source).toMatch(/className="(landing-chart|trace)-lines"/)
    })
  }
})
