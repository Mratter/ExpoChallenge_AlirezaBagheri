import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  evaluateToolbox,
  isToolboxEvaluationResponse,
  ToolboxEvaluationError,
} from '../src/api'
import { toolboxInputFixture, toolboxResponseFixture } from './toolboxFixtures'

afterEach(() => vi.unstubAllGlobals())

describe('model toolbox evidence client', () => {
  it('accepts the exact versioned inference response and rejects malformed model output', () => {
    expect(isToolboxEvaluationResponse(toolboxResponseFixture())).toBe(true)

    const badVector = toolboxResponseFixture()
    badVector.input.vector.pop()
    expect(isToolboxEvaluationResponse(badVector)).toBe(false)

    const badScoreline = toolboxResponseFixture()
    badScoreline.benchmark_summary.head_to_head.ties = 1
    expect(isToolboxEvaluationResponse(badScoreline)).toBe(false)

    const badRuntime = toolboxResponseFixture() as unknown as { runtime: { real_model_inference: boolean } }
    badRuntime.runtime.real_model_inference = false
    expect(isToolboxEvaluationResponse(badRuntime)).toBe(false)
  })

  it('posts structured public signals and returns real-model output', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => toolboxResponseFixture(),
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await evaluateToolbox(toolboxInputFixture)
    expect(result.model.id).toBe('adaptive-cascade-mlp-v2-300k')
    expect(fetchMock).toHaveBeenCalledWith('/api/workbench/v1/toolbox/evaluate', expect.objectContaining({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(toolboxInputFixture),
    }))
  })

  it('surfaces canonical validation details instead of stale output', async () => {
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

    const error = await evaluateToolbox(toolboxInputFixture).catch((caught: unknown) => caught)
    expect(error).toBeInstanceOf(ToolboxEvaluationError)
    expect(error).toMatchObject({
      status: 422,
      message: 'Toolbox input validation failed.',
      details: [{ field: 'phase_window', message: 'Must be between 1 and 12.', type: 'range' }],
    })
  })
})
