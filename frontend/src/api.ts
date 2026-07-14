import type { CompareResponse, Scenario } from './types'

type ApiError = { error?: { code?: string; message?: string; details?: { message: string }[] } }

export class ComparisonError extends Error {
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.name = 'ComparisonError'
    this.code = code
  }
}

export async function runComparison(
  seed: number,
  scenario: Scenario,
  signal?: AbortSignal,
): Promise<CompareResponse> {
  const response = await fetch('/api/v1/simulations/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seed, scenario }),
    signal,
  })
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as ApiError
    const detail = payload.error?.details?.[0]?.message
    throw new ComparisonError(
      payload.error?.code ?? 'REQUEST_FAILED',
      detail ?? payload.error?.message ?? `Comparison failed (${response.status})`,
    )
  }
  return (await response.json()) as CompareResponse
}
