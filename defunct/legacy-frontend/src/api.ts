import type {
  AuditActor,
  CompareResponse,
  ExecutedPlan,
  JudgeDemoResponse,
  PlanRecord,
  PlanningSession,
  SavedResultSummary,
  Scenario,
  V5DevelopmentSnapshot,
  V5JudgeDemo,
  V5OperatorPlan,
  V5ProfilesResponse,
} from './types'

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

async function responseError(response: Response): Promise<ComparisonError> {
  const payload = (await response.json().catch(() => ({}))) as ApiError
  return new ComparisonError(
    payload.error?.code ?? 'REQUEST_FAILED',
    payload.error?.message ?? `Request failed (${response.status})`,
  )
}

export async function listSimulations(signal?: AbortSignal): Promise<SavedResultSummary[]> {
  const response = await fetch('/api/v1/simulations', { signal })
  if (!response.ok) throw await responseError(response)
  const payload = (await response.json()) as { results: SavedResultSummary[] }
  return payload.results
}

export async function loadSimulation(
  resultId: string,
  signal?: AbortSignal,
): Promise<CompareResponse> {
  const response = await fetch(`/api/v1/simulations/${resultId}`, { signal })
  if (!response.ok) throw await responseError(response)
  return (await response.json()) as CompareResponse
}

export async function loadJudgeDemo(signal?: AbortSignal): Promise<JudgeDemoResponse> {
  const response = await fetch('/api/v1/scientific-v4/judge-demo', { signal })
  if (!response.ok) throw await responseError(response)
  return (await response.json()) as JudgeDemoResponse
}

export async function loadV5Profiles(signal?: AbortSignal): Promise<V5ProfilesResponse> {
  const response = await fetch('/api/v5/profiles', { signal })
  if (!response.ok) throw await responseError(response)
  return (await response.json()) as V5ProfilesResponse
}

export async function loadV5DevelopmentSnapshot(
  profile: string,
  signal?: AbortSignal,
): Promise<V5DevelopmentSnapshot> {
  const query = new URLSearchParams({ profile })
  const response = await fetch(`/api/v5/development/snapshot?${query}`, { signal })
  if (!response.ok) throw await responseError(response)
  return (await response.json()) as V5DevelopmentSnapshot
}

export type V5OperatorIdentity = {
  operator_id: string
  session_id: string
  reason: string
}

export async function createV5OperatorPlan(
  profileId: string,
  seed: number,
  identity: V5OperatorIdentity,
  executionMode: 'approval_required' | 'simulation_only_auto_execute' = 'approval_required',
): Promise<V5OperatorPlan> {
  return postJson('/api/v5/operator/plans', {
    profile_id: profileId,
    seed,
    ...identity,
    execution_mode: executionMode,
  })
}

export async function transitionV5OperatorPlan(
  plan: V5OperatorPlan,
  action: 'review' | 'approve' | 'reject' | 'reproject' | 'execute',
  identity: V5OperatorIdentity,
): Promise<V5OperatorPlan> {
  return postJson(`/api/v5/operator/plans/${plan.plan_id}/${action}`, {
    expected_version: plan.version,
    ...identity,
  })
}

export async function overrideV5OperatorPlan(
  plan: V5OperatorPlan,
  identity: V5OperatorIdentity,
  change: { project_id: string; amount: number },
): Promise<V5OperatorPlan> {
  return postJson(`/api/v5/operator/plans/${plan.plan_id}/override`, {
    expected_version: plan.version,
    ...identity,
    changes: [change],
  })
}

export async function loadV5JudgeDemo(
  inputs?: {
    checkpointPath?: string
    checkpointSha256?: string
    candidateId?: string
    sharedValidationIndexPath?: string
    sharedValidationIndexSha256?: string
  },
  signal?: AbortSignal,
): Promise<V5JudgeDemo> {
  if (inputs) {
    return postJson('/api/v5/judge-demo', {
      checkpoint_path: inputs.checkpointPath,
      checkpoint_sha256: inputs.checkpointSha256,
      candidate_id: inputs.candidateId ?? 'relational_gnn_ppo',
      shared_validation_index_path: inputs.sharedValidationIndexPath,
      shared_validation_index_sha256: inputs.sharedValidationIndexSha256,
    })
  }
  const response = await fetch('/api/v5/judge-demo', { signal })
  if (!response.ok) throw await responseError(response)
  return (await response.json()) as V5JudgeDemo
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw await responseError(response)
  return (await response.json()) as T
}

export async function createPlanningSession(
  label: string,
  sourceResultId: string,
  simulationAutoExecute = false,
): Promise<PlanningSession> {
  return postJson('/api/v1/planning-sessions', {
    label,
    source_result_id: sourceResultId,
    simulation_auto_execute: simulationAutoExecute,
  })
}

export async function createPlan(
  sessionId: string,
  sourceResultId: string,
  actor: AuditActor,
  reason: string,
): Promise<PlanRecord> {
  return postJson('/api/v1/plans', {
    session_id: sessionId,
    source_result_id: sourceResultId,
    actor,
    reason,
  })
}

export async function loadPlan(planId: string): Promise<PlanRecord> {
  const response = await fetch(`/api/v1/plans/${planId}`)
  if (!response.ok) throw await responseError(response)
  return (await response.json()) as PlanRecord
}

type PlanTransition = {
  expected_version: number
  actor: AuditActor
  reason: string
}

export async function transitionPlan(
  planId: string,
  action: 'review' | 'approve' | 'reject' | 'reproject',
  request: PlanTransition,
): Promise<PlanRecord> {
  return postJson(`/api/v1/plans/${planId}/${action}`, request)
}

export async function overridePlan(
  planId: string,
  request: PlanTransition & { changes: Array<{ day: number; proposal: number[] }> },
): Promise<PlanRecord> {
  return postJson(`/api/v1/plans/${planId}/override`, request)
}

export async function executePlan(
  planId: string,
  request: PlanTransition,
): Promise<ExecutedPlan> {
  return postJson(`/api/v1/plans/${planId}/execute`, request)
}
