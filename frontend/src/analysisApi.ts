import {
  actionOrder,
  observationOrder,
  services,
  type OfficialOutcome,
  type Service,
  type Vector5,
  type Vector22,
} from './types'

export type AnalysisPlanner = 'candidate' | 'baseline'
export type RecoveryPlanFormat = 'csv' | 'pdf'

export type AnalysisExpectation = {
  resultId: string
  policySha256?: string
  shockScheduleSha256?: string
  originalTrajectorySha256?: string
}

export type AnalysisComparisonIdentity = {
  result_id: string
  policy: { sha256: string }
  shock_schedule_sha256: string
  candidate?: { trajectory_sha256: string }
}

export type ExplanationChannel = {
  observation_index: number
  observation_name: string
  observed_value: number
  mean_absolute_action_delta: number
  normalized_influence: number
  influence_rank: number
  most_affected_action_index: number
  most_affected_action: string
  signed_action_delta: number
}

export type ExplanationDay = {
  day: number
  base_raw_action: Vector22
  channels: ExplanationChannel[]
}

export type ExplanationResponse = {
  schema_version: '1.0.0'
  result_id: string
  method: {
    id: string
    description: string
    interpretation: string
    causal: false
    occlusion_value: 0
    batch_size_per_day: 73
    normalization: string
    future_tape_visible: false
  }
  policy: { id: string; sha256: string }
  shock_schedule_sha256: string
  future_tape_visible: false
  day_count: 30
  observation_count: 73
  action_count: 22
  observation_order: string[]
  action_order: string[]
  days: ExplanationDay[]
}

export type CounterfactualRequest = {
  day: number
  material_shares?: Vector5
  crew_shares?: Vector5
}

export type CounterfactualSummary = {
  solved: boolean
  rauc: number
  final_resilience: number
  minimum_resilience: number
  critical_service_days: number
  hard_violation_count: number
  absolute_outcome: OfficialOutcome
  trajectory_sha256: string
}

export type CounterfactualResponse = {
  schema_version: '1.0.0'
  result_id: string
  analysis_id: string
  analysis_only: true
  persisted: false
  policy_sha256: string
  shock_schedule_sha256: string
  same_disaster_tape: true
  future_tape_visible: false
  treatment: {
    day: number
    material_shares: Vector5 | null
    crew_shares: Vector5 | null
  }
  unchanged_prefix: {
    days: number
    original_sha256: string
    counterfactual_sha256: string
    matches: true
  }
  selected_day_realized_allocations: {
    services: Service[]
    original: { material: Vector5; crew: Vector5 }
    counterfactual: { material: Vector5; crew: Vector5 }
  }
  original: CounterfactualSummary
  counterfactual: CounterfactualSummary
  daily_deltas: Array<{
    day: number
    services_end: Vector5
    preparedness_end: Vector5
    resilience: number
    reward: number
  }>
}

type ErrorEnvelope = {
  error?: {
    code?: unknown
    message?: unknown
    details?: unknown
  }
}

export class AnalysisApiError extends Error {
  readonly code: string
  readonly status: number | null

  constructor(code: string, message: string, status: number | null = null) {
    super(message)
    this.name = 'AnalysisApiError'
    this.code = code
    this.status = status
  }
}

function contractError(message: string): AnalysisApiError {
  return new AnalysisApiError('INVALID_ANALYSIS_CONTRACT', message)
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function objectAt(value: unknown, path: string): Record<string, unknown> {
  if (!isObject(value)) throw contractError(`${path} must be an object.`)
  return value
}

function stringAt(value: unknown, path: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw contractError(`${path} must be a non-empty string.`)
  }
  return value
}

function numberAt(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw contractError(`${path} must be a finite number.`)
  }
  return value
}

function integerAt(value: unknown, path: string, minimum: number, maximum: number): number {
  const parsed = numberAt(value, path)
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw contractError(`${path} must be an integer from ${minimum} to ${maximum}.`)
  }
  return parsed
}

function booleanAt(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') throw contractError(`${path} must be a boolean.`)
  return value
}

function hashAt(value: unknown, path: string): string {
  const parsed = stringAt(value, path)
  if (!/^[0-9a-f]{64}$/.test(parsed)) {
    throw contractError(`${path} must be a lowercase SHA-256 digest.`)
  }
  return parsed
}

function vectorAt(value: unknown, length: 5, path: string): Vector5
function vectorAt(value: unknown, length: 22, path: string): Vector22
function vectorAt(value: unknown, length: 5 | 22, path: string): Vector5 | Vector22 {
  if (
    !Array.isArray(value)
    || value.length !== length
    || !value.every((entry) => typeof entry === 'number' && Number.isFinite(entry))
  ) {
    throw contractError(`${path} must contain exactly ${length} finite numbers.`)
  }
  return value as Vector5 | Vector22
}

function exactOrderAt(
  value: unknown,
  expected: readonly string[],
  path: string,
): void {
  if (
    !Array.isArray(value)
    || value.length !== expected.length
    || value.some((entry, index) => entry !== expected[index])
  ) {
    throw contractError(`${path} does not match the canonical order.`)
  }
}

function validateExpectedIdentity(
  root: Record<string, unknown>,
  expectation: AnalysisExpectation,
): void {
  if (root.result_id !== expectation.resultId) {
    throw contractError('Analysis result_id does not match the requested comparison.')
  }
  if (
    expectation.policySha256 !== undefined
    && root.policy_sha256 !== undefined
    && root.policy_sha256 !== expectation.policySha256
  ) {
    throw contractError('Analysis policy SHA-256 does not match the comparison.')
  }
  if (
    expectation.shockScheduleSha256 !== undefined
    && root.shock_schedule_sha256 !== expectation.shockScheduleSha256
  ) {
    throw contractError('Analysis disaster-tape SHA-256 does not match the comparison.')
  }
}

function validateOfficialOutcome(value: unknown, solved: boolean, path: string): OfficialOutcome {
  const outcome = objectAt(value, path)
  if (booleanAt(outcome.solved, `${path}.solved`) !== solved) {
    throw contractError(`${path}.solved disagrees with its summary.`)
  }
  if (outcome.status !== (solved ? 'solved' : 'failed')) {
    throw contractError(`${path}.status disagrees with solved.`)
  }
  hashAt(outcome.definition_sha256, `${path}.definition_sha256`)
  vectorAt(outcome.recovery_targets, 5, `${path}.recovery_targets`)
  vectorAt(outcome.tail_minimum_services, 5, `${path}.tail_minimum_services`)
  if (
    !Array.isArray(outcome.target_met_by_service)
    || outcome.target_met_by_service.length !== services.length
    || !outcome.target_met_by_service.every((entry) => typeof entry === 'boolean')
  ) {
    throw contractError(`${path}.target_met_by_service must contain five booleans.`)
  }
  if (outcome.assessment_tail_days !== 3) {
    throw contractError(`${path}.assessment_tail_days must be 3.`)
  }
  for (const field of [
    'resilience_auc',
    'resilience_auc_floor',
    'critical_service_days',
    'critical_service_day_cap',
    'hard_violation_count',
    'max_conservation_residual',
  ]) numberAt(outcome[field], `${path}.${field}`)
  vectorAt(outcome.terminal_pending_arrivals, 5, `${path}.terminal_pending_arrivals`)
  vectorAt(outcome.terminal_pending_capacity, 5, `${path}.terminal_pending_capacity`)
  return value as OfficialOutcome
}

export function parseExplanations(
  value: unknown,
  expectation: AnalysisExpectation,
): ExplanationResponse {
  const root = objectAt(value, 'explanation response')
  if (root.schema_version !== '1.0.0') {
    throw contractError('Explanation schema_version must be 1.0.0.')
  }
  stringAt(root.result_id, 'explanation response.result_id')
  const policy = objectAt(root.policy, 'explanation response.policy')
  stringAt(policy.id, 'explanation response.policy.id')
  hashAt(policy.sha256, 'explanation response.policy.sha256')
  hashAt(root.shock_schedule_sha256, 'explanation response.shock_schedule_sha256')
  if (root.future_tape_visible !== false) {
    throw contractError('Explanation must not expose the future disaster tape.')
  }
  if (
    root.day_count !== 30
    || root.observation_count !== observationOrder.length
    || root.action_count !== actionOrder.length
  ) {
    throw contractError('Explanation dimensions do not match the runtime contract.')
  }
  exactOrderAt(root.observation_order, observationOrder, 'explanation response.observation_order')
  exactOrderAt(root.action_order, actionOrder, 'explanation response.action_order')

  const method = objectAt(root.method, 'explanation response.method')
  stringAt(method.id, 'explanation response.method.id')
  stringAt(method.description, 'explanation response.method.description')
  stringAt(method.interpretation, 'explanation response.method.interpretation')
  if (
    method.causal !== false
    || method.occlusion_value !== 0
    || method.batch_size_per_day !== observationOrder.length
    || method.future_tape_visible !== false
  ) {
    throw contractError('Explanation method safety contract is invalid.')
  }
  stringAt(method.normalization, 'explanation response.method.normalization')

  if (!Array.isArray(root.days) || root.days.length !== 30) {
    throw contractError('Explanation must contain exactly 30 days.')
  }
  root.days.forEach((dayValue, dayIndex) => {
    const dayPath = `explanation response.days[${dayIndex}]`
    const day = objectAt(dayValue, dayPath)
    if (day.day !== dayIndex + 1) throw contractError(`${dayPath}.day is not sequential.`)
    vectorAt(day.base_raw_action, 22, `${dayPath}.base_raw_action`)
    if (!Array.isArray(day.channels) || day.channels.length !== observationOrder.length) {
      throw contractError(`${dayPath}.channels must contain all 73 observations.`)
    }
    const ranks = new Set<number>()
    let influenceSum = 0
    day.channels.forEach((channelValue, observationIndex) => {
      const channelPath = `${dayPath}.channels[${observationIndex}]`
      const channel = objectAt(channelValue, channelPath)
      if (
        channel.observation_index !== observationIndex
        || channel.observation_name !== observationOrder[observationIndex]
      ) throw contractError(`${channelPath} does not match the canonical observation.`)
      numberAt(channel.observed_value, `${channelPath}.observed_value`)
      const sensitivity = numberAt(
        channel.mean_absolute_action_delta,
        `${channelPath}.mean_absolute_action_delta`,
      )
      const influence = numberAt(channel.normalized_influence, `${channelPath}.normalized_influence`)
      if (sensitivity < 0 || influence < 0 || influence > 1) {
        throw contractError(`${channelPath} contains an invalid influence value.`)
      }
      influenceSum += influence
      const rank = integerAt(channel.influence_rank, `${channelPath}.influence_rank`, 1, 73)
      if (ranks.has(rank)) throw contractError(`${dayPath} contains duplicate influence ranks.`)
      ranks.add(rank)
      const actionIndex = integerAt(
        channel.most_affected_action_index,
        `${channelPath}.most_affected_action_index`,
        0,
        actionOrder.length - 1,
      )
      if (channel.most_affected_action !== actionOrder[actionIndex]) {
        throw contractError(`${channelPath}.most_affected_action does not match its index.`)
      }
      numberAt(channel.signed_action_delta, `${channelPath}.signed_action_delta`)
    })
    if (ranks.size !== observationOrder.length) {
      throw contractError(`${dayPath} must contain every influence rank exactly once.`)
    }
    if (influenceSum !== 0 && Math.abs(influenceSum - 1) > 1e-7) {
      throw contractError(`${dayPath} normalized influences must sum to one.`)
    }
  })

  const explanationExpectation = {
    resultId: expectation.resultId,
    shockScheduleSha256: expectation.shockScheduleSha256,
  }
  validateExpectedIdentity(root, explanationExpectation)
  if (expectation.policySha256 !== undefined && policy.sha256 !== expectation.policySha256) {
    throw contractError('Explanation policy SHA-256 does not match the comparison.')
  }
  return value as ExplanationResponse
}

function validateNormalizedShares(value: unknown, path: string): Vector5 | null {
  if (value === null) return null
  const vector = vectorAt(value, 5, path)
  if (vector.some((entry) => entry < 0)) {
    throw contractError(`${path} cannot contain negative values.`)
  }
  const total = vector.reduce((sum, entry) => sum + entry, 0)
  if (Math.abs(total - 1) > 1e-7) {
    throw contractError(`${path} must sum to one.`)
  }
  return vector
}

function validateCounterfactualSummary(value: unknown, path: string): CounterfactualSummary {
  const summary = objectAt(value, path)
  const solved = booleanAt(summary.solved, `${path}.solved`)
  for (const field of [
    'rauc',
    'final_resilience',
    'minimum_resilience',
    'critical_service_days',
    'hard_violation_count',
  ]) numberAt(summary[field], `${path}.${field}`)
  hashAt(summary.trajectory_sha256, `${path}.trajectory_sha256`)
  validateOfficialOutcome(summary.absolute_outcome, solved, `${path}.absolute_outcome`)
  return value as CounterfactualSummary
}

export function parseCounterfactual(
  value: unknown,
  expectation: AnalysisExpectation,
): CounterfactualResponse {
  const root = objectAt(value, 'counterfactual response')
  if (root.schema_version !== '1.0.0') {
    throw contractError('Counterfactual schema_version must be 1.0.0.')
  }
  stringAt(root.result_id, 'counterfactual response.result_id')
  hashAt(root.analysis_id, 'counterfactual response.analysis_id')
  hashAt(root.policy_sha256, 'counterfactual response.policy_sha256')
  hashAt(root.shock_schedule_sha256, 'counterfactual response.shock_schedule_sha256')
  if (
    root.analysis_only !== true
    || root.persisted !== false
    || root.same_disaster_tape !== true
    || root.future_tape_visible !== false
  ) {
    throw contractError('Counterfactual safety and persistence flags are invalid.')
  }

  const treatment = objectAt(root.treatment, 'counterfactual response.treatment')
  const day = integerAt(treatment.day, 'counterfactual response.treatment.day', 1, 30)
  const material = validateNormalizedShares(
    treatment.material_shares,
    'counterfactual response.treatment.material_shares',
  )
  const crew = validateNormalizedShares(
    treatment.crew_shares,
    'counterfactual response.treatment.crew_shares',
  )
  if (material === null && crew === null) {
    throw contractError('Counterfactual treatment must include at least one allocation.')
  }

  const prefix = objectAt(root.unchanged_prefix, 'counterfactual response.unchanged_prefix')
  if (prefix.days !== day - 1 || prefix.matches !== true) {
    throw contractError('Counterfactual unchanged-prefix proof is invalid.')
  }
  const originalPrefix = hashAt(
    prefix.original_sha256,
    'counterfactual response.unchanged_prefix.original_sha256',
  )
  const changedPrefix = hashAt(
    prefix.counterfactual_sha256,
    'counterfactual response.unchanged_prefix.counterfactual_sha256',
  )
  if (originalPrefix !== changedPrefix) {
    throw contractError('Counterfactual changed evidence before the selected day.')
  }

  const realized = objectAt(
    root.selected_day_realized_allocations,
    'counterfactual response.selected_day_realized_allocations',
  )
  exactOrderAt(
    realized.services,
    services,
    'counterfactual response.selected_day_realized_allocations.services',
  )
  for (const side of ['original', 'counterfactual'] as const) {
    const allocations = objectAt(
      realized[side],
      `counterfactual response.selected_day_realized_allocations.${side}`,
    )
    vectorAt(
      allocations.material,
      5,
      `counterfactual response.selected_day_realized_allocations.${side}.material`,
    )
    vectorAt(
      allocations.crew,
      5,
      `counterfactual response.selected_day_realized_allocations.${side}.crew`,
    )
  }

  validateCounterfactualSummary(root.original, 'counterfactual response.original')
  validateCounterfactualSummary(root.counterfactual, 'counterfactual response.counterfactual')
  if (
    expectation.originalTrajectorySha256 !== undefined
    && objectAt(root.original, 'counterfactual response.original').trajectory_sha256
      !== expectation.originalTrajectorySha256
  ) {
    throw contractError('Counterfactual original trajectory does not match the comparison.')
  }
  if (!Array.isArray(root.daily_deltas) || root.daily_deltas.length !== 30) {
    throw contractError('Counterfactual must report exactly 30 daily deltas.')
  }
  root.daily_deltas.forEach((deltaValue, index) => {
    const path = `counterfactual response.daily_deltas[${index}]`
    const delta = objectAt(deltaValue, path)
    if (delta.day !== index + 1) throw contractError(`${path}.day is not sequential.`)
    vectorAt(delta.services_end, 5, `${path}.services_end`)
    vectorAt(delta.preparedness_end, 5, `${path}.preparedness_end`)
    numberAt(delta.resilience, `${path}.resilience`)
    numberAt(delta.reward, `${path}.reward`)
  })

  validateExpectedIdentity(root, expectation)
  return value as CounterfactualResponse
}

export function normalizeShareVector(values: readonly number[], label: string): Vector5 {
  if (values.length !== services.length || values.some((entry) => !Number.isFinite(entry))) {
    throw new AnalysisApiError(
      'INVALID_COUNTERFACTUAL',
      `${label} must contain five finite values.`,
    )
  }
  if (values.some((entry) => entry < 0)) {
    throw new AnalysisApiError('INVALID_COUNTERFACTUAL', `${label} cannot contain negative values.`)
  }
  const total = values.reduce((sum, entry) => sum + entry, 0)
  if (total <= 0) {
    throw new AnalysisApiError('INVALID_COUNTERFACTUAL', `${label} must have a positive total.`)
  }
  return values.map((entry) => entry / total) as Vector5
}

export function createCounterfactualRequest(
  day: number,
  allocations: {
    materialShares?: readonly number[]
    crewShares?: readonly number[]
  },
): CounterfactualRequest {
  if (!Number.isInteger(day) || day < 1 || day > 30) {
    throw new AnalysisApiError('INVALID_COUNTERFACTUAL', 'Day must be an integer from 1 to 30.')
  }
  if (allocations.materialShares === undefined && allocations.crewShares === undefined) {
    throw new AnalysisApiError(
      'INVALID_COUNTERFACTUAL',
      'Provide material shares, crew shares, or both.',
    )
  }
  return {
    day,
    ...(allocations.materialShares === undefined
      ? {}
      : { material_shares: normalizeShareVector(allocations.materialShares, 'Material shares') }),
    ...(allocations.crewShares === undefined
      ? {}
      : { crew_shares: normalizeShareVector(allocations.crewShares, 'Crew shares') }),
  }
}

function resourceUrl(resultId: string, resource: string): string {
  return `/api/v1/simulations/${encodeURIComponent(resultId)}/${resource}`
}

export function recoveryPlanUrl(
  resultId: string,
  planner: AnalysisPlanner,
  format: RecoveryPlanFormat,
): string {
  const query = new URLSearchParams({ planner, format })
  return `${resourceUrl(resultId, 'recovery-plan')}?${query.toString()}`
}

async function responseError(response: Response): Promise<AnalysisApiError> {
  const payload: unknown = await response.json().catch(() => null)
  const envelope: ErrorEnvelope = isObject(payload) ? payload : {}
  const error = isObject(envelope.error) ? envelope.error : {}
  const details = Array.isArray(error.details) ? error.details : []
  const first = details.length > 0 && isObject(details[0]) ? details[0] : null
  const detailMessage = first === null
    ? null
    : typeof first.message === 'string'
      ? first.message
      : typeof first.msg === 'string'
        ? first.msg
        : null
  return new AnalysisApiError(
    typeof error.code === 'string' ? error.code : 'ANALYSIS_REQUEST_FAILED',
    detailMessage
      ?? (typeof error.message === 'string' ? error.message : `Request failed (${response.status}).`),
    response.status,
  )
}

export async function fetchDecisionExplanations(
  comparison: AnalysisComparisonIdentity,
  signal?: AbortSignal,
): Promise<ExplanationResponse> {
  const response = await fetch(resourceUrl(comparison.result_id, 'explanations'), { signal })
  if (!response.ok) throw await responseError(response)
  return parseExplanations(await response.json(), {
    resultId: comparison.result_id,
    policySha256: comparison.policy.sha256,
    shockScheduleSha256: comparison.shock_schedule_sha256,
  })
}

export async function runCounterfactualAnalysis(
  comparison: AnalysisComparisonIdentity,
  request: CounterfactualRequest,
  signal?: AbortSignal,
): Promise<CounterfactualResponse> {
  const response = await fetch(resourceUrl(comparison.result_id, 'counterfactuals'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok) throw await responseError(response)
  return parseCounterfactual(await response.json(), {
    resultId: comparison.result_id,
    policySha256: comparison.policy.sha256,
    shockScheduleSha256: comparison.shock_schedule_sha256,
    originalTrajectorySha256: comparison.candidate?.trajectory_sha256,
  })
}
