import {
  actionGroups,
  actionOrder,
  defaultScenario,
  environmentContract,
  observationOrder,
  services,
  type CompareResponse,
  type Metadata,
  type SavedResultSummary,
  type Scenario,
} from './types'

type ApiError = {
  error?: {
    code?: string
    message?: string
    details?: Array<{ message?: string; msg?: string }>
  }
}

export class ComparisonError extends Error {
  readonly code: string
  readonly status: number | null

  constructor(code: string, message: string, status: number | null = null) {
    super(message)
    this.name = 'ComparisonError'
    this.code = code
    this.status = status
  }
}

function contractError(message: string): ComparisonError {
  return new ComparisonError('INVALID_API_CONTRACT', message)
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isStringArray(value: unknown, length?: number): value is string[] {
  return Array.isArray(value)
    && (length === undefined || value.length === length)
    && value.every((entry) => typeof entry === 'string')
}

function isNumberArray(value: unknown, length: number): value is number[] {
  return Array.isArray(value)
    && value.length === length
    && value.every(isFiniteNumber)
}

function isBooleanArray(value: unknown, length: number): value is boolean[] {
  return Array.isArray(value)
    && value.length === length
    && value.every((entry) => typeof entry === 'boolean')
}

function requireObject(value: unknown, path: string): Record<string, unknown> {
  if (!isObject(value)) throw contractError(`${path} must be an object.`)
  return value
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw contractError(`${path} must be a non-empty string.`)
  }
  return value
}

function requireNumber(value: unknown, path: string): number {
  if (!isFiniteNumber(value)) throw contractError(`${path} must be a finite number.`)
  return value
}

function requireInteger(value: unknown, path: string, minimum: number, maximum: number): number {
  const integer = requireNumber(value, path)
  if (!Number.isInteger(integer) || integer < minimum || integer > maximum) {
    throw contractError(`${path} must be an integer from ${minimum} to ${maximum}.`)
  }
  return integer
}

function requireBoolean(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') throw contractError(`${path} must be a boolean.`)
  return value
}

function requireVector(value: unknown, length: number, path: string): number[] {
  if (!isNumberArray(value, length)) {
    throw contractError(`${path} must contain exactly ${length} finite numbers.`)
  }
  return value
}

function requireHash(value: unknown, path: string): string {
  const hash = requireString(value, path)
  if (!/^[0-9a-f]{64}$/.test(hash)) throw contractError(`${path} must be a SHA-256 digest.`)
  return hash
}

function requireExactStringArray(
  value: unknown,
  expected: readonly string[],
  path: string,
): void {
  if (
    !isStringArray(value, expected.length)
    || value.some((entry, index) => entry !== expected[index])
  ) throw contractError(`${path} does not match the canonical order.`)
}

function validatePolicyIdentity(value: unknown, path: string): Record<string, unknown> {
  const policy = requireObject(value, path)
  requireString(policy.id, `${path}.id`)
  requireString(policy.path_stem, `${path}.path_stem`)
  requireString(policy.artifact_type, `${path}.artifact_type`)
  requireString(policy.runtime, `${path}.runtime`)
  requireHash(policy.sha256, `${path}.sha256`)
  const observation = requireObject(
    policy.observation_contract,
    `${path}.observation_contract`,
  )
  if (
    observation.source !== 'raw_environment_observation'
    || observation.input_name !== 'observation'
    || observation.dtype !== 'float32'
    || observation.normalization !== 'embedded_in_onnx'
    || !isNumberArray(observation.shape, 1)
    || observation.shape[0] !== environmentContract.observationCount
  ) throw contractError(`${path}.observation_contract is incompatible with the environment.`)
  return policy
}

function validateOutcome(value: unknown, path: string): boolean {
  const outcome = requireObject(value, path)
  const solved = requireBoolean(outcome.solved, `${path}.solved`)
  if (outcome.status !== (solved ? 'solved' : 'failed')) {
    throw contractError(`${path}.status disagrees with the official solved flag.`)
  }
  requireHash(outcome.definition_sha256, `${path}.definition_sha256`)
  requireVector(outcome.recovery_targets, services.length, `${path}.recovery_targets`)
  requireVector(outcome.tail_minimum_services, services.length, `${path}.tail_minimum_services`)
  if (!isBooleanArray(outcome.target_met_by_service, services.length)) {
    throw contractError(`${path}.target_met_by_service must contain five booleans.`)
  }
  if (outcome.assessment_tail_days !== environmentContract.assessmentTailDays) {
    throw contractError(`${path}.assessment_tail_days must be ${environmentContract.assessmentTailDays}.`)
  }
  const checks = requireObject(outcome.checks, `${path}.checks`)
  for (const name of [
    'zero_hard_violations',
    'conservation_verified',
    'assessment_tail_targets_met',
    'resilience_auc_met',
    'critical_service_day_cap_met',
    'terminal_pending_within_capacity',
  ]) requireBoolean(checks[name], `${path}.checks.${name}`)
  return solved
}

function validateDay(value: unknown, dayIndex: number, planner: string): void {
  const path = `${planner}.trajectory[${dayIndex}]`
  const day = requireObject(value, path)
  if (day.day !== dayIndex + 1) throw contractError(`${path}.day is not sequential.`)
  for (const field of [
    'services_before', 'services_after_shock', 'allocation', 'material_allocation',
    'crew_allocation', 'stock_release', 'preparedness_requested',
    'preparedness_investment', 'preparedness_before', 'preparedness_after_hazard',
    'preparedness_gain_requested', 'preparedness_gain', 'preparedness_end',
    'lower_bounds', 'upper_bounds', 'crew_lower_bounds', 'crew_upper_bounds',
    'support', 'throughput', 'public_next_day_risk', 'gain', 'strain', 'services_end',
  ]) requireVector(day[field], services.length, `${path}.${field}`)
  requireVector(day.raw_action, actionOrder.length, `${path}.raw_action`)
  for (const field of [
    'available_budget', 'available_crew', 'material_used', 'material_unspent',
    'crew_used', 'crew_idle', 'preparedness_alignment_reward', 'backlog_pressure',
    'resilience', 'reward', 'hard_violation_count',
  ]) requireNumber(day[field], `${path}.${field}`)
  for (const field of ['projection', 'crew_projection']) {
    const projection = requireObject(day[field], `${path}.${field}`)
    requireNumber(projection.distance, `${path}.${field}.distance`)
    requireNumber(projection.constraint_violations, `${path}.${field}.constraint_violations`)
  }
  const shock = requireObject(day.shock, `${path}.shock`)
  if (shock.day !== dayIndex + 1) throw contractError(`${path}.shock.day is not sequential.`)
  requireVector(shock.impact, services.length, `${path}.shock.impact`)
  requireVector(shock.public_risk_next, services.length, `${path}.shock.public_risk_next`)
  requireBoolean(shock.assessment_tail, `${path}.shock.assessment_tail`)
  requireNumber(shock.severity, `${path}.shock.severity`)
  requireBoolean(shock.forced, `${path}.shock.forced`)
  const logistics = requireObject(day.logistics, `${path}.logistics`)
  for (const field of [
    'depot_capacity', 'depot_stock_before', 'pending_arrivals',
    'pending_arrivals_landed', 'pending_arrivals_held', 'depot_stock_after_pending',
    'depot_damage_penalty', 'depot_damage_days_remaining', 'depot_damage_factor',
    'throughput_factor', 'mutual_aid_net', 'depot_stock_ready',
    'preparedness_material_requested', 'preparedness_material_consumed',
    'preparedness_effective_work', 'depot_stock_after_preparedness',
    'repair_material_committed', 'preparedness_crew_assigned',
    'preparedness_crew_utilized', 'preparedness_crew_capacity_effective',
    'preparedness_crew_capacity_physical', 'repair_crew_assigned',
    'same_day_delivery_scheduled', 'same_day_delivery_landed',
    'same_day_delivery_held', 'delayed_delivery_scheduled', 'repair_reserve',
    'repair_request', 'total_stock_release_budget',
    'stock_release_remaining_after_preparedness', 'stock_release_limit',
    'crew_capacity_effective', 'crew_capacity_physical', 'repair_dispatch',
    'repair_supply', 'spoilage', 'depot_stock_end', 'pending_next_day',
    'capacity_overflow', 'conservation_residual',
  ]) requireVector(logistics[field], services.length, `${path}.logistics.${field}`)
  requireNumber(logistics.road_capacity, `${path}.logistics.road_capacity`)
  if (!Array.isArray(logistics.mutual_aid_transfers)) {
    throw contractError(`${path}.logistics.mutual_aid_transfers must be an array.`)
  }
}

function validatePlanner(value: unknown, path: string): boolean {
  const planner = requireObject(value, path)
  const trajectory = planner.trajectory
  if (!Array.isArray(trajectory) || trajectory.length !== defaultScenario.horizon_days) {
    throw contractError(`${path}.trajectory must contain exactly ${defaultScenario.horizon_days} days.`)
  }
  trajectory.forEach((day, index) => validateDay(day, index, path))
  for (const field of [
    'rauc', 'final_resilience', 'minimum_resilience',
    'post_shock_recovery_shortfall_auc', 'days_to_pre_shock_recovery_after_largest_loss',
    'largest_shock_loss_day', 'critical_service_days', 'hard_violation_count',
    'constraint_violations', 'max_logistics_conservation_residual',
  ]) requireNumber(planner[field], `${path}.${field}`)
  requireVector(planner.final_depot_stock, services.length, `${path}.final_depot_stock`)
  requireVector(planner.final_pending_arrivals, services.length, `${path}.final_pending_arrivals`)
  requireHash(planner.trajectory_sha256, `${path}.trajectory_sha256`)
  const solved = validateOutcome(planner.absolute_outcome, `${path}.absolute_outcome`)
  const terminalIndex = defaultScenario.horizon_days - 1
  const terminal = requireObject(trajectory[terminalIndex], `${path}.trajectory[${terminalIndex}]`)
  const terminalSolved = validateOutcome(
    terminal.absolute_outcome,
    `${path}.trajectory[${terminalIndex}].absolute_outcome`,
  )
  if (terminalSolved !== solved) {
    throw contractError(`${path} terminal and summary outcomes disagree.`)
  }
  return solved
}

/** Runtime validation prevents incompatible or partial evidence from being rendered. */
export function parseComparison(value: unknown): CompareResponse {
  const root = requireObject(value, 'comparison response')
  if (root.schema_version !== environmentContract.schemaVersion || root.engine_version !== 'city-recovery-env-v3') {
    throw contractError('The response does not match the configured environment schema.')
  }
  const environment = requireObject(root.environment, 'environment')
  if (
    environment.id !== environmentContract.id
    || environment.observation_count !== environmentContract.observationCount
    || environment.action_count !== environmentContract.actionCount
  ) throw contractError('The environment interface is not the canonical 73-input / 22-action contract.')
  const scenario = requireObject(root.scenario, 'scenario')
  if (scenario.horizon_days !== defaultScenario.horizon_days || scenario.assessment_tail_days !== environmentContract.assessmentTailDays) {
    throw contractError('The scenario is not the canonical 30-day / 3-day-tail protocol.')
  }
  requireVector(scenario.initial_services, services.length, 'scenario.initial_services')
  requireVector(scenario.priorities, services.length, 'scenario.priorities')
  requireVector(scenario.recovery_targets, services.length, 'scenario.recovery_targets')
  requireExactStringArray(root.observation_order, observationOrder, 'observation_order')
  requireExactStringArray(root.action_order, actionOrder, 'action_order')
  if (!Array.isArray(root.services) || root.services.join('|') !== services.join('|')) {
    throw contractError('The returned service order does not match the canonical contract.')
  }
  if (!Array.isArray(root.shock_schedule) || root.shock_schedule.length !== defaultScenario.horizon_days) {
    throw contractError('The shared shock schedule must contain 30 days.')
  }
  const candidateSolved = validatePlanner(root.candidate, 'candidate')
  const baselineSolved = validatePlanner(root.baseline, 'baseline')
  const comparison = requireObject(root.comparison, 'comparison')
  if (comparison.primary_metric !== 'independent_absolute_disaster_solved') {
    throw contractError('The primary metric is not the official independent solved verdict.')
  }
  if (comparison.candidate_solved !== candidateSolved || comparison.baseline_solved !== baselineSolved) {
    throw contractError('The comparison verdict disagrees with a planner outcome receipt.')
  }
  requireNumber(comparison.secondary_rauc_candidate_minus_baseline, 'comparison.secondary_rauc_candidate_minus_baseline')
  validatePolicyIdentity(root.policy, 'policy')
  const baselineSpec = requireObject(root.baseline_spec, 'baseline_spec')
  requireString(baselineSpec.id, 'baseline_spec.id')
  requireString(baselineSpec.version, 'baseline_spec.version')
  requireHash(root.result_id, 'result_id')
  requireHash(root.shock_schedule_sha256, 'shock_schedule_sha256')
  requireHash(root.engine_spec_sha256, 'engine_spec_sha256')
  requireHash(root.outcome_definition_sha256, 'outcome_definition_sha256')
  return value as CompareResponse
}

export function parseMetadata(value: unknown): Metadata {
  const root = requireObject(value, 'metadata')
  const model = validatePolicyIdentity(root.model, 'metadata.model')
  const environment = requireObject(root.environment, 'metadata.environment')
  if (
    root.schema_version !== environmentContract.schemaVersion
    || environment.id !== environmentContract.id
    || environment.version !== environmentContract.version
    || environment.observation_count !== environmentContract.observationCount
    || environment.action_count !== environmentContract.actionCount
    || environment.policy_neutral_transition !== true
    || environment.future_tape_visible !== false
    || model.observation_count !== environmentContract.observationCount
    || model.action_count !== environmentContract.actionCount
  ) throw contractError('Metadata does not describe the canonical runtime contract.')
  requireString(root.app, 'metadata.app')
  requireString(root.version, 'metadata.version')
  requireInteger(root.default_seed, 'metadata.default_seed', 0, 4_294_967_295)
  requireExactStringArray(model.observation_order, observationOrder, 'metadata.model.observation_order')
  requireExactStringArray(model.action_order, actionOrder, 'metadata.model.action_order')
  if (!isStringArray(root.services, services.length) || root.services.join('|') !== services.join('|')) {
    throw contractError('Metadata service order does not match the canonical contract.')
  }
  requireExactStringArray(model.action_groups, actionGroups, 'metadata.model.action_groups')
  requireHash(environment.spec_sha256, 'metadata.environment.spec_sha256')
  requireHash(root.outcome_definition_sha256, 'metadata.outcome_definition_sha256')
  requireObject(root.outcome_definition, 'metadata.outcome_definition')
  const baseline = requireObject(root.baseline, 'metadata.baseline')
  if (
    baseline.id !== 'reactive-public-state-heuristic-v3'
    || baseline.version !== '3.0.0'
    || baseline.uses_same_observation_contract !== true
    || baseline.uses_same_action_contract !== true
    || baseline.uses_public_risk_signal !== true
    || baseline.future_tape_visible !== false
  ) throw contractError('Metadata baseline identity or public-causal contract drifted.')
  requireObject(root.persistence, 'metadata.persistence')
  requireString(root.determinism, 'metadata.determinism')
  return value as Metadata
}

function parseSavedSummaries(value: unknown): SavedResultSummary[] {
  const root = requireObject(value, 'saved simulations')
  if (!Array.isArray(root.results)) throw contractError('Saved simulation index is malformed.')
  return root.results.map((entry, index) => {
    const row = requireObject(entry, `saved simulations[${index}]`)
    if (row.engine_version !== 'city-recovery-env-v3' || row.horizon_days !== 30) {
      throw contractError(`saved simulations[${index}] uses an incompatible runtime contract.`)
    }
    requireHash(row.result_id, `saved simulations[${index}].result_id`)
    requireBoolean(row.candidate_solved, `saved simulations[${index}].candidate_solved`)
    requireBoolean(row.baseline_solved, `saved simulations[${index}].baseline_solved`)
    return entry as SavedResultSummary
  })
}

async function responseError(response: Response): Promise<ComparisonError> {
  const payload = (await response.json().catch(() => ({}))) as ApiError
  const detail = payload.error?.details?.[0]
  return new ComparisonError(
    payload.error?.code ?? 'REQUEST_FAILED',
    detail?.message ?? detail?.msg ?? payload.error?.message ?? `Request failed (${response.status})`,
    response.status,
  )
}

export async function fetchMetadata(signal?: AbortSignal): Promise<Metadata> {
  const response = await fetch('/api/v1/meta', { signal })
  if (!response.ok) throw await responseError(response)
  return parseMetadata(await response.json())
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
  if (!response.ok) throw await responseError(response)
  return parseComparison(await response.json())
}

export async function listSimulations(signal?: AbortSignal): Promise<SavedResultSummary[]> {
  const response = await fetch('/api/v1/simulations?engine_version=city-recovery-env-v3', { signal })
  if (!response.ok) throw await responseError(response)
  return parseSavedSummaries(await response.json())
}

export async function loadSimulation(
  resultId: string,
  signal?: AbortSignal,
): Promise<CompareResponse> {
  const response = await fetch(`/api/v1/simulations/${encodeURIComponent(resultId)}`, { signal })
  if (!response.ok) throw await responseError(response)
  return parseComparison(await response.json())
}
