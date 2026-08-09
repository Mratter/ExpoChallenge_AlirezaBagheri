import {
  actionGroupsV3,
  actionOrderV3,
  observationOrderV3,
  services,
  type CompareResponse,
  type MetadataV3,
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

function requireExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  path: string,
): void {
  const actual = Object.keys(value).sort()
  const required = [...expected].sort()
  if (actual.length !== required.length || actual.some((key, index) => key !== required[index])) {
    throw contractError(`${path} fields do not match the sealed contract.`)
  }
}

function requireExactStringArray(
  value: unknown,
  expected: readonly string[],
  path: string,
): void {
  if (
    !isStringArray(value, expected.length)
    || value.some((entry, index) => entry !== expected[index])
  ) throw contractError(`${path} does not match the canonical V3 order.`)
}

function rounded(value: number, digits: number): number {
  return Number(value.toFixed(digits))
}

function wilsonInterval(successes: number, total: number): [number, number] {
  const z = 1.959963984540054
  const proportion = successes / total
  const denominator = 1 + z * z / total
  const center = (proportion + z * z / (2 * total)) / denominator
  const margin = z * Math.sqrt(
    proportion * (1 - proportion) / total + z * z / (4 * total * total),
  ) / denominator
  return [
    rounded(Math.max(0, center - margin), 8),
    rounded(Math.min(1, center + margin), 8),
  ]
}

function nearlyEqual(left: number, right: number, tolerance = 1e-12): boolean {
  return Math.abs(left - right) <= tolerance
}

function validateOutcome(value: unknown, path: string): boolean {
  const outcome = requireObject(value, path)
  const solved = requireBoolean(outcome.solved, `${path}.solved`)
  if (outcome.status !== (solved ? 'solved' : 'failed')) {
    throw contractError(`${path}.status disagrees with the official solved flag.`)
  }
  requireHash(outcome.definition_sha256, `${path}.definition_sha256`)
  requireVector(outcome.recovery_targets, 5, `${path}.recovery_targets`)
  requireVector(outcome.tail_minimum_services, 5, `${path}.tail_minimum_services`)
  if (!isBooleanArray(outcome.target_met_by_service, 5)) {
    throw contractError(`${path}.target_met_by_service must contain five booleans.`)
  }
  if (outcome.assessment_tail_days !== 3) {
    throw contractError(`${path}.assessment_tail_days must be 3.`)
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
  ]) requireVector(day[field], 5, `${path}.${field}`)
  requireVector(day.raw_action, 22, `${path}.raw_action`)
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
  requireVector(shock.impact, 5, `${path}.shock.impact`)
  requireVector(shock.public_risk_next, 5, `${path}.shock.public_risk_next`)
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
  ]) requireVector(logistics[field], 5, `${path}.logistics.${field}`)
  requireNumber(logistics.road_capacity, `${path}.logistics.road_capacity`)
  if (!Array.isArray(logistics.mutual_aid_transfers)) {
    throw contractError(`${path}.logistics.mutual_aid_transfers must be an array.`)
  }
}

function validatePlanner(value: unknown, path: string): boolean {
  const planner = requireObject(value, path)
  const trajectory = planner.trajectory
  if (!Array.isArray(trajectory) || trajectory.length !== 30) {
    throw contractError(`${path}.trajectory must contain exactly 30 days.`)
  }
  trajectory.forEach((day, index) => validateDay(day, index, path))
  for (const field of [
    'rauc', 'final_resilience', 'minimum_resilience',
    'post_shock_recovery_shortfall_auc', 'days_to_pre_shock_recovery_after_largest_loss',
    'largest_shock_loss_day', 'critical_service_days', 'hard_violation_count',
    'constraint_violations', 'max_logistics_conservation_residual',
  ]) requireNumber(planner[field], `${path}.${field}`)
  requireVector(planner.final_depot_stock, 5, `${path}.final_depot_stock`)
  requireVector(planner.final_pending_arrivals, 5, `${path}.final_pending_arrivals`)
  requireHash(planner.trajectory_sha256, `${path}.trajectory_sha256`)
  const solved = validateOutcome(planner.absolute_outcome, `${path}.absolute_outcome`)
  const terminal = requireObject(trajectory[29], `${path}.trajectory[29]`)
  const terminalSolved = validateOutcome(
    terminal.absolute_outcome,
    `${path}.trajectory[29].absolute_outcome`,
  )
  if (terminalSolved !== solved) {
    throw contractError(`${path} terminal and summary outcomes disagree.`)
  }
  return solved
}

/** Runtime validation prevents incompatible or partial evidence from being rendered as V3. */
export function parseComparison(value: unknown): CompareResponse {
  const root = requireObject(value, 'comparison response')
  if (root.schema_version !== '4.0.0' || root.engine_version !== 'city-recovery-env-v3') {
    throw contractError('The response is not a CityRecoveryEnv-v3 schema 4 result.')
  }
  const environment = requireObject(root.environment, 'environment')
  if (
    environment.id !== 'CityRecoveryEnv-v3'
    || environment.observation_count !== 73
    || environment.action_count !== 22
  ) throw contractError('The environment interface is not the frozen 73-input / 22-action V3 contract.')
  const scenario = requireObject(root.scenario, 'scenario')
  if (scenario.horizon_days !== 30 || scenario.assessment_tail_days !== 3) {
    throw contractError('The scenario is not the frozen 30-day / 3-day-tail V3 protocol.')
  }
  requireVector(scenario.initial_services, 5, 'scenario.initial_services')
  requireVector(scenario.priorities, 5, 'scenario.priorities')
  requireVector(scenario.recovery_targets, 5, 'scenario.recovery_targets')
  if (!isStringArray(root.observation_order, 73) || !isStringArray(root.action_order, 22)) {
    throw contractError('The returned policy channel order is incomplete.')
  }
  if (!Array.isArray(root.services) || root.services.join('|') !== services.join('|')) {
    throw contractError('The returned service order does not match the V3 contract.')
  }
  if (!Array.isArray(root.shock_schedule) || root.shock_schedule.length !== 30) {
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
  const policy = requireObject(root.policy, 'policy')
  requireString(policy.id, 'policy.id')
  requireHash(policy.sha256, 'policy.sha256')
  requireHash(policy.sb3_checkpoint_sha256, 'policy.sb3_checkpoint_sha256')
  requireHash(policy.manifest_sha256, 'policy.manifest_sha256')
  const baselineSpec = requireObject(root.baseline_spec, 'baseline_spec')
  requireString(baselineSpec.id, 'baseline_spec.id')
  requireString(baselineSpec.version, 'baseline_spec.version')
  requireHash(root.result_id, 'result_id')
  requireHash(root.shock_schedule_sha256, 'shock_schedule_sha256')
  requireHash(root.engine_spec_sha256, 'engine_spec_sha256')
  requireHash(root.outcome_definition_sha256, 'outcome_definition_sha256')
  return value as CompareResponse
}

function validateBenchmarkPlanner(
  value: unknown,
  path: string,
  expectedId: string,
  expectedVersion?: string,
): {
    solved: number
    meanRauc: number
  } {
  const planner = requireObject(value, path)
  requireExactKeys(
    planner,
    expectedVersion === undefined
      ? ['id', 'solved_count', 'failed_count', 'solve_rate', 'solve_rate_wilson_ci95', 'mean_resilience_auc']
      : ['id', 'version', 'solved_count', 'failed_count', 'solve_rate', 'solve_rate_wilson_ci95', 'mean_resilience_auc'],
    path,
  )
  if (planner.id !== expectedId || (expectedVersion !== undefined && planner.version !== expectedVersion)) {
    throw contractError(`${path} identity does not match the sealed V3 contract.`)
  }
  const solved = requireInteger(planner.solved_count, `${path}.solved_count`, 0, 40)
  const failed = requireInteger(planner.failed_count, `${path}.failed_count`, 0, 40)
  const rate = requireNumber(planner.solve_rate, `${path}.solve_rate`)
  if (failed !== 40 - solved || !nearlyEqual(rate, solved / 40)) {
    throw contractError(`${path} solve count, failure count, and rate disagree.`)
  }
  const interval = requireVector(planner.solve_rate_wilson_ci95, 2, `${path}.solve_rate_wilson_ci95`)
  const expectedInterval = wilsonInterval(solved, 40)
  if (
    interval[0] < 0
    || interval[1] > 1
    || interval[0] > interval[1]
    || !nearlyEqual(interval[0], expectedInterval[0])
    || !nearlyEqual(interval[1], expectedInterval[1])
  ) throw contractError(`${path}.solve_rate_wilson_ci95 does not match the solved count.`)
  const meanRauc = requireNumber(planner.mean_resilience_auc, `${path}.mean_resilience_auc`)
  if (meanRauc < 0 || meanRauc > 1) {
    throw contractError(`${path}.mean_resilience_auc must be in [0,1].`)
  }
  return { solved, meanRauc }
}

function parseBenchmark(value: unknown): MetadataV3['benchmark'] {
  const benchmark = requireObject(value, 'benchmark')
  requireExactKeys(benchmark, [
    'artifact_sha256', 'artifact_type', 'status', 'primary_metric', 'synthetic_case_count',
    'candidate', 'baseline', 'paired_absolute_outcomes',
    'secondary_head_to_head_resilience_auc', 'invariants', 'rows_sha256', 'ledger_sha256',
  ], 'benchmark')
  if (
    benchmark.artifact_type !== 'city_recovery_v3_final_benchmark'
    || benchmark.status !== 'complete'
    || benchmark.primary_metric !== 'independent_absolute_disasters_solved'
    || benchmark.synthetic_case_count !== 40
  ) throw contractError('The final benchmark identity, status, metric, or 40-case scope is invalid.')
  const candidate = validateBenchmarkPlanner(
    benchmark.candidate,
    'benchmark.candidate',
    'city-recovery-sb3-ppo-v3-selected',
  )
  const baseline = validateBenchmarkPlanner(
    benchmark.baseline,
    'benchmark.baseline',
    'reactive-public-state-heuristic-v3',
    '3.0.0',
  )
  const pairs = requireObject(benchmark.paired_absolute_outcomes, 'benchmark.paired_absolute_outcomes')
  const pairKeys = ['both_solved', 'ppo_only', 'heuristic_only', 'neither'] as const
  requireExactKeys(pairs, pairKeys, 'benchmark.paired_absolute_outcomes')
  const pairCounts = pairKeys.map((key) => (
    requireInteger(pairs[key], `benchmark.paired_absolute_outcomes.${key}`, 0, 40)
  ))
  if (pairCounts.reduce((sum, count) => sum + count, 0) !== 40) {
    throw contractError('The paired benchmark counts do not cover exactly 40 cases.')
  }
  if (
    candidate.solved !== Number(pairs.both_solved) + Number(pairs.ppo_only)
    || baseline.solved !== Number(pairs.both_solved) + Number(pairs.heuristic_only)
  ) throw contractError('The independent solved counts disagree with the paired outcome cells.')

  const headToHead = requireObject(
    benchmark.secondary_head_to_head_resilience_auc,
    'benchmark.secondary_head_to_head_resilience_auc',
  )
  const headKeys = ['candidate_wins', 'baseline_wins', 'ties'] as const
  requireExactKeys(
    headToHead,
    [...headKeys, 'candidate_mean_minus_baseline_mean'],
    'benchmark.secondary_head_to_head_resilience_auc',
  )
  const headCounts = headKeys.map((key) => requireInteger(
    headToHead[key],
    `benchmark.secondary_head_to_head_resilience_auc.${key}`,
    0,
    40,
  ))
  if (headCounts.reduce((sum, count) => sum + count, 0) !== 40) {
    throw contractError('The secondary head-to-head counts do not cover exactly 40 cases.')
  }
  const delta = requireNumber(
    headToHead.candidate_mean_minus_baseline_mean,
    'benchmark.secondary_head_to_head_resilience_auc.candidate_mean_minus_baseline_mean',
  )
  if (!nearlyEqual(delta, rounded(candidate.meanRauc - baseline.meanRauc, 10), 1.1e-10)) {
    throw contractError('The secondary mean resilience-AUC delta disagrees with the planner means.')
  }

  const invariants = requireObject(benchmark.invariants, 'benchmark.invariants')
  requireExactKeys(invariants, [
    'all_rows_present', 'unique_rows', 'same_tapes', 'deterministic_replay_mismatches',
    'candidate_hard_violations', 'baseline_hard_violations', 'maximum_conservation_residual',
  ], 'benchmark.invariants')
  const residual = requireNumber(
    invariants.maximum_conservation_residual,
    'benchmark.invariants.maximum_conservation_residual',
  )
  if (
    invariants.all_rows_present !== true
    || invariants.unique_rows !== true
    || invariants.same_tapes !== true
    || invariants.deterministic_replay_mismatches !== 0
    || invariants.candidate_hard_violations !== 0
    || invariants.baseline_hard_violations !== 0
    || residual < 0
    || residual > 1e-6
  ) throw contractError('The final benchmark invariants did not pass.')
  requireHash(benchmark.artifact_sha256, 'benchmark.artifact_sha256')
  requireHash(benchmark.rows_sha256, 'benchmark.rows_sha256')
  requireHash(benchmark.ledger_sha256, 'benchmark.ledger_sha256')
  return value as MetadataV3['benchmark']
}

export function parseMetadata(value: unknown): MetadataV3 {
  const root = requireObject(value, 'metadata')
  const model = requireObject(root.model, 'metadata.model')
  const environment = requireObject(root.environment, 'metadata.environment')
  if (
    root.schema_version !== '4.0.0'
    || model.id !== 'city-recovery-sb3-ppo-v3-selected'
    || model.version !== '3.0.0'
    || model.schema_version !== '4.0.0'
    || model.manifest_schema_version !== 1
    || model.artifact_type !== 'stable_baselines3_ppo'
    || model.algorithm !== 'PPO'
    || model.training_library !== 'stable-baselines3'
    || environment.id !== 'CityRecoveryEnv-v3'
    || environment.version !== '3.0.0'
    || environment.observation_count !== 73
    || environment.action_count !== 22
    || environment.policy_neutral_transition !== true
    || environment.future_tape_visible !== false
    || model.observation_count !== 73
    || model.action_count !== 22
  ) throw contractError('Metadata does not describe the canonical selected V3 release.')
  requireExactStringArray(model.observation_order, observationOrderV3, 'metadata.model.observation_order')
  requireExactStringArray(model.action_order, actionOrderV3, 'metadata.model.action_order')
  if (!isStringArray(root.services, 5) || root.services.join('|') !== services.join('|')) {
    throw contractError('Metadata service order does not match the V3 contract.')
  }
  requireExactStringArray(model.action_groups, actionGroupsV3, 'metadata.model.action_groups')
  const completedTransitions = requireInteger(
    model.completed_training_transitions,
    'metadata.model.completed_training_transitions',
    92_160,
    645_120,
  )
  if (
    model.trainable_parameters !== 322_733
    || model.requested_training_transitions !== 645_120
    || ![92_160, 184_320, 276_480, 368_640, 460_800, 552_960, 645_120]
      .includes(completedTransitions)
  ) throw contractError('Metadata model parameter or transition counts drifted from the sealed V3 protocol.')
  for (const field of [
    'onnx_sha256', 'selected_checkpoint_sha256', 'manifest_sha256',
    'parity_sha256', 'scientific_source_sha256',
  ]) requireHash(model[field], `metadata.model.${field}`)
  requireHash(environment.spec_sha256, 'metadata.environment.spec_sha256')
  requireHash(environment.outcome_definition_sha256, 'metadata.environment.outcome_definition_sha256')
  const baseline = requireObject(root.baseline, 'metadata.baseline')
  if (
    baseline.id !== 'reactive-public-state-heuristic-v3'
    || baseline.version !== '3.0.0'
    || baseline.uses_same_observation_contract !== true
    || baseline.uses_same_action_contract !== true
    || baseline.uses_public_risk_signal !== true
    || baseline.future_tape_visible !== false
  ) throw contractError('Metadata baseline identity or public-causal contract drifted.')
  parseBenchmark(root.benchmark)
  return value as MetadataV3
}

function parseSavedSummaries(value: unknown): SavedResultSummary[] {
  const root = requireObject(value, 'saved simulations')
  if (!Array.isArray(root.results)) throw contractError('Saved simulation index is malformed.')
  return root.results.map((entry, index) => {
    const row = requireObject(entry, `saved simulations[${index}]`)
    if (row.engine_version !== 'city-recovery-env-v3' || row.horizon_days !== 30) {
      throw contractError(`saved simulations[${index}] is not a V3 run.`)
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

export async function fetchMetadata(signal?: AbortSignal): Promise<MetadataV3> {
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
