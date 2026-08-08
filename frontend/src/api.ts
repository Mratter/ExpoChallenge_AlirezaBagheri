import type {
  MeasuredWorkbenchBenchmark,
  ProvenanceItem,
  ToolboxEvaluationRequest,
  ToolboxEvaluationResponse,
  ToolboxStructuredInput,
  ToolboxValidationDetail,
  WorkbenchBenchmark,
  WorkbenchOverview,
} from './types'

const SHA256_PATTERN = /^[a-f0-9]{64}$/
const REQUIRED_SYNTHETIC_DISCLOSURE = 'Engineered synthetic benchmark of learnable observable patterns; not real-world validation.'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0
}

function isNumberArray(value: unknown, length: number): value is number[] {
  return Array.isArray(value) && value.length === length && value.every(isFiniteNumber)
}

function isBoundedNumberArray(value: unknown, length: number, minimum: number, maximum: number): value is number[] {
  return isNumberArray(value, length) && value.every((item) => minimum <= item && item <= maximum)
}

function isToolboxStructuredInput(value: unknown): value is ToolboxStructuredInput {
  if (!isRecord(value)) return false
  return isBoundedNumberArray(value.public_forecast_signal, 5, -8, 8)
    && isBoundedNumberArray(value.visible_service_need, 5, 0, 1.5)
    && isNonNegativeInteger(value.public_regime)
    && value.public_regime <= 3
    && isBoundedNumberArray(value.current_service_health, 5, 0, 1)
    && isNonNegativeInteger(value.phase_window)
    && value.phase_window >= 1
    && value.phase_window <= 12
}

function isDirection(value: unknown): value is 'higher_is_better' | 'lower_is_better' {
  return value === 'higher_is_better' || value === 'lower_is_better'
}

function isProvenanceItem(value: unknown): value is ProvenanceItem {
  if (!isRecord(value)) return false
  return isNonEmptyString(value.label)
    && isNonEmptyString(value.source_repository)
    && isNonEmptyString(value.path)
    && typeof value.sha256 === 'string'
    && SHA256_PATTERN.test(value.sha256)
}

function isMeasuredBenchmark(value: Record<string, unknown>, trackIds: Set<string>): value is unknown & MeasuredWorkbenchBenchmark {
  if (value.status !== 'measured'
    || !isNonEmptyString(value.benchmark_id)
    || !isNonEmptyString(value.name)
    || value.evidence_class !== 'sealed_synthetic_pattern_learning_showcase'
    || !isNonEmptyString(value.model_track_id)
    || !trackIds.has(value.model_track_id)
    || !isNonNegativeInteger(value.scenario_total)
    || value.scenario_total <= 0
    || value.synthetic_disclosure !== REQUIRED_SYNTHETIC_DISCLOSURE
    || !isNonEmptyString(value.note)
    || !Array.isArray(value.limitations)
    || value.limitations.length === 0
    || !value.limitations.every(isNonEmptyString)
    || !Array.isArray(value.provenance)
    || value.provenance.length === 0
    || !value.provenance.every(isProvenanceItem)
    || !isRecord(value.objective)
    || !isRecord(value.head_to_head)
    || !isRecord(value.secondary)) return false

  const total = value.scenario_total
  const objective = value.objective
  if (!isNonEmptyString(objective.label)
    || !isNonEmptyString(objective.definition)
    || !isNonNegativeInteger(objective.success_threshold)
    || objective.counts_are_independent_not_complementary !== true
    || !isRecord(objective.learned_policy)
    || !isRecord(objective.static_heuristic)) return false

  const learnedObjective = objective.learned_policy
  const heuristicObjective = objective.static_heuristic
  if (!isNonEmptyString(learnedObjective.label)
    || !isNonNegativeInteger(learnedObjective.passes)
    || !isNonNegativeInteger(learnedObjective.misses)
    || learnedObjective.passes + learnedObjective.misses !== total
    || !isNonEmptyString(heuristicObjective.label)
    || !isNonNegativeInteger(heuristicObjective.passes)
    || !isNonNegativeInteger(heuristicObjective.misses)
    || heuristicObjective.passes + heuristicObjective.misses !== total) return false

  const headToHead = value.head_to_head
  if (!isRecord(headToHead.metric)
    || !isNonEmptyString(headToHead.metric.id)
    || !isNonEmptyString(headToHead.metric.label)
    || !isDirection(headToHead.metric.direction)
    || !isNonEmptyString(headToHead.metric.tie_rule)
    || !isNonNegativeInteger(headToHead.learned_wins)
    || !isNonNegativeInteger(headToHead.heuristic_wins)
    || !isNonNegativeInteger(headToHead.ties)
    || headToHead.learned_wins + headToHead.heuristic_wins + headToHead.ties !== total
    || !isFiniteNumber(headToHead.learned_mean)
    || !isFiniteNumber(headToHead.heuristic_mean)
    || !isFiniteNumber(headToHead.paired_mean_difference)
    || !Array.isArray(headToHead.paired_bootstrap_ci95)
    || headToHead.paired_bootstrap_ci95.length !== 2
    || !headToHead.paired_bootstrap_ci95.every(isFiniteNumber)
    || headToHead.paired_bootstrap_ci95[0] > headToHead.paired_bootstrap_ci95[1]) return false

  const secondary = value.secondary
  return isRecord(secondary.metric)
    && isNonEmptyString(secondary.metric.id)
    && isNonEmptyString(secondary.metric.label)
    && isDirection(secondary.metric.direction)
    && isFiniteNumber(secondary.learned_mean)
    && isFiniteNumber(secondary.heuristic_mean)
}

function isWorkbenchBenchmark(value: unknown, trackIds: Set<string>): value is WorkbenchBenchmark {
  if (!isRecord(value)) return false
  if (value.status === 'not_yet_run') return isNonEmptyString(value.note)
  return isMeasuredBenchmark(value, trackIds)
}

export class OverviewError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'OverviewError'
  }
}

export function isWorkbenchOverview(value: unknown): value is WorkbenchOverview {
  if (!isRecord(value)) return false
  const candidate = value as Partial<WorkbenchOverview>
  if (candidate.schema_version !== 'model-workbench-v1'
    || typeof candidate.project?.name !== 'string'
  ) return false
  if (!Array.isArray(candidate.tracks)
    || candidate.tracks.length === 0
    || !candidate.tracks.every((track) => (
      typeof track?.id === 'string'
      && typeof track?.training?.started === 'boolean'
      && isNonNegativeInteger(track.training.transitions)
      && typeof track?.training?.unit === 'string'
      && track.training.unit.trim().length > 0
      && Array.isArray(track?.evaluation?.metrics)
    ))) return false
  const trackIds = new Set(candidate.tracks.map((track) => track.id))
  return Array.isArray(candidate.pipeline)
    && isWorkbenchBenchmark(candidate.benchmark, trackIds)
}

export async function loadWorkbenchOverview(signal?: AbortSignal): Promise<WorkbenchOverview> {
  const response = await fetch('/api/workbench/v1/overview', { signal })
  if (!response.ok) throw new OverviewError(`Evidence API returned ${response.status}.`)
  const payload: unknown = await response.json()
  if (!isWorkbenchOverview(payload)) throw new OverviewError('Evidence API returned an incompatible document.')
  return payload
}

export function isToolboxEvaluationResponse(value: unknown): value is ToolboxEvaluationResponse {
  if (!isRecord(value)
    || value.schema_version !== 'model-toolbox-evaluation-v1'
    || !isNonEmptyString(value.benchmark_id)
    || !isNonEmptyString(value.synthetic_disclosure)
    || !isRecord(value.input)
    || !isRecord(value.model)
    || !isRecord(value.heuristic)
    || !isRecord(value.comparison)
    || !isRecord(value.benchmark_summary)
    || !isRecord(value.runtime)) return false

  const input = value.input
  if (!isToolboxStructuredInput(input.structured)
    || !Array.isArray(input.feature_order)
    || input.feature_order.length !== 21
    || !input.feature_order.every(isNonEmptyString)
    || !isNumberArray(input.vector, 21)
    || !isRecord(input.normalization)) return false
  const normalization = input.normalization
  if (normalization.method !== 'embedded_z_score'
    || normalization.input_vector_is_raw !== true
    || normalization.normalization_executed_inside_model !== true
    || !isNumberArray(normalization.mean, 21)
    || !isNumberArray(normalization.scale, 21)
    || normalization.scale.some((item) => item <= 0)) return false

  const model = value.model
  if (!isNonEmptyString(model.id)
    || !isNonNegativeInteger(model.parameter_count)
    || model.parameter_count <= 0
    || !isNonNegativeInteger(model.action_index)
    || model.action_index > 4
    || !isNonEmptyString(model.action_label)
    || !isNumberArray(model.logits, 5)
    || !isBoundedNumberArray(model.probabilities, 5, 0, 1)
    || !isFiniteNumber(model.confidence)
    || model.confidence < 0
    || model.confidence > 1
    || !isFiniteNumber(model.probability_margin)
    || model.probability_margin < 0
    || model.probability_margin > 1) return false
  const probabilityTotal = model.probabilities.reduce((total, probability) => total + probability, 0)
  if (Math.abs(probabilityTotal - 1) > 1e-5
    || Math.abs(model.confidence - model.probabilities[model.action_index]) > 1e-6) return false

  const heuristic = value.heuristic
  if (heuristic.id !== 'static-visible-need-heuristic-v1'
    || heuristic.rule !== 'argmax_visible_service_need'
    || !isNonNegativeInteger(heuristic.action_index)
    || heuristic.action_index > 4
    || !isNonEmptyString(heuristic.action_label)
    || !isNumberArray(heuristic.scores, 5)
    || typeof value.comparison.same_action !== 'boolean') return false

  const summary = value.benchmark_summary
  if (!isNonNegativeInteger(summary.scenario_total)
    || summary.scenario_total <= 0
    || !isRecord(summary.objective_passes)
    || !isNonNegativeInteger(summary.objective_passes.model)
    || !isNonNegativeInteger(summary.objective_passes.heuristic)
    || summary.objective_passes.model > summary.scenario_total
    || summary.objective_passes.heuristic > summary.scenario_total
    || !isRecord(summary.head_to_head)
    || !isNonNegativeInteger(summary.head_to_head.model_wins)
    || !isNonNegativeInteger(summary.head_to_head.heuristic_wins)
    || !isNonNegativeInteger(summary.head_to_head.ties)
    || summary.head_to_head.model_wins + summary.head_to_head.heuristic_wins + summary.head_to_head.ties !== summary.scenario_total) return false

  return value.runtime.engine === 'onnxruntime'
    && value.runtime.execution_provider === 'CPUExecutionProvider'
    && typeof value.runtime.onnx_sha256 === 'string'
    && SHA256_PATTERN.test(value.runtime.onnx_sha256)
    && value.runtime.real_model_inference === true
}

function validationDetails(value: unknown): ToolboxValidationDetail[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!isRecord(item)
      || !isNonEmptyString(item.field)
      || !isNonEmptyString(item.message)
      || !isNonEmptyString(item.type)) return []
    return [{ field: item.field, message: item.message, type: item.type }]
  })
}

export class ToolboxEvaluationError extends Error {
  readonly status: number
  readonly details: ToolboxValidationDetail[]

  constructor(message: string, status: number, details: ToolboxValidationDetail[] = []) {
    super(message)
    this.name = 'ToolboxEvaluationError'
    this.status = status
    this.details = details
  }
}

export async function evaluateToolbox(
  request: ToolboxEvaluationRequest,
  signal?: AbortSignal,
): Promise<ToolboxEvaluationResponse> {
  const response = await fetch('/api/workbench/v1/toolbox/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  })
  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new ToolboxEvaluationError(`Model toolbox returned ${response.status} without JSON.`, response.status)
  }
  if (!response.ok) {
    const error = isRecord(payload) && isRecord(payload.error) ? payload.error : undefined
    const message = error && isNonEmptyString(error.message)
      ? error.message
      : `Model toolbox returned ${response.status}.`
    throw new ToolboxEvaluationError(message, response.status, validationDetails(error?.details))
  }
  if (!isToolboxEvaluationResponse(payload)) {
    throw new ToolboxEvaluationError('Model toolbox returned an incompatible document.', response.status)
  }
  return payload
}
