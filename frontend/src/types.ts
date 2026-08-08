export type MetricPolarity = 'positive' | 'negative' | 'neutral'

export type EvidenceMetric = {
  id: string
  label: string
  value: number | string | boolean | null
  display: string
  polarity: MetricPolarity
  note: string
}

export type ProvenanceItem = {
  label: string
  source_repository: string
  path: string
  sha256: string
}

export type ModelTrack = {
  id: string
  name: string
  role: string
  status: string
  evidence_class: string
  claim_eligible: boolean
  architecture: {
    family: string
    parameters: number | null
    inputs: string
    outputs: string
    runtime: string
  }
  training: {
    started: boolean
    transitions: number
    unit: string
    seed_count: number
    hardware: string
    note: string
  }
  evaluation: {
    headline: string
    metrics: EvidenceMetric[]
  }
  safety: {
    hard_violations: number | null
    resource_violations: number | null
    crew_violations: number | null
    replay_verified: boolean | null
  }
  limitations: string[]
  provenance: ProvenanceItem[]
}

export type PipelineStage = {
  id: string
  label: string
  detail: string
}

export type BenchmarkDirection = 'higher_is_better' | 'lower_is_better'

export type UnmeasuredWorkbenchBenchmark = {
  status: 'not_yet_run'
  note: string
}

export type MeasuredWorkbenchBenchmark = {
  status: 'measured'
  benchmark_id: string
  name: string
  evidence_class: string
  model_track_id: string
  scenario_total: number
  objective: {
    label: string
    definition: string
    success_threshold: number
    learned_policy: {
      label: string
      passes: number
      misses: number
    }
    static_heuristic: {
      label: string
      passes: number
      misses: number
    }
    counts_are_independent_not_complementary: true
  }
  head_to_head: {
    metric: {
      id: string
      label: string
      direction: BenchmarkDirection
      tie_rule: string
    }
    learned_wins: number
    heuristic_wins: number
    ties: number
    learned_mean: number
    heuristic_mean: number
    paired_mean_difference: number
    paired_bootstrap_ci95: [number, number]
  }
  secondary: {
    metric: {
      id: string
      label: string
      direction: BenchmarkDirection
    }
    learned_mean: number
    heuristic_mean: number
  }
  synthetic_disclosure: string
  limitations: string[]
  note: string
  provenance: ProvenanceItem[]
}

export type WorkbenchBenchmark = UnmeasuredWorkbenchBenchmark | MeasuredWorkbenchBenchmark

export type WorkbenchOverview = {
  schema_version: 'model-workbench-v1'
  project: {
    name: string
    summary: string
    metric_note: string
  }
  tracks: ModelTrack[]
  pipeline: PipelineStage[]
  benchmark: WorkbenchBenchmark
}

export type ToolboxStructuredInput = {
  public_forecast_signal: [number, number, number, number, number]
  visible_service_need: [number, number, number, number, number]
  public_regime: number
  current_service_health: [number, number, number, number, number]
  phase_window: number
}

export type ToolboxEvaluationRequest = ToolboxStructuredInput

export type ToolboxEvaluationResponse = {
  schema_version: 'model-toolbox-evaluation-v1'
  benchmark_id: string
  synthetic_disclosure: string
  input: {
    structured: ToolboxStructuredInput
    feature_order: string[]
    vector: number[]
    normalization: {
      method: 'embedded_z_score'
      input_vector_is_raw: true
      normalization_executed_inside_model: true
      mean: number[]
      scale: number[]
    }
  }
  model: {
    id: string
    parameter_count: number
    action_index: number
    action_label: string
    logits: number[]
    probabilities: number[]
    confidence: number
    probability_margin: number
  }
  heuristic: {
    id: 'static-visible-need-heuristic-v1'
    rule: 'argmax_visible_service_need'
    action_index: number
    action_label: string
    scores: number[]
  }
  comparison: {
    same_action: boolean
  }
  benchmark_summary: {
    scenario_total: number
    objective_passes: {
      model: number
      heuristic: number
    }
    head_to_head: {
      model_wins: number
      heuristic_wins: number
      ties: number
    }
  }
  runtime: {
    engine: 'onnxruntime'
    execution_provider: 'CPUExecutionProvider'
    onnx_sha256: string
    real_model_inference: true
  }
}

export type ToolboxValidationDetail = {
  field: string
  message: string
  type: string
}
