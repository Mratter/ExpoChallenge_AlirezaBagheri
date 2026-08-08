import type {
  MeasuredWorkbenchBenchmark,
  ToolboxEvaluationResponse,
  ToolboxStructuredInput,
} from '../src/types'

export const toolboxInputFixture: ToolboxStructuredInput = {
  public_forecast_signal: [1.25, 0.1, -0.45, 0.35, -0.2],
  visible_service_need: [0.25, 1.1, 0.2, 0.15, 0.1],
  public_regime: 2,
  current_service_health: [0.68, 0.71, 0.64, 0.73, 0.69],
  phase_window: 1,
}

export function toolboxResponseFixture(): ToolboxEvaluationResponse {
  return {
    schema_version: 'model-toolbox-evaluation-v1',
    benchmark_id: 'adaptive-cascades-showcase-v2',
    synthetic_disclosure: 'Engineered synthetic benchmark of learnable observable patterns; not real-world validation.',
    input: {
      structured: {
        ...toolboxInputFixture,
        public_forecast_signal: [...toolboxInputFixture.public_forecast_signal],
        visible_service_need: [...toolboxInputFixture.visible_service_need],
        current_service_health: [...toolboxInputFixture.current_service_health],
      },
      feature_order: [
        'public_forecast_signal_transport',
        'public_forecast_signal_housing',
        'public_forecast_signal_food',
        'public_forecast_signal_healthcare',
        'public_forecast_signal_public_services',
        'visible_service_need_transport',
        'visible_service_need_housing',
        'visible_service_need_food',
        'visible_service_need_healthcare',
        'visible_service_need_public_services',
        'public_regime_0',
        'public_regime_1',
        'public_regime_2',
        'public_regime_3',
        'current_service_health_transport',
        'current_service_health_housing',
        'current_service_health_food',
        'current_service_health_healthcare',
        'current_service_health_public_services',
        'phase_sin',
        'phase_cos',
      ],
      vector: [
        1.25, 0.1, -0.45, 0.35, -0.2,
        0.25, 1.1, 0.2, 0.15, 0.1,
        0, 0, 1, 0,
        0.68, 0.71, 0.64, 0.73, 0.69,
        0, 1,
      ],
      normalization: {
        method: 'embedded_z_score',
        input_vector_is_raw: true,
        normalization_executed_inside_model: true,
        mean: Array<number>(21).fill(0),
        scale: Array<number>(21).fill(1),
      },
    },
    model: {
      id: 'adaptive-cascade-mlp-v2-300k',
      parameter_count: 300113,
      action_index: 3,
      action_label: 'healthcare',
      logits: [-1, 0.2, 0.5, 2, 0.1],
      probabilities: [0.03, 0.08, 0.12, 0.69, 0.08],
      confidence: 0.69,
      probability_margin: 0.57,
    },
    heuristic: {
      id: 'static-visible-need-heuristic-v1',
      rule: 'argmax_visible_service_need',
      action_index: 1,
      action_label: 'housing',
      scores: [0.25, 1.1, 0.2, 0.15, 0.1],
    },
    comparison: { same_action: false },
    benchmark_summary: {
      scenario_total: 40,
      objective_passes: { model: 38, heuristic: 20 },
      head_to_head: { model_wins: 38, heuristic_wins: 0, ties: 2 },
    },
    runtime: {
      engine: 'onnxruntime',
      execution_provider: 'CPUExecutionProvider',
      onnx_sha256: 'b3edf8007feb749ddc33fc3ebbb008a02ef98d561bd74cfde286dde030a4dae0',
      real_model_inference: true,
    },
  }
}

export function toolboxBenchmarkFixture(): MeasuredWorkbenchBenchmark {
  return {
    status: 'measured',
    benchmark_id: 'adaptive-cascades-showcase-v2',
    name: 'Adaptive Cascades Synthetic Showcase v2',
    evidence_class: 'sealed_synthetic_pattern_learning_showcase',
    model_track_id: 'showcase-adaptive-v2',
    scenario_total: 40,
    objective: {
      label: 'Contain at least 10 of 12 cascade windows',
      definition: 'A scenario passes when at least 10 of 12 cascade windows are contained.',
      success_threshold: 10,
      learned_policy: { label: 'Adaptive Cascade MLP v2', passes: 38, misses: 2 },
      static_heuristic: { label: 'Static visible-need heuristic', passes: 20, misses: 20 },
      counts_are_independent_not_complementary: true,
    },
    head_to_head: {
      metric: {
        id: 'cascade_windows_contained',
        label: 'Cascade windows contained per scenario',
        direction: 'higher_is_better',
        tie_rule: 'exact equality',
      },
      learned_wins: 38,
      heuristic_wins: 0,
      ties: 2,
      learned_mean: 11.5,
      heuristic_mean: 9.25,
      paired_mean_difference: 2.25,
      paired_bootstrap_ci95: [1.775, 2.75],
    },
    secondary: {
      metric: { id: 'critical_service_deficit_auc', label: 'Critical service deficit AUC', direction: 'lower_is_better' },
      learned_mean: 0.303831090705,
      heuristic_mean: 0.33655319428,
    },
    synthetic_disclosure: 'Engineered synthetic benchmark of learnable observable patterns; not real-world validation.',
    limitations: ['Purpose-built artificial patterns.'],
    note: 'Independent objective counts and matched head-to-head results.',
    provenance: [{ label: 'Result', source_repository: 'test', path: 'result.json', sha256: 'a'.repeat(64) }],
  }
}
