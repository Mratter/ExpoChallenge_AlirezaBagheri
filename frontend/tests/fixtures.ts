import type { ModelTrack, WorkbenchOverview } from '../src/types'

function track(overrides: Partial<ModelTrack> & Pick<ModelTrack, 'id' | 'name'>): ModelTrack {
  return {
    id: overrides.id,
    name: overrides.name,
    role: overrides.role ?? 'Research track',
    status: overrides.status ?? 'trained_evaluated',
    evidence_class: overrides.evidence_class ?? 'synthetic evidence',
    claim_eligible: overrides.claim_eligible ?? false,
    architecture: overrides.architecture ?? {
      family: 'Test architecture', parameters: 100, inputs: 'Input state', outputs: 'Action', runtime: 'Test runtime',
    },
    training: overrides.training ?? {
      started: true, transitions: 100, unit: 'transitions', seed_count: 1, hardware: 'CPU', note: 'Test training receipt.',
    },
    evaluation: overrides.evaluation ?? { headline: 'Measured test result.', metrics: [] },
    safety: overrides.safety ?? { hard_violations: 0, resource_violations: 0, crew_violations: 0, replay_verified: true },
    limitations: overrides.limitations ?? ['Synthetic only.'],
    provenance: overrides.provenance ?? [{ label: 'Evidence', source_repository: 'test', path: 'evidence.json', sha256: 'a'.repeat(64) }],
  }
}

export function overviewFixture(): WorkbenchOverview {
  return {
    schema_version: 'model-workbench-v1',
    project: {
      name: 'Autonomous City Recovery Planner',
      summary: 'A hybrid planning system for synthetic city recovery.',
      metric_note: 'Planning outcomes, not classification accuracy.',
    },
    tracks: [
      track({
        id: 'production-v2',
        name: 'Production demo v2',
        role: 'Current trained demonstration model',
        claim_eligible: true,
        training: { started: true, transitions: 30000, unit: 'transitions', seed_count: 1, hardware: 'CPU', note: '32 training scenario units.' },
        evaluation: {
          headline: 'Won 32 of 40 held-out synthetic scenarios on resilience AUC.',
          metrics: [
            { id: 'scenario_wins', label: 'Matched scenario wins', value: 32, display: '32 / 40', polarity: 'positive', note: 'Higher AUC.' },
            { id: 'scenario_total', label: 'Held-out scenarios', value: 40, display: '40', polarity: 'neutral', note: 'Fixed holdout.' },
            { id: 'baseline_wins', label: 'Baseline scenario wins', value: 8, display: '8 / 40', polarity: 'neutral', note: 'No ties.' },
            { id: 'candidate_rauc', label: 'AI resilience AUC', value: 0.4434, display: '0.4434', polarity: 'positive', note: 'Higher is better.' },
            { id: 'baseline_rauc', label: 'GLOP resilience AUC', value: 0.4349, display: '0.4349', polarity: 'neutral', note: 'Comparator.' },
            { id: 'relative_improvement_percent', label: 'Relative resilience gain', value: 1.96, display: '+1.96%', polarity: 'positive', note: 'Not accuracy.' },
            { id: 'deterministic_executions', label: 'Determinism executions', value: 200, display: '200', polarity: 'positive', note: 'Repeated.' },
          ],
        },
      }),
      track({ id: 'scientific-v4', name: 'Scientific relational GNN v4', status: 'trained_preliminary_no_claim' }),
      track({ id: 'pilot-r9', name: 'CUDA relational pilot R9', status: 'trained_no_go' }),
      track({
        id: 'architecture-r22-v10',
        name: 'Structured graph scheduler R22 / V10',
        role: 'Untrained next-generation architecture',
        status: 'untrained_terminal_no_go',
        architecture: { family: 'HGT + GRU', parameters: 3673671, inputs: 'Typed graph', outputs: 'Structured schedule', runtime: 'CUDA planned' },
        training: { started: false, transitions: 0, unit: 'transitions', seed_count: 0, hardware: 'CUDA planned', note: 'Training never started.' },
        evaluation: {
          headline: 'A privileged diagnostic failed the training gate.',
          metrics: [
            { id: 'diagnostic_reduction_percent', label: 'Reachability reduction', value: 7.34, display: '7.34%', polarity: 'negative', note: 'Not trained performance.' },
            { id: 'required_gate_percent', label: 'Required gate', value: 40, display: '40%', polarity: 'neutral', note: 'Training prerequisite.' },
          ],
        },
      }),
    ],
    pipeline: [
      { id: 'observe', label: 'Observe', detail: 'Build the public state.' },
      { id: 'propose', label: 'Propose', detail: 'The policy proposes priorities.' },
      { id: 'project', label: 'Make feasible', detail: 'The solver enforces exact constraints.' },
      { id: 'simulate', label: 'Simulate', detail: 'Advance a matched tape.' },
      { id: 'compare', label: 'Compare', detail: 'Measure both planners.' },
      { id: 'verify', label: 'Verify', detail: 'Bind hashes and replay.' },
    ],
    benchmark: { status: 'not_yet_run', note: 'The new pattern-learning benchmark has no measured result.' },
  }
}

export function measuredOverviewFixture(): WorkbenchOverview {
  const overview = overviewFixture()
  overview.tracks.splice(1, 0, track({
    id: 'showcase-adaptive-v2',
    name: 'Adaptive Cascade MLP v2 (300k)',
    role: 'Trained model for the sealed artificial pattern-learning showcase',
    evidence_class: 'sealed_synthetic_pattern_learning_showcase',
    claim_eligible: true,
    architecture: {
      family: 'Supervised multilayer perceptron with deterministic ONNX export',
      parameters: 300113,
      inputs: '21 public observable telemetry and recovery-state features',
      outputs: '5 recovery-action classes',
      runtime: 'ONNX Runtime CPUExecutionProvider',
    },
    training: {
      started: true,
      transitions: 9600,
      unit: 'labeled windows',
      seed_count: 1,
      hardware: 'CPU',
      note: '9,600 labeled windows from 800 synthetic training scenarios; 120 supervised epochs completed in 66.73 seconds.',
    },
    evaluation: { headline: 'Passed the registered objective in 38 of 40 sealed synthetic scenarios.', metrics: [] },
  }))
  overview.benchmark = {
    status: 'measured',
    benchmark_id: 'adaptive-cascades-showcase-v2',
    name: 'Adaptive Cascades Synthetic Showcase v2',
    evidence_class: 'sealed_synthetic_pattern_learning_showcase',
    model_track_id: 'showcase-adaptive-v2',
    scenario_total: 40,
    objective: {
      label: 'Contain at least 10 of 12 cascade windows',
      definition: 'A scenario passes when at least 10 of 12 hidden cascade windows are contained by the public action chosen for that window.',
      success_threshold: 10,
      learned_policy: { label: 'Adaptive Cascade MLP v2 (300k)', passes: 38, misses: 2 },
      static_heuristic: { label: 'Static visible-need heuristic', passes: 20, misses: 20 },
      counts_are_independent_not_complementary: true,
    },
    head_to_head: {
      metric: {
        id: 'cascade_windows_contained',
        label: 'Cascade windows contained per scenario',
        direction: 'higher_is_better',
        tie_rule: 'exact equality in integer contained-window count',
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
    limitations: [
      'Purpose-built artificial patterns; not evidence of real-disaster performance.',
      'Objective pass counts are independent; they are not a single complementary win/loss scoreline.',
      'Version 1 remains preserved as immutable archived predecessor evidence.',
    ],
    note: 'The 38/40 learned-policy and 20/40 static-heuristic figures are independent objective pass counts. The complementary matched head-to-head scoreline is 38 learned wins, 0 heuristic wins, and 2 ties.',
    provenance: [
      {
        label: 'Sealed result',
        source_repository: 'city-model-workbench',
        path: 'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/final/result.json',
        sha256: 'a69fbb96087298abaec35bae2a2797cca24c696ff8ee1463913d6e1d84cd5a5b',
      },
    ],
  }
  return overview
}
