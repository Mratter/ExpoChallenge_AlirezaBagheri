/**
 * Display names for the two planners in the public runtime.
 *
 * The baseline is a hand-fitted rule, not a naive one: its allocation weights
 * carry tuned exponents and coefficients chosen against this environment, so
 * the interface names it accordingly.
 *
 * These are presentation labels only. The published Hurricane Maria
 * retrospective is a write-once bundle and still carries its original
 * `Reactive heuristic` label inside the frozen receipt, report, and generated
 * evidence module; `benchmarkDisplayLabel` maps that frozen label for display
 * without altering the artifact.
 */
export const ppoName = 'PPO policy'
export const reactiveHeuristicName = 'Fine tuned reactive heuristic'
/** Compact form for legends, readouts, and other width-constrained chrome. */
export const reactiveHeuristicShortName = 'Fine tuned heuristic'

const frozenReactiveLabel = 'Reactive heuristic'

export function benchmarkDisplayLabel(frozenLabel: string): string {
  return frozenLabel === frozenReactiveLabel ? reactiveHeuristicName : frozenLabel
}
