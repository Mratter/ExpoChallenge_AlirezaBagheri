import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle,
  Check,
  CircleX,
  GitCompareArrows,
  GitFork,
  LoaderCircle,
  PencilLine,
  Play,
  RefreshCw,
  Scale,
  ShieldCheck,
} from 'lucide-react'
import {
  createPlan,
  createPlanningSession,
  executePlan,
  loadJudgeDemo,
  overridePlan,
  transitionPlan,
} from '../api'
import {
  services,
  type AuditActor,
  type CompareResponse,
  type JudgeBranchMode,
  type JudgeDemoResponse,
  type JudgeInitialOutcome,
  type PlanRecord,
  type PlanningSession,
  type Service,
} from '../types'
import './operator.css'

const labels: Record<Service, string> = {
  transport: 'Transport',
  housing: 'Housing',
  food: 'Food',
  healthcare: 'Healthcare',
  public_services: 'Public services',
}

function formatUnits(value: number): string {
  return value.toFixed(1)
}

function sameNumbers(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index])
}

function assertSessionMatchesResult(session: PlanningSession, result: CompareResponse): void {
  const provenance = session.checkpoint_selection_provenance
  if (
    session.source_result_id !== result.result_id
    || provenance.source_result_id !== result.result_id
    || provenance.source_result_policy_sha256 !== result.policy.sha256
    || provenance.source_result_schema_version !== result.schema_version
    || (result.engine_spec_sha256 !== undefined
      && provenance.source_engine_spec_sha256 !== result.engine_spec_sha256)
  ) {
    throw new Error('The planning session does not match the simulation result on screen.')
  }
}

function assertPlanMatchesResult(plan: PlanRecord, result: CompareResponse): void {
  assertSessionMatchesResult({
    schema_version: '1.0.0',
    session_id: plan.session_id,
    label: 'plan-source-check',
    source_result_id: plan.source_result_id,
    mode: 'simulation',
    simulation_only: true,
    simulation_auto_execute: plan.simulation_auto_execute,
    simulator_version_hash: plan.simulator_version_hash,
    checkpoint_selection_provenance: plan.checkpoint_selection_provenance,
    created_at: plan.created_at,
  }, result)
  if (plan.original_plan.length !== result.candidate.trajectory.length) {
    throw new Error('The authoritative plan does not match the simulation result on screen.')
  }
  const matches = plan.original_plan.every((day, index) => {
    const source = result.candidate.trajectory[index]
    return day.day === source.day
      && day.available_budget === source.available_budget
      && sameNumbers(day.original_policy_proposal, source.raw_proposal)
      && sameNumbers(day.original_feasible_allocation, source.allocation)
      && sameNumbers(day.lower_bounds, source.lower_bounds)
      && sameNumbers(day.upper_bounds, source.upper_bounds)
  })
  if (!matches) {
    throw new Error('The authoritative plan does not match the simulation result on screen.')
  }
}

function approvedAllocation(plan: PlanRecord | null, day: number, fallback: number[]): number[] {
  return plan?.reprojected_plan.find((item) => item.day === day)?.feasible_allocation ?? fallback
}

function serviceTotals(matrix: number[][]): number[] {
  return services.map((_, serviceIndex) => matrix.reduce(
    (total, district) => total + (district[serviceIndex] ?? 0),
    0,
  ))
}

function criticalAvailability(matrix: number[][]): number {
  const values = matrix.flatMap((district) => district.slice(3, 5))
  if (!values.length) return 0
  return 100 * values.reduce((sum, value) => sum + value, 0) / values.length
}

function JudgeInitialPolicyCard({
  label,
  outcome,
}: {
  label: string
  outcome: JudgeInitialOutcome
}) {
  const rawTotals = serviceTotals(outcome.first_day_decision.raw_priorities)
  const projectedTotals = serviceTotals(outcome.first_day_decision.projected_allocation)
  const projection = outcome.first_day_decision.projection_diagnostics

  return (
    <article className="judge-initial-policy">
      <header>
        <div>
          <span>{label}</span>
          <b>{outcome.policy_id}</b>
        </div>
        <small>day {outcome.first_day_decision.day}</small>
      </header>
      <div className="judge-projection-flow" aria-label={`${label} raw proposal to QP projection`}>
        <div className="judge-projection-heading">
          <span>Raw policy priority</span>
          <span aria-hidden="true">→</span>
          <span>QP allocation</span>
        </div>
        {services.map((service, index) => (
          <div key={service}>
            <small>{labels[service]}</small>
            <span>{(rawTotals[index] * 100).toFixed(1)}%</span>
            <span aria-hidden="true">→</span>
            <b>{projectedTotals[index].toFixed(1)} units</b>
          </div>
        ))}
      </div>
      <div className="judge-projection-proof">
        <span><b>{projection.distance.toFixed(3)}</b> QP distance</span>
        <span><b>{projection.constraint_violations}</b> violations</span>
        <span><b>{outcome.first_day_decision.service_pool.toFixed(1)}</b> unit pool</span>
      </div>
      <div className="judge-initial-trajectory" aria-label={`${label} initial trajectory`}>
        {outcome.trajectory.map((day) => {
          const availability = criticalAvailability(day.services_end)
          return (
            <div key={day.day}>
              <small>Day {day.day}</small>
              <span><i style={{ width: `${Math.max(0, Math.min(100, availability))}%` }} /></span>
              <b>{availability.toFixed(1)}% critical</b>
            </div>
          )
        })}
      </div>
    </article>
  )
}

function JudgeDemoProtocol({
  result,
  onEvidenceChange,
  requestedReplan,
}: {
  result: CompareResponse
  onEvidenceChange: (evidence: JudgeDemoResponse | null) => void
  requestedReplan: number
}) {
  const [step, setStep] = useState(0)
  const [selectedBranch, setSelectedBranch] = useState<JudgeBranchMode>(
    'open_loop_original_learned',
  )
  const [evidence, setEvidence] = useState<JudgeDemoResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [requestNumber, setRequestNumber] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    setEvidence(null)
    onEvidenceChange(null)
    void loadJudgeDemo(controller.signal)
      .then((payload) => {
        setEvidence(payload)
        onEvidenceChange(payload)
        setStep(0)
      })
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return
        setEvidence(null)
        onEvidenceChange(null)
        setError(
          caught instanceof Error
            ? caught.message
            : 'Verified v4 Judge evidence is not available.',
        )
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [onEvidenceChange, requestNumber])

  useEffect(() => {
    if (requestedReplan > 0 && evidence?.verification_status === 'verified') {
      setSelectedBranch('replan_learned')
    }
  }, [evidence, requestedReplan])

  const report = evidence?.report
  const steps = report ? [
    `Load held-out seed ${report.inputs.held_out_seed}`,
    'Verify the relational checkpoint and validation-selected baseline',
    'Run both policies on one identical initial tape',
    `Replay the learned prefix through day ${report.inputs.prefix_days}`,
    `Hold the explicit ${report.branch_point.explicit_active_disruption.type ?? 'disruption'} active on day ${report.branch_point.active_day}`,
    'Restore one exact current-state snapshot',
    'Install identical conditional tails across all three branches',
    'Compare mean, tail-risk, and worst-district outcomes',
    'Trace the counterfactual through graph and service dependencies',
    'Keep every performance claim preliminary and inconclusive',
  ] : []
  const branchLabels: Record<JudgeBranchMode, { title: string; description: string }> = {
    open_loop_original_learned: {
      title: 'Continue original',
      description: 'Replay proposals recorded before the branch; no policy call.',
    },
    replan_baseline: {
      title: 'Replan · baseline',
      description: 'Call the validation-selected non-learning policy from the same state.',
    },
    replan_learned: {
      title: 'Replan · learned',
      description: 'Call the verified relational policy again from the same state.',
    },
  }
  const selectedAggregate = report?.aggregates[selectedBranch]
  const selectedOutcome = report?.conditional_rollouts[0]?.branches[selectedBranch]
  const sharedProof = report?.conditional_rollouts[0]?.matched_proof
  const supportedClaimCount = report?.claim_status_rows.filter(
    (row) => row.evidence_status === 'supported',
  ).length ?? 0
  const inconclusiveClaimCount = report?.claim_status_rows.filter(
    (row) => row.evidence_status !== 'supported',
  ).length ?? 0
  const branchModes = Object.keys(branchLabels) as JudgeBranchMode[]
  const compactHash = (value: string | undefined) => value ? `${value.slice(0, 8)}…${value.slice(-6)}` : '—'
  const advanceStep = () => {
    if (!steps.length) return
    setStep((current) => (current + 1) % steps.length)
  }

  return (
    <section
      className="judge-demo-card"
      aria-labelledby="judge-demo-title"
      aria-busy={loading}
    >
      <header className="judge-demo-header">
        <div>
          <p className="operator-kicker"><span>Judge Demo Mode</span> · v4 preliminary evidence</p>
          <h3 id="judge-demo-title">One state. One future tape. Three decisions.</h3>
          <p>
            The branch view is separate from the v2 approval record above. It presents verified
            synthetic evidence and never executes a plan.
          </p>
        </div>
        <span className={`judge-verification ${report ? 'verified' : 'unavailable'}`}>
          {loading ? <LoaderCircle className="judge-spinner" size={15} /> : report ? <Check size={15} /> : <AlertTriangle size={15} />}
          {loading ? 'Checking evidence' : report ? 'Verified artifact' : 'Evidence unavailable'}
        </span>
      </header>

      {loading ? (
        <div className="judge-demo-state" role="status" aria-live="polite">
          <LoaderCircle className="judge-spinner" size={20} />
          <div>
            <b>Verifying the scientific evidence chain</b>
            <p>The v2 plan review remains available while the v4 suite and report are checked.</p>
          </div>
        </div>
      ) : error || !report || !evidence ? (
        <div className="judge-demo-state judge-demo-error" role="alert">
          <AlertTriangle size={20} />
          <div>
            <b>Verified three-branch evidence is not ready</b>
            <p>{error ?? 'The Judge report did not pass verification.'}</p>
            <p>
              The visible v2 result for seed {result.seed} remains reviewable; no v4 branch
              outcome is inferred from it.
            </p>
            <button
              type="button"
              className="operator-secondary"
              onClick={() => setRequestNumber((current) => current + 1)}
            >
              <RefreshCw size={15} />Check again
            </button>
          </div>
        </div>
      ) : (
        <>
          <section className="judge-initial-comparison" aria-labelledby="judge-initial-title">
            <header>
              <div>
                <p className="operator-kicker">Initial matched comparison</p>
                <h4 id="judge-initial-title">Baseline and learned plans on one frozen tape</h4>
              </div>
              <span><Check size={13} />same tape <code>{compactHash(report.initial_comparison.tape_sha256)}</code></span>
            </header>
            <div className="judge-initial-grid">
              <JudgeInitialPolicyCard
                label="Baseline initial plan"
                outcome={report.initial_comparison.baseline}
              />
              <JudgeInitialPolicyCard
                label="Learned initial plan"
                outcome={report.initial_comparison.learned}
              />
            </div>
          </section>

          <div className="judge-demo-sequence">
            <div className="judge-sequence-heading">
              <span>Evidence sequence</span>
              <b>{step + 1} / {steps.length}</b>
            </div>
            <ol className="judge-demo-steps">
              {steps.map((label, index) => (
                <li key={label} className={index <= step ? 'complete' : ''}>
                  <span>{index < step ? <Check size={13} /> : index + 1}</span>
                  <p>{label}</p>
                </li>
              ))}
            </ol>
            <button type="button" className="operator-secondary" onClick={advanceStep}>
              <Play size={15} />
              {step === steps.length - 1 ? 'Restart evidence sequence' : 'Advance evidence'}
            </button>
          </div>

          <div className="judge-branch-lab">
            <div className="judge-common-state">
              <GitFork size={18} />
              <div>
                <span>Matched branch point · day {report.branch_point.active_day}</span>
                <b>{report.branch_point.explicit_active_disruption.type ?? 'Disruption'} active · state {compactHash(report.branch_point.branch_state_sha256)}</b>
              </div>
              <small>{report.inputs.matched_rollout_seeds.length} conditional tails</small>
            </div>

            <div className="judge-branch-tabs" role="tablist" aria-label="Matched decision branch">
              {branchModes.map((mode) => (
                <button
                  type="button"
                  role="tab"
                  id={`judge-tab-${mode}`}
                  aria-controls={`judge-panel-${mode}`}
                  aria-selected={selectedBranch === mode}
                  tabIndex={selectedBranch === mode ? 0 : -1}
                  className={selectedBranch === mode ? 'active' : ''}
                  key={mode}
                  onClick={() => setSelectedBranch(mode)}
                  onKeyDown={(event) => {
                    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
                    event.preventDefault()
                    const index = branchModes.indexOf(mode)
                    const nextIndex = event.key === 'Home'
                      ? 0
                      : event.key === 'End'
                        ? branchModes.length - 1
                        : (index + (event.key === 'ArrowRight' ? 1 : -1) + branchModes.length)
                          % branchModes.length
                    const nextMode = branchModes[nextIndex]
                    setSelectedBranch(nextMode)
                    window.requestAnimationFrame(() => {
                      document.getElementById(`judge-tab-${nextMode}`)?.focus()
                    })
                  }}
                >
                  <span>{branchLabels[mode].title}</span>
                  <small>{mode === 'open_loop_original_learned' ? 'Recorded' : 'Fresh policy call'}</small>
                </button>
              ))}
            </div>

            <div
              className="judge-branch-panel"
              role="tabpanel"
              id={`judge-panel-${selectedBranch}`}
              aria-labelledby={`judge-tab-${selectedBranch}`}
              tabIndex={0}
            >
              <div className="judge-branch-title">
                <div>
                  <p className="operator-kicker">Selected branch</p>
                  <h4>{branchLabels[selectedBranch].title}</h4>
                </div>
                <span>{selectedOutcome?.policy_id}</span>
              </div>
              <p>{branchLabels[selectedBranch].description}</p>
              <div className="judge-demo-evidence" aria-label={`${branchLabels[selectedBranch].title} outcome metrics`}>
                <span>
                  <b>{selectedAggregate?.mean_cumulative_critical_service_days_lost.toFixed(2)}</b>
                  mean critical-service days lost
                </span>
                <span>
                  <b>{selectedAggregate?.cvar_10_weighted_unmet_need.toFixed(2)}</b>
                  10% tail unmet need
                </span>
                <span>
                  <b>{selectedAggregate?.maximum_worst_district_service_days_lost.toFixed(2)}</b>
                  worst observed district loss
                </span>
              </div>
              <div className="judge-proof-line">
                <span><Check size={13} />same state <code>{compactHash(sharedProof?.branch_state_sha256_before_tail)}</code></span>
                <span><Check size={13} />same future <code>{compactHash(sharedProof?.conditional_tail_sha256)}</code></span>
                <span><Check size={13} />{selectedAggregate?.hard_resource_constraint_violations ?? 0} hard constraint violations</span>
              </div>
            </div>
          </div>

          <div className="judge-evidence-boundaries">
            <section className="judge-claim-status" aria-labelledby="judge-claim-title">
              <div>
                <p className="operator-kicker">Claim status</p>
                <h4 id="judge-claim-title">Evidence boundaries by claim</h4>
              </div>
              <ul>
                {report.claim_status_rows.map((row) => (
                  <li key={row.claim}>
                    <span>{row.claim}</span>
                    <b>{row.evidence_status}</b>
                    <small>{row.claim_eligible ? 'claim eligible' : 'not claim-eligible'}</small>
                  </li>
                ))}
              </ul>
            </section>

            <aside className="judge-counterfactual" aria-label="Traceable synthetic counterfactual">
              <p className="operator-kicker">Traceable counterfactual</p>
              <blockquote>{report.counterfactual.narrative}</blockquote>
              <div className="judge-counterfactual-proof">
                <span><Check size={13} />actual matched rerun</span>
                <span><Check size={13} />removed action re-projected feasibly</span>
                <span>{report.counterfactual.matched_rollout_seeds.length} matched seeds</span>
              </div>
              <p>{report.counterfactual.interpretation_boundary}</p>
            </aside>
          </div>

          <footer className="judge-demo-footer">
            <p role="note">
              <AlertTriangle size={15} />
              Preliminary synthetic proxy only. {supportedClaimCount} listed claims supported;
              {' '}{inconclusiveClaimCount} remain inconclusive or not demonstrated.
            </p>
            <span>Report {compactHash(report.report_sha256)} · artifact {compactHash(evidence.artifact_manifest_sha256)}</span>
          </footer>
        </>
      )}
    </section>
  )
}

export function PlanReviewPanel({ result }: { result: CompareResponse }) {
  const [actor] = useState<AuditActor>(() => ({
    session_label: `analyst-toolbox-${globalThis.crypto?.randomUUID?.() ?? Math.random().toString(16).slice(2)}`,
  }))
  const [plan, setPlan] = useState<PlanRecord | null>(null)
  const [selectedDay, setSelectedDay] = useState(1)
  const [draftProposal, setDraftProposal] = useState<number[]>(
    result.candidate.trajectory[0].raw_proposal,
  )
  const [showBaseline, setShowBaseline] = useState(false)
  const [editing, setEditing] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [judgeEvidence, setJudgeEvidence] = useState<JudgeDemoResponse | null>(null)
  const [judgeReplanRequest, setJudgeReplanRequest] = useState(0)
  const handleJudgeEvidenceChange = useCallback(
    (evidence: JudgeDemoResponse | null) => setJudgeEvidence(evidence),
    [],
  )

  const candidateDay = result.candidate.trajectory[selectedDay - 1]
  const baselineDay = result.baseline.trajectory[selectedDay - 1]
  const authoritativeDay = plan?.original_plan.find((item) => item.day === selectedDay)
  const visibleRawProposal = authoritativeDay?.original_policy_proposal ?? candidateDay.raw_proposal
  const visibleFeasibleAllocation = (
    authoritativeDay?.original_feasible_allocation ?? candidateDay.allocation
  )
  const reprojected = plan?.reprojected_plan.find((item) => item.day === selectedDay)
  const approved = approvedAllocation(plan, selectedDay, visibleFeasibleAllocation)
  const totalSolverChange = useMemo(
    () => result.candidate.trajectory.reduce((sum, day) => sum + day.projection.distance, 0),
    [result],
  )
  const judgeReplanAvailable = judgeEvidence?.verification_status === 'verified'
    && judgeEvidence.report.conditional_rollouts.some(
      (rollout) => rollout.branches.replan_learned.replanning,
    )

  const runAction = async (action: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    setMessage(null)
    try {
      await action()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The operator action failed.')
    } finally {
      setBusy(false)
    }
  }

  const startReview = () => runAction(async () => {
    const session = await createPlanningSession(
      `Review ${result.scenario.name}`,
      result.result_id,
    )
    assertSessionMatchesResult(session, result)
    const created = await createPlan(
      session.session_id,
      result.result_id,
      actor,
      'Create an operator-reviewed simulation plan from the frozen policy proposal.',
    )
    assertPlanMatchesResult(created, result)
    setPlan(created)
    setMessage('Plan proposed. Operational execution remains locked pending human approval.')
  })

  const changeStatus = (
    action: 'review' | 'approve' | 'reject' | 'reproject',
    reason: string,
  ) => runAction(async () => {
    if (!plan) return
    const next = await transitionPlan(plan.plan_id, action, {
      expected_version: plan.version,
      actor,
      reason,
    })
    setPlan(next)
    setMessage(`Plan ${next.status}.`)
  })

  const saveOverride = () => runAction(async () => {
    if (!plan) return
    const overridden = await overridePlan(plan.plan_id, {
      expected_version: plan.version,
      actor,
      reason: `Modify day ${selectedDay} allocation before approval.`,
      changes: [{ day: selectedDay, proposal: draftProposal }],
    })
    setPlan(overridden)
    setEditing(false)
    setMessage('Operator change recorded. Re-project it before approval.')
  })

  const approve = () => runAction(async () => {
    if (!plan) return
    let current = plan
    if (current.status === 'proposed') {
      current = await transitionPlan(current.plan_id, 'review', {
        expected_version: current.version,
        actor,
        reason: 'Human operator completed plan review.',
      })
    }
    const approvedPlan = await transitionPlan(current.plan_id, 'approve', {
      expected_version: current.version,
      actor,
      reason: 'Human operator explicitly approved this simulation plan.',
    })
    setPlan(approvedPlan)
    setMessage('Plan approved for simulation execution only.')
  })

  const execute = () => runAction(async () => {
    if (!plan) return
    const executed = await executePlan(plan.plan_id, {
      expected_version: plan.version,
      actor,
      reason: 'Execute the explicitly approved plan inside the simulator.',
    })
    setPlan(executed.plan)
    setMessage(`Simulation execution recorded as ${executed.execution.execution_id.slice(0, 8)}.`)
  })

  const selectDay = (day: number) => {
    setSelectedDay(day)
    const recorded = plan?.original_plan.find((item) => item.day === day)
    setDraftProposal(
      recorded?.original_policy_proposal ?? result.candidate.trajectory[day - 1].raw_proposal,
    )
    setEditing(false)
  }

  return (
    <div className="operator-workspace">
      <section className="operator-review-card" aria-labelledby="operator-review-title">
        <header className="operator-review-header">
          <div>
            <p className="operator-kicker">
              Autonomous City Recovery Planner · human oversight / simulation only
            </p>
            <h3 id="operator-review-title">Plan approval desk</h3>
            <p>
              An AI-assisted city-recovery planning and simulation platform. The policy can plan
              autonomously inside the digital twin, while operational decisions remain subject to
              human approval.
            </p>
          </div>
          <div className={`plan-status status-${plan?.status ?? 'untracked'}`}>
            <ShieldCheck size={17} />
            <span><b>{plan?.status ?? 'untracked'}</b>{plan ? `version ${plan.version}` : 'no review record'}</span>
          </div>
        </header>

        <div className="operator-summary" aria-label="Solver contribution summary">
          <span><Scale size={15} /><b>{totalSolverChange.toFixed(2)}</b> total L2 projection distance</span>
          <span><b>{candidateDay.projection.distance.toFixed(2)}</b> selected-day distance</span>
          <span><b>{candidateDay.projection.bindings.filter((item) => item.lower || item.upper).length}</b> binding constraints</span>
          <span><b>{candidateDay.projection.constraint_violations}</b> post-projection violations</span>
        </div>

        <div className="operator-day-strip" role="tablist" aria-label="Plan day">
          {(plan?.original_plan ?? result.candidate.trajectory).map((day) => (
            <button
              type="button"
              role="tab"
              aria-selected={selectedDay === day.day}
              className={selectedDay === day.day ? 'active' : ''}
              key={day.day}
              onClick={() => selectDay(day.day)}
            >
              {day.day}
            </button>
          ))}
        </div>

        <div className={`proposal-grid ${showBaseline ? 'show-baseline' : ''}`}>
          <div className="proposal-column neural-column">
            <span>Raw neural proposal</span>
            <strong>{formatUnits(visibleRawProposal.reduce((sum, value) => sum + value, 0))} units</strong>
            {services.map((service, index) => (
              <div key={service}><small>{labels[service]}</small><b>{formatUnits(visibleRawProposal[index])}</b></div>
            ))}
          </div>
          <div className="proposal-column projected-column">
            <span>Solver-projected plan</span>
            <strong>{formatUnits(visibleFeasibleAllocation.reduce((sum, value) => sum + value, 0))} units</strong>
            {services.map((service, index) => (
              <div key={service}><small>{labels[service]}</small><b>{formatUnits(visibleFeasibleAllocation[index])}</b></div>
            ))}
          </div>
          <div className="proposal-column approved-column">
            <span>Operator-approved plan</span>
            <strong>{plan?.status === 'approved' || plan?.status === 'executed' ? plan.status : 'pending review'}</strong>
            {services.map((service, index) => (
              <div key={service}>
                <small>{labels[service]}</small>
                {editing ? (
                  <input
                    aria-label={`${labels[service]} operator proposal`}
                    type="number"
                    min="0"
                    step="0.1"
                    value={draftProposal[index]}
                    onChange={(event) => setDraftProposal((current) => current.map(
                      (value, valueIndex) => valueIndex === index ? Number(event.target.value) : value,
                    ))}
                  />
                ) : <b>{formatUnits(approved[index])}</b>}
              </div>
            ))}
          </div>
          {showBaseline ? (
            <div className="proposal-column baseline-column">
              <span>Legacy visible baseline</span>
              <strong>OR-Tools GLOP · not strongest-selected</strong>
              {services.map((service, index) => (
                <div key={service}><small>{labels[service]}</small><b>{formatUnits(baselineDay.allocation[index])}</b></div>
              ))}
            </div>
          ) : null}
        </div>

        {reprojected ? (
          <div className="reprojection-callout" role="status">
            <RefreshCw size={15} />
            <p>
              Re-projection changed {reprojected.diagnostics.allocations_changed} allocations;
              L1 {reprojected.diagnostics.l1_distance.toFixed(2)}, L2 {reprojected.diagnostics.l2_distance.toFixed(2)}.
              {reprojected.diagnostics.solver_rescue ? ' The solver rescued an infeasible operator proposal.' : ' The operator proposal was already feasible.'}
            </p>
          </div>
        ) : null}

        <div className="operator-actions" aria-label="Plan lifecycle actions">
          {!plan ? (
            <button type="button" className="operator-primary" disabled={busy} onClick={() => void startReview()}>
              <ShieldCheck size={16} />Start operator review
            </button>
          ) : (
            <>
              <button type="button" className="operator-primary" disabled={busy || !['proposed', 'reviewed', 'reprojected'].includes(plan.status)} onClick={() => void approve()}>
                <Check size={16} />Approve
              </button>
              <button type="button" className="operator-danger" disabled={busy || plan.status !== 'proposed'} onClick={() => void changeStatus('reject', 'Operator rejected the proposed plan.')}>
                <CircleX size={16} />Reject
              </button>
              <button type="button" className="operator-secondary" disabled={busy || plan.status !== 'proposed'} onClick={() => setEditing((value) => !value)}>
                <PencilLine size={16} />Modify
              </button>
              {editing ? (
                <button type="button" className="operator-secondary" disabled={busy} onClick={() => void saveOverride()}>
                  Save override
                </button>
              ) : null}
              <button type="button" className="operator-secondary" disabled={busy || plan.status !== 'overridden'} onClick={() => void changeStatus('reproject', 'Re-run feasibility projection after operator changes.')}>
                <RefreshCw size={16} />Re-project
              </button>
              <button type="button" className="operator-secondary" disabled={busy} onClick={() => setShowBaseline((value) => !value)}>
                <GitCompareArrows size={16} />Compare with baseline
              </button>
              <button type="button" className="operator-secondary" disabled={busy || plan.status !== 'proposed'} onClick={() => {
                setDraftProposal(visibleRawProposal)
                setEditing(false)
                setMessage('Continuing the original policy proposal for the selected day.')
              }}>
                Continue original plan
              </button>
              <button
                type="button"
                className="operator-secondary"
                disabled={busy || !judgeReplanAvailable}
                title={judgeReplanAvailable
                  ? 'Show the verified learned branch from the matched current state'
                  : 'A verified matched current-state Judge report is required'}
                onClick={() => {
                  if (!judgeEvidence || !plan) return
                  setJudgeReplanRequest((current) => current + 1)
                  setMessage(
                    `Showing the verified v4 learned replan from day ${judgeEvidence.report.branch_point.active_day}. The v2 ${plan.status} plan remains unchanged at version ${plan.version}.`,
                  )
                }}
              >
                Replan from current state
              </button>
              <button type="button" className="operator-primary execute-button" disabled={busy || plan.status !== 'approved'} onClick={() => void execute()}>
                <Play size={16} />Execute approved simulation plan
              </button>
            </>
          )}
        </div>
        {message ? <p className="operator-message" role="status">{message}</p> : null}
        {error ? <p className="operator-error" role="alert">{error}</p> : null}
        {plan ? (
          <p role="note">
            Audit binding covers checkpoint selection only; the final evaluation protocol has not
            been executed.
          </p>
        ) : null}
      </section>

      <JudgeDemoProtocol
        result={result}
        onEvidenceChange={handleJudgeEvidenceChange}
        requestedReplan={judgeReplanRequest}
      />
    </div>
  )
}
