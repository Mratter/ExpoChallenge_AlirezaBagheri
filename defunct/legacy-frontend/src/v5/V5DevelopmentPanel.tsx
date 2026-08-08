import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  createV5OperatorPlan,
  loadV5DevelopmentSnapshot,
  loadV5JudgeDemo,
  loadV5Profiles,
  overrideV5OperatorPlan,
  transitionV5OperatorPlan,
} from '../api'
import type {
  V5DevelopmentSnapshot,
  V5JudgeDemo,
  V5JudgePostEventStrategy,
  V5JudgePolicyResult,
  V5OperatorPlan,
  V5ProfilesResponse,
  V5Sector,
} from '../types'
import './v5-dashboard.css'

const sectorLabels: Record<V5Sector, string> = {
  transport: 'Transport',
  housing: 'Housing',
  food: 'Food',
  healthcare: 'Healthcare',
  public_services: 'Public services',
  resilience: 'Resilience',
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function units(value: number): string {
  return value.toFixed(2)
}

function statusLabel(status: string): string {
  return status.replaceAll('_', ' ')
}

function EmptyValue({ label = 'Unavailable' }: { label?: string }) {
  return <span className="v5-empty-value">{label}</span>
}

function StrategicHeatmap({ snapshot }: { snapshot: V5DevelopmentSnapshot }) {
  const maximum = Math.max(...snapshot.strategic_action.desired_allocation.flat(), 1e-9)
  return (
    <div
      className="v5-table-scroll"
      role="region"
      aria-label="Scrollable sector by district strategic allocation"
      tabIndex={0}
    >
      <table className="v5-heatmap">
        <caption>Sector × district desired allocation before feasibility projection</caption>
        <thead>
          <tr>
            <th>Sector</th>
            {snapshot.strategic_action.district_ids.map((district) => (
              <th key={district}>{district.replace('district_', 'D')}</th>
            ))}
            <th>Sector share</th>
          </tr>
        </thead>
        <tbody>
          {snapshot.strategic_action.sector_names.map((sector, sectorIndex) => (
            <tr key={sector}>
              <th scope="row">{sectorLabels[sector]}</th>
              {snapshot.strategic_action.desired_allocation[sectorIndex].map((value, districtIndex) => (
                <td
                  key={snapshot.strategic_action.district_ids[districtIndex]}
                  style={{
                    backgroundColor: `rgba(79, 107, 88, ${0.07 + 0.58 * value / maximum})`,
                  }}
                >
                  {units(value)}
                </td>
              ))}
              <td><b>{percent(snapshot.strategic_action.sector_shares[sectorIndex])}</b></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function LatentVector({ snapshot }: { snapshot: V5DevelopmentSnapshot }) {
  return (
    <details className="v5-vector-disclosure">
      <summary>Inspect all 55 ordered outputs</summary>
      <p>{snapshot.strategic_action.interpretation}</p>
      <ol aria-label="Exact 55-output latent action" className="v5-vector-grid">
        {snapshot.strategic_action.latent_vector.map((value, index) => (
          <li key={index}>
            <span>{index}</span>
            <code>{value.toFixed(6)}</code>
          </li>
        ))}
      </ol>
    </details>
  )
}

function DevelopmentContent({ snapshot }: { snapshot: V5DevelopmentSnapshot }) {
  const relationEntries = Object.entries(snapshot.dependency_graph.relation_counts)
  return (
    <>
      <section className="v5-identity-grid" aria-label="v5 model and action identity">
        <article>
          <span>Action schema</span>
          <strong>{snapshot.action_schema.output_count} outputs</strong>
          <small>{snapshot.action_schema.version}</small>
        </article>
        <article>
          <span>Current policy</span>
          <strong>{snapshot.current_policy.policy_id}</strong>
          <small>Development baseline · no learned checkpoint</small>
        </article>
        <article>
          <span>Model capacity</span>
          <strong>{snapshot.model.trainable_parameter_count.toLocaleString('en-US')}</strong>
          <small>{snapshot.model.candidate_id} · checkpoint pending</small>
        </article>
        <article>
          <span>Proposal state</span>
          <strong>{snapshot.strategic_action.status}</strong>
          <small>Read-only · not executed</small>
        </article>
      </section>

      <section className="v5-card v5-strategy-card" aria-labelledby="v5-strategy-title">
        <header className="v5-card-heading">
          <div>
            <p className="section-kicker">Exact strategic proposal</p>
            <h4 id="v5-strategy-title">6 sectors × 8 conditional district heads + reserve</h4>
          </div>
          <div className="v5-reserve-readout" aria-label="Proposed carrying reserve">
            <span>Carrying reserve</span>
            <b>{units(snapshot.carrying_reserve.proposed_amount)} units</b>
            <small>{percent(snapshot.carrying_reserve.proposed_fraction)} proposed</small>
          </div>
        </header>
        <StrategicHeatmap snapshot={snapshot} />
        <LatentVector snapshot={snapshot} />
      </section>

      <div className="v5-summary-grid">
        <section className="v5-card" aria-labelledby="v5-graph-title">
          <p className="section-kicker">Dependency graph</p>
          <h4 id="v5-graph-title">{snapshot.dependency_graph.node_count} nodes · {snapshot.dependency_graph.directed_edge_count} directed edges</h4>
          <dl className="v5-compact-facts">
            <div><dt>Critical edges</dt><dd>{snapshot.dependency_graph.critical_edge_count}</dd></div>
            {relationEntries.map(([relation, count]) => (
              <div key={relation}><dt>{relation}</dt><dd>{count}</dd></div>
            ))}
          </dl>
        </section>

        <section className="v5-card" aria-labelledby="v5-projects-title">
          <p className="section-kicker">Project modes</p>
          <h4 id="v5-projects-title">Temporary and permanent restoration</h4>
          <dl className="v5-compact-facts">
            <div><dt>Temporary</dt><dd>{snapshot.projects.mode_counts.temporary}</dd></div>
            <div><dt>Permanent</dt><dd>{snapshot.projects.mode_counts.permanent}</dd></div>
            <div><dt>Resilience</dt><dd>{snapshot.projects.mode_counts.resilience}</dd></div>
          </dl>
          <p className="v5-subnote">Current project state: <b>{snapshot.projects.current_lifecycle_status}</b>. No transition was executed.</p>
          <details className="v5-project-disclosure">
            <summary>Inspect highest proposed projects</summary>
            <ul aria-label="Highest proposed temporary permanent and resilience projects">
              {snapshot.projects.top_proposals.map((project) => (
                <li key={project.project_id}>
                  <span><b>{project.mode}</b> · {project.district_id} / {sectorLabels[project.sector]}</span>
                  <strong>{units(project.proposed_amount)} units</strong>
                  <small>{project.status}</small>
                </li>
              ))}
            </ul>
          </details>
        </section>

        <section className="v5-card" aria-labelledby="v5-forecast-title">
          <p className="section-kicker">Public forecast only</p>
          <h4 id="v5-forecast-title">Forecast uncertainty</h4>
          <dl className="v5-compact-facts">
            <div><dt>Event probability</dt><dd>{percent(snapshot.forecast.event_probability)}</dd></div>
            <div><dt>Uncertainty</dt><dd>{percent(snapshot.forecast.uncertainty)}</dd></div>
            <div><dt>Severity interval</dt><dd>{percent(snapshot.forecast.severity_low)}–{percent(snapshot.forecast.severity_high)}</dd></div>
            <div><dt>Aid window</dt><dd>days {snapshot.forecast.aid_arrival_day_low}–{snapshot.forecast.aid_arrival_day_high}</dd></div>
          </dl>
          <p className="v5-subnote">{snapshot.displacement_and_equity.worst_district.scope}</p>
        </section>

        <section className="v5-card" aria-labelledby="v5-equity-title">
          <p className="section-kicker">Displacement and equity</p>
          <h4 id="v5-equity-title">Worst-district snapshot</h4>
          <dl className="v5-compact-facts">
            <div><dt>Displaced</dt><dd>{snapshot.displacement_and_equity.total_displaced_people.toLocaleString()}</dd></div>
            <div><dt>Worst district</dt><dd>{snapshot.displacement_and_equity.worst_district.district_id}</dd></div>
            <div><dt>Critical need proxy</dt><dd>{percent(snapshot.displacement_and_equity.worst_district.value)}</dd></div>
            <div><dt>Projection distance</dt><dd>{snapshot.projection.distance === null ? <EmptyValue label="Pending execution" /> : snapshot.projection.distance.toFixed(3)}</dd></div>
          </dl>
        </section>
      </div>

      <section className="v5-card" aria-labelledby="v5-comparison-title">
        <div className="v5-card-heading">
          <div>
            <p className="section-kicker">Matched evidence status</p>
            <h4 id="v5-comparison-title">Baseline / MPC / learned comparison</h4>
          </div>
          <div className="v5-evidence-labels" aria-label="Pilot versus final evidence labels">
            <span>STATUS / {statusLabel(snapshot.evidence.status).toUpperCase()}</span>
            <span>{snapshot.evidence.pilot.label}</span>
            <span>{snapshot.evidence.final.label}</span>
          </div>
        </div>
        <div className="v5-table-scroll" role="region" aria-label="Scrollable v5 policy comparison" tabIndex={0}>
          <table className="v5-comparison-table">
            <caption>v5 policy comparison</caption>
            <thead><tr><th>Family</th><th>Policy</th><th>Status</th><th>Endpoint evidence</th></tr></thead>
            <tbody>
              {snapshot.policy_comparison.map((item) => (
                <tr key={item.family}>
                  <th scope="row">{item.family}</th>
                  <td><code>{item.policy}</code></td>
                  <td>{statusLabel(item.status)}</td>
                  <td><EmptyValue label="Unavailable / pending" /><small>{item.reason}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="v5-card v5-lifecycle" aria-labelledby="v5-lifecycle-title">
        <p className="section-kicker">Human review boundary</p>
        <h4 id="v5-lifecycle-title">Preserved plan lifecycle</h4>
        <ol aria-label="Plan lifecycle statuses">
          {snapshot.projects.lifecycle_statuses.map((status) => (
            <li key={status} className={status === snapshot.projects.current_lifecycle_status ? 'current' : ''}>{status}</li>
          ))}
        </ol>
      </section>

      <footer className="v5-proof-footer">
        <div><b>Simulator</b><code>{snapshot.simulator.simulator_version_hash}</code></div>
        <div><b>Action schema</b><code>{snapshot.action_schema.sha256}</code></div>
        <p>This snapshot remains pre-transition. Projection, approval, execution, and matched endpoint evidence appear only in the separately recorded panels below.</p>
      </footer>
    </>
  )
}

const lifecycleLabels = ['proposed', 'reviewed', 'approved', 'rejected', 'overridden', 'reprojected', 'executed'] as const

function OperatorLifecyclePanel({
  profileId,
  seed,
  dryRunOnly,
}: {
  profileId: string
  seed: number
  dryRunOnly: boolean
}) {
  const [plan, setPlan] = useState<V5OperatorPlan | null>(null)
  const [operatorId, setOperatorId] = useState('local-operator')
  const [sessionId, setSessionId] = useState('v5-dashboard-session')
  const [reason, setReason] = useState('Review the proposed recovery allocation.')
  const [overrideProject, setOverrideProject] = useState('')
  const [overrideAmount, setOverrideAmount] = useState('0')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setPlan(null)
    setOverrideProject('')
  }, [profileId, seed])

  const identity = () => ({
    operator_id: operatorId.trim(),
    session_id: sessionId.trim(),
    reason: reason.trim(),
  })

  const accept = (next: V5OperatorPlan) => {
    setPlan(next)
    if (!overrideProject && next.comparison.length) {
      setOverrideProject(next.comparison[0].project_id)
      setOverrideAmount(next.comparison[0].operator_requested.toFixed(3))
    }
  }

  const run = async (operation: () => Promise<V5OperatorPlan>) => {
    if (!operatorId.trim() || !sessionId.trim() || !reason.trim()) {
      setError('Operator, session, and reason are required for every recorded action.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      accept(await operation())
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The lifecycle action failed.')
    } finally {
      setBusy(false)
    }
  }

  const transition = (action: 'review' | 'approve' | 'reject' | 'reproject' | 'execute') => {
    if (!plan) return
    void run(() => transitionV5OperatorPlan(plan, action, identity()))
  }

  const override = () => {
    if (!plan) return
    const amount = Number(overrideAmount)
    if (!Number.isFinite(amount) || amount < 0) {
      setError('Override amount must be a finite nonnegative number.')
      return
    }
    void run(() => overrideV5OperatorPlan(plan, identity(), {
      project_id: overrideProject,
      amount,
    }))
  }

  return (
    <section className="v5-card v5-operator-console" aria-labelledby="v5-operator-title">
      <header className="v5-card-heading">
        <div>
          <p className="section-kicker">Typed operator boundary</p>
          <h4 id="v5-operator-title">Proposal review and exact approval</h4>
          <p className="v5-subnote">Every mutation records operator, session, reason, time, version, and hash-linked evidence.</p>
        </div>
        <span className={`v5-status-stamp ${plan?.status ?? 'idle'}`}>
          {plan ? plan.status : 'No review record'}
        </span>
      </header>

      <div className="v5-operator-identity">
        <label><span>Operator ID</span><input aria-label="v5 operator ID" value={operatorId} onChange={(event) => setOperatorId(event.target.value)} /></label>
        <label><span>Session ID</span><input aria-label="v5 session ID" value={sessionId} onChange={(event) => setSessionId(event.target.value)} /></label>
        <label className="reason"><span>Recorded reason</span><input aria-label="v5 action reason" value={reason} onChange={(event) => setReason(event.target.value)} /></label>
      </div>

      {!plan ? (
        <div className="v5-operator-start">
          <p>{dryRunOnly ? 'This final configuration is dry-run only. Switch to a development profile for manual approval.' : 'Opening review runs feasibility projection but does not advance simulator physics.'}</p>
          <button
            type="button"
            disabled={busy || dryRunOnly}
            onClick={() => void run(() => createV5OperatorPlan(profileId, seed, identity()))}
          >
            {busy ? 'Opening review…' : 'Open auditable review'}
          </button>
        </div>
      ) : (
        <>
          <ol className="v5-review-rail" aria-label="Audited v5 lifecycle">
            {lifecycleLabels.map((status) => {
              const event = plan.audit_events.find((item) => item.status === status)
              return (
                <li key={status} className={status === plan.status ? 'current' : event ? 'complete' : ''}>
                  <span>{status}</span>
                  <small>{event ? `#${event.sequence} · ${event.operator_id}` : 'not recorded'}</small>
                </li>
              )
            })}
          </ol>

          <div className="v5-operator-actions" aria-label="v5 lifecycle actions">
            {plan.actions_available.includes('review') ? <button type="button" disabled={busy} onClick={() => transition('review')}>Mark reviewed</button> : null}
            {plan.actions_available.includes('approve') ? <button type="button" disabled={busy} onClick={() => transition('approve')}>Approve exact plan</button> : null}
            {plan.actions_available.includes('reproject') ? <button type="button" disabled={busy} onClick={() => transition('reproject')}>Reproject override</button> : null}
            {plan.actions_available.includes('execute') ? <button className="primary" type="button" disabled={busy} onClick={() => transition('execute')}>Execute approved step</button> : null}
            {plan.actions_available.includes('reject') ? <button className="danger" type="button" disabled={busy} onClick={() => transition('reject')}>Reject plan</button> : null}
          </div>

          {plan.actions_available.includes('override') ? (
            <div className="v5-override-editor">
              <label>
                <span>Project override</span>
                <select aria-label="v5 override project" value={overrideProject} onChange={(event) => {
                  setOverrideProject(event.target.value)
                  const row = plan.comparison.find((item) => item.project_id === event.target.value)
                  if (row) setOverrideAmount(row.operator_requested.toFixed(3))
                }}>
                  {plan.comparison.map((row) => <option key={row.project_id} value={row.project_id}>{row.project_id}</option>)}
                </select>
              </label>
              <label><span>Requested units</span><input aria-label="v5 override amount" type="number" min="0" step="0.01" value={overrideAmount} onChange={(event) => setOverrideAmount(event.target.value)} /></label>
              <button type="button" disabled={busy} onClick={override}>Record override</button>
            </div>
          ) : null}

          <div className="v5-projection-ledger">
            <div><span>L1 solver change</span><b>{units(plan.solver_projection.solver_change.allocation_l1_distance)}</b></div>
            <div><span>L2 solver change</span><b>{units(plan.solver_projection.solver_change.allocation_l2_distance)}</b></div>
            <div><span>Modified fraction</span><b>{percent(plan.solver_projection.solver_change.modified_allocation_fraction)}</b></div>
            <div><span>Change class</span><b>{statusLabel(plan.solver_projection.solver_change.change_class)}</b></div>
          </div>

          <div className="v5-table-scroll" role="region" aria-label="Scrollable raw projected and approved allocations" tabIndex={0}>
            <table className="v5-plan-comparison">
              <caption>Raw proposal → solver projection → operator request → approved execution</caption>
              <thead><tr><th>Project</th><th>Raw</th><th>Projected</th><th>Operator</th><th>Reprojected</th><th>Approved</th></tr></thead>
              <tbody>{plan.comparison.map((row) => (
                <tr key={row.project_id}>
                  <th scope="row">{row.project_id}</th>
                  <td>{units(row.raw_proposal)}</td>
                  <td>{units(row.solver_projection)}</td>
                  <td>{units(row.operator_requested)}</td>
                  <td>{row.reprojected === null ? '—' : units(row.reprojected)}</td>
                  <td>{row.approved === null ? '—' : units(row.approved)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>

          <details className="v5-audit-disclosure">
            <summary>Inspect {plan.audit_events.length} hash-linked audit events</summary>
            <ol>{plan.audit_events.map((event) => (
              <li key={event.event_sha256}>
                <b>{event.action}</b><span>{event.reason}</span><code>{event.event_sha256}</code>
              </li>
            ))}</ol>
          </details>
        </>
      )}
      {error ? <p className="v5-inline-error" role="alert">{error}</p> : null}
    </section>
  )
}

const judgeMetricRows = [
  ['cumulative_critical_service_days_lost', 'Critical-service days lost'],
  ['cvar_10_weighted_unmet_need', 'Unmet-need CVaR'],
  ['worst_district_service_days_lost', 'Worst-district loss'],
] as const

function JudgePlanEvidence({ label, result }: { label: string; result: V5JudgePolicyResult }) {
  const rawByProject = new Map(
    result.initial_plan.raw_policy_proposal.projects.map((item) => [item.project_id, item]),
  )
  const change = result.initial_plan.solver_change
  return (
    <article className="v5-judge-plan">
      <header>
        <div><span>{label}</span><b>{result.policy_id}</b></div>
        <small>{change.raw_proposal_already_feasible ? 'Raw proposal feasible' : `${change.changed_project_count} allocation${change.changed_project_count === 1 ? '' : 's'} changed by QP`}</small>
      </header>
      <div className="v5-judge-plan-stats">
        <span>L1 <b>{units(change.project_allocation_l1_distance)}</b></span>
        <span>L2 <b>{units(change.allocation_l2_distance)}</b></span>
        <span>Modified <b>{percent(change.fraction_project_allocations_changed)}</b></span>
        <span>Interventions <b>{change.intervention_count}</b></span>
      </div>
      <details>
        <summary>Raw proposal → QP projection ({result.initial_plan.qp_projection.allocations.length} projects)</summary>
        <div className="v5-table-scroll" role="region" aria-label={`${label} raw policy proposal and QP projection`} tabIndex={0}>
          <table>
            <thead><tr><th>Project</th><th>Raw</th><th>Projected</th></tr></thead>
            <tbody>{result.initial_plan.qp_projection.allocations.map((project) => (
              <tr key={project.project_id}>
                <th scope="row">{project.project_id}</th>
                <td>{units(rawByProject.get(project.project_id)?.requested_amount ?? 0)}</td>
                <td>{units(project.amount)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <code>{result.initial_plan.prepared_action_sha256}</code>
      </details>
      <details>
        <summary>Complete conditional action sequence ({result.expected_trajectory.planned_actions.length} steps)</summary>
        <p className="v5-subnote">Each action is recomputed from that step’s public observation on the disclosed fixed synthetic tape.</p>
        <div className="v5-table-scroll" role="region" aria-label={`${label} complete conditional action sequence`} tabIndex={0}>
          <table>
            <thead><tr><th>Step</th><th>Raw reserve</th><th>Projected reserve</th><th>QP changes</th><th>Projection L2</th></tr></thead>
            <tbody>{result.expected_trajectory.planned_actions.map((plan) => (
              <tr key={plan.step}><th scope="row">{plan.step}</th><td>{units(plan.raw_policy_proposal.reserve_amount)}</td><td>{units(plan.qp_projection.reserve_amount)}</td><td>{plan.solver_change.changed_project_count}</td><td>{units(plan.solver_change.allocation_l2_distance)}</td></tr>
            ))}</tbody>
          </table>
        </div>
        <code>{result.expected_trajectory.planned_actions_sha256}</code>
      </details>
      <details>
        <summary>Expected fixed-tape trajectory ({result.expected_trajectory.points.length} states)</summary>
        <p className="v5-subnote">{result.expected_trajectory.semantics}</p>
        <div className="v5-table-scroll" role="region" aria-label={`${label} expected fixed-tape trajectory`} tabIndex={0}>
          <table>
            <thead><tr><th>Step</th><th>Critical loss</th><th>Unmet-need CVaR</th><th>Worst district</th><th>Spendable</th></tr></thead>
            <tbody>{result.expected_trajectory.points.map((point) => (
              <tr key={point.step}>
                <th scope="row">{point.step}</th>
                <td>{units(point.cumulative_critical_service_days_lost)}</td>
                <td>{units(point.cvar_10_weighted_unmet_need)}</td>
                <td>{units(point.worst_district_service_days_lost)}</td>
                <td>{units(point.spendable_resources)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <code>{result.expected_trajectory.trajectory_sha256}</code>
      </details>
    </article>
  )
}

function JudgeAggregateTable({ label, result }: { label: string; result: V5JudgePolicyResult }) {
  return (
    <div className="v5-table-scroll" role="region" aria-label={`${label} raw event-suite aggregates`} tabIndex={0}>
      <table className="v5-judge-metrics">
        <caption>{label} · continue versus public-observation replan</caption>
        <thead><tr><th>Endpoint</th><th>Continue mean</th><th>Replan mean</th><th>Replan CVaR</th><th>Replan worst</th><th>Δ mean</th></tr></thead>
        <tbody>{judgeMetricRows.map(([metric, metricLabel]) => {
          const continued = result.strategies.continue_open_loop.metrics[metric]
          const replanned = result.strategies.replan_public_observation.metrics[metric]
          return <tr key={metric}><th scope="row">{metricLabel}</th><td>{units(continued.mean)}</td><td>{units(replanned.mean)}</td><td>{units(replanned.cvar_20_upper)}</td><td>{units(replanned.worst)}</td><td>{units(result.replan_minus_continue[metric].mean)}</td></tr>
        })}</tbody>
      </table>
    </div>
  )
}

type JudgeResponseView = 'compare' | 'continue' | 'replan'

function JudgePostEventTable({ demo, view }: { demo: V5JudgeDemo; view: JudgeResponseView }) {
  const learned = demo.post_event_comparison.strategies.learned_replan_from_current_state
  const allRows: Array<{ label: string; result: V5JudgePostEventStrategy }> = [
    { label: 'Continue original plan', result: demo.post_event_comparison.strategies.continue_original_plan },
    { label: 'Baseline replan', result: demo.post_event_comparison.strategies.baseline_replan_from_current_state },
    ...(learned.status === 'evaluated' ? [{ label: 'Learned replan', result: learned }] : []),
  ]
  const rows = allRows.filter(({ label }) => (
    view === 'compare'
    || (view === 'continue' && label === 'Continue original plan')
    || (view === 'replan' && label !== 'Continue original plan')
  ))
  return (
    <div className="v5-table-scroll" role="region" aria-label="Common-state post-event strategy comparison" tabIndex={0}>
      <table className="v5-post-event-table">
        <caption>All choices branch after the event from the same exact current state</caption>
        <thead><tr><th>Choice</th><th>Policy</th>{judgeMetricRows.map(([, label]) => <th colSpan={3} key={label}>{label}</th>)}</tr><tr><th /><th />{judgeMetricRows.flatMap(([, label]) => [<th key={`${label}-mean`}>Mean</th>, <th key={`${label}-cvar`}>CVaR</th>, <th key={`${label}-worst`}>Worst</th>])}</tr></thead>
        <tbody>{rows.map(({ label, result }) => <tr key={label}>
          <th scope="row">{label}</th><td>{result.policy_id}</td>
          {judgeMetricRows.flatMap(([metric, metricLabel]) => {
            const summary = result.metrics[metric]
            return [<td key={`${metricLabel}-mean`}>{units(summary.mean)}</td>, <td key={`${metricLabel}-cvar`}>{units(summary.cvar_20_upper)}</td>, <td key={`${metricLabel}-worst`}>{units(summary.worst)}</td>]
          })}
        </tr>)}</tbody>
      </table>
    </div>
  )
}

function JudgeDemoPanel() {
  const [demo, setDemo] = useState<V5JudgeDemo | null>(null)
  const [checkpointPath, setCheckpointPath] = useState('')
  const [checkpointSha, setCheckpointSha] = useState('')
  const [candidateId, setCandidateId] = useState('relational_gnn_ppo')
  const [validationIndexPath, setValidationIndexPath] = useState('')
  const [validationIndexSha, setValidationIndexSha] = useState('')
  const [responseView, setResponseView] = useState<JudgeResponseView>('compare')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runDemo = async () => {
    if (Boolean(checkpointPath.trim()) !== Boolean(checkpointSha.trim())) {
      setError('Checkpoint path and exact SHA-256 must be supplied together.')
      return
    }
    if (Boolean(validationIndexPath.trim()) !== Boolean(validationIndexSha.trim())) {
      setError('Shared-validation index path and exact SHA-256 must be supplied together.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const hasEvidenceInputs = Boolean(checkpointPath.trim() || validationIndexPath.trim())
      setDemo(await loadV5JudgeDemo(hasEvidenceInputs ? {
        checkpointPath: checkpointPath.trim() || undefined,
        checkpointSha256: checkpointSha.trim() || undefined,
        candidateId: checkpointPath.trim() ? candidateId : undefined,
        sharedValidationIndexPath: validationIndexPath.trim() || undefined,
        sharedValidationIndexSha256: validationIndexSha.trim() || undefined,
      } : undefined))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The fixed Judge Demo failed.')
    } finally {
      setBusy(false)
    }
  }
  const learnedResult = demo?.learned.status === 'evaluated' ? demo.learned : null

  return (
    <section className="v5-card v5-judge-demo" aria-labelledby="v5-judge-title">
      <header className="v5-card-heading">
        <div>
          <p className="section-kicker">Fixed held-out event suite</p>
          <h4 id="v5-judge-title">Judge Demo: continue, replan, remove one action</h4>
          <p className="v5-subnote">All registered event injections are reported. Lower endpoint costs are better; no event can be omitted after results are seen.</p>
        </div>
        <span className="v5-no-cherry-pick">ALL EVENTS / RAW ROWS</span>
      </header>
      <details className="v5-checkpoint-inputs">
        <summary>Optional checksum-verified baseline and learned evidence</summary>
        <div>
          <label><span>Shared-validation index path</span><input aria-label="v5 shared-validation index path" value={validationIndexPath} onChange={(event) => setValidationIndexPath(event.target.value)} /></label>
          <label><span>Shared-validation index SHA-256</span><input aria-label="v5 shared-validation index SHA-256" value={validationIndexSha} onChange={(event) => setValidationIndexSha(event.target.value)} /></label>
          <label><span>Checkpoint path</span><input aria-label="v5 learned checkpoint path" value={checkpointPath} onChange={(event) => setCheckpointPath(event.target.value)} /></label>
          <label><span>Exact SHA-256</span><input aria-label="v5 learned checkpoint SHA-256" value={checkpointSha} onChange={(event) => setCheckpointSha(event.target.value)} /></label>
          <label>
            <span>Checkpoint candidate</span>
            <select aria-label="v5 learned checkpoint candidate" value={candidateId} onChange={(event) => setCandidateId(event.target.value)}>
              <option value="mlp_ppo">MLP PPO</option>
              <option value="homogeneous_gnn_ppo">Homogeneous GNN PPO</option>
              <option value="relational_gnn_ppo">Relational GNN PPO</option>
              <option value="residual_relational_gnn_ppo">Residual relational GNN PPO</option>
              <option value="shuffled_edge_gnn_ppo">Shuffled-edge GNN diagnostic</option>
              <option value="no_risk_gnn_ppo">No-risk GNN diagnostic</option>
              <option value="fixed_reserve_gnn_ppo">Fixed-reserve GNN diagnostic</option>
              <option value="no_equity_gnn_ppo">No-equity GNN diagnostic</option>
            </select>
          </label>
        </div>
      </details>
      <button className="v5-run-judge" type="button" disabled={busy} onClick={() => void runDemo()}>{busy ? 'Running fixed event suite…' : 'Run fixed Judge Demo'}</button>
      {error ? <p className="v5-inline-error" role="alert">{error}</p> : null}
      {demo ? (
        <>
          <div className="v5-judge-identity">
            <div><span>Validation seed</span><b>{demo.validation_design.fixed_seed}</b></div>
            <div><span>Injected events</span><b>{demo.event_protocol.event_count} / {demo.event_protocol.event_count}</b></div>
            <div><span>Baseline</span><b>{demo.baseline_selection.strongest_baseline_verified ? 'strongest / verified' : 'default / not ranked'}</b></div>
            <div><span>Learned</span><b>{demo.learned.status}</b></div>
          </div>
          <p className="v5-synthetic-disclosure">
            Synthetic proxy disclosure: {demo.synthetic_proxy_disclosure.scenario_and_outcomes}; {demo.synthetic_proxy_disclosure.event_injections}; trajectories are {demo.synthetic_proxy_disclosure.expected_trajectories}. These are not operational forecasts.
          </p>
          {demo.learned.status === 'not_evaluated' ? <p className="v5-not-evaluated">Learned / not evaluated — {demo.learned.reason}</p> : null}
          <div className="v5-judge-plans">
            <JudgePlanEvidence label={demo.baseline_selection.strongest_baseline_verified ? 'Evidence-derived strongest baseline plan' : 'Configured default baseline plan'} result={demo.baseline} />
            {demo.learned.status === 'evaluated' ? <JudgePlanEvidence label="Verified learned plan" result={demo.learned} /> : null}
          </div>
          <section className="v5-post-event-comparison" aria-labelledby="v5-post-event-title">
            <header>
              <div>
                <p className="section-kicker">Exact common post-event branch</p>
                <h5 id="v5-post-event-title">Choose how to respond from the current state</h5>
              </div>
              <code>{demo.post_event_comparison.comparison_sha256}</code>
            </header>
            <div className="v5-judge-response-controls" role="group" aria-label="Judge post-event response view">
              <button type="button" aria-pressed={responseView === 'compare'} onClick={() => setResponseView('compare')}>Compare with baseline</button>
              <button type="button" aria-pressed={responseView === 'continue'} onClick={() => setResponseView('continue')}>Continue original plan</button>
              <button type="button" aria-pressed={responseView === 'replan'} onClick={() => setResponseView('replan')}>Replan from current state</button>
            </div>
            <p className="v5-common-state-proof">Event realized before branching · same post-event state for every choice · original schedule reads no future observations.</p>
            <JudgePostEventTable demo={demo} view={responseView} />
            {demo.post_event_comparison.strategies.learned_replan_from_current_state.status === 'not_evaluated' ? <p className="v5-not-evaluated">Learned replan / not evaluated — {demo.post_event_comparison.strategies.learned_replan_from_current_state.reason}</p> : null}
            <details>
              <summary>Inspect common state and event-tape bindings</summary>
              <ul>{demo.post_event_comparison.strategies.continue_original_plan.raw_rollouts.map((row) => <li key={row.event_id}><b>{row.event_id}</b><code>{row.post_event_state_sha256}</code></li>)}</ul>
            </details>
          </section>
          <div className="v5-judge-aggregate-grids">
            <JudgeAggregateTable label="Baseline" result={demo.baseline} />
            {demo.learned.status === 'evaluated' ? <JudgeAggregateTable label="Learned" result={demo.learned} /> : null}
          </div>
          <details className="v5-raw-rollouts">
            <summary>Inspect all {demo.event_protocol.event_count} raw paired event rows</summary>
            <table>
              <thead><tr><th>Event</th><th>Severity</th><th>Baseline continue loss</th><th>Baseline replan loss</th>{learnedResult ? <><th>Learned continue loss</th><th>Learned replan loss</th></> : null}</tr></thead>
              <tbody>{demo.baseline.event_injections.map((event, index) => (
                <tr key={event.event_id}><th scope="row">{event.event_id}</th><td>{percent(event.severity)}</td><td>{units(demo.baseline.strategies.continue_open_loop.raw_rollouts[index].cumulative_critical_service_days_lost)}</td><td>{units(demo.baseline.strategies.replan_public_observation.raw_rollouts[index].cumulative_critical_service_days_lost)}</td>{learnedResult ? <><td>{units(learnedResult.strategies.continue_open_loop.raw_rollouts[index].cumulative_critical_service_days_lost)}</td><td>{units(learnedResult.strategies.replan_public_observation.raw_rollouts[index].cumulative_critical_service_days_lost)}</td></> : null}</tr>
              ))}</tbody>
            </table>
          </details>
          <article className="v5-explanation-proof">
            <p className="section-kicker">Verified matched action removal</p>
            <h5>{demo.explanation.named_project_id}</h5>
            <p>{demo.explanation.narrative}</p>
            <small>Affected districts: {demo.explanation.affected_district_ids.join(', ')} · {demo.explanation.endpoint_summary.rollout_count} paired tapes</small>
            <code>{demo.explanation.explanation_sha256}</code>
          </article>
          <section className="v5-judge-claims" aria-label="Judge Demo scientific claim status">
            <p className="section-kicker">Scientific claim status</p>
            <ul>{demo.scientific_claims.map((claim) => (
              <li key={claim.claim_id}>
                <b>{statusLabel(claim.status)}</b>
                <span>{claim.statement}</span>
                {claim.reason ? <small>{claim.reason}</small> : null}
              </li>
            ))}</ul>
          </section>
        </>
      ) : null}
    </section>
  )
}

export function V5DevelopmentPanel() {
  const [registry, setRegistry] = useState<V5ProfilesResponse | null>(null)
  const [selectedProfile, setSelectedProfile] = useState('v5_diagnostic')
  const [snapshot, setSnapshot] = useState<V5DevelopmentSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  const loadSnapshot = useCallback(async (profile: string, signal?: AbortSignal) => {
    setLoading(true)
    setError(null)
    try {
      const response = await loadV5DevelopmentSnapshot(profile, signal)
      setSnapshot(response)
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      setSnapshot(null)
      setError(caught instanceof Error ? caught.message : 'The v5 snapshot is unavailable.')
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    controllerRef.current = controller
    void loadV5Profiles(controller.signal)
      .then((response) => {
        setRegistry(response)
        setSelectedProfile(response.default_profile_id)
        return loadSnapshot(response.default_profile_id, controller.signal)
      })
      .catch((caught) => {
        if (caught instanceof DOMException && caught.name === 'AbortError') return
        setError(caught instanceof Error ? caught.message : 'The v5 registry is unavailable.')
        setLoading(false)
      })
    return () => controllerRef.current?.abort()
  }, [loadSnapshot])

  const selectedMetadata = useMemo(
    () => registry?.profiles.find((profile) => profile.profile_id === selectedProfile),
    [registry, selectedProfile],
  )

  const changeProfile = (profile: string) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setSelectedProfile(profile)
    void loadSnapshot(profile, controller.signal)
  }

  return (
    <section className="v5-development-panel" aria-labelledby="v5-panel-title" aria-busy={loading}>
      <header className="v5-panel-header">
        <div>
          <p className="section-kicker">Separate v5 namespace / audited development console</p>
          <h3 id="v5-panel-title">v5 strategic preview</h3>
          <p className="v5-disclaimer">Developmental v5 simulator and model-selection evidence. Not operationally validated.</p>
        </div>
        <label className="v5-profile-selector">
          <span>Simulator version</span>
          <select
            aria-label="Simulator version"
            value={selectedProfile}
            disabled={!registry || loading}
            onChange={(event) => changeProfile(event.target.value)}
          >
            {(registry?.profiles ?? []).map((profile) => (
              <option key={profile.profile_id} value={profile.profile_id}>
                {profile.label} · {profile.horizon_days} days
              </option>
            ))}
          </select>
          <small>{selectedMetadata?.dry_run_only ? 'Dry-run configuration only' : 'Development profile'}</small>
        </label>
      </header>

      {loading && !snapshot ? <div className="v5-loading" role="status">Loading registered v5 contract…</div> : null}
      {error ? (
        <div className="v5-error" role="alert">
          <b>v5 evidence unavailable</b><span>{error}</span>
          <button type="button" onClick={() => changeProfile(selectedProfile)}>Retry</button>
        </div>
      ) : null}
      {snapshot ? (
        <>
          <DevelopmentContent snapshot={snapshot} />
          <OperatorLifecyclePanel
            profileId={snapshot.simulator.profile_id}
            seed={snapshot.simulator.seed}
            dryRunOnly={snapshot.simulator.dry_run_only}
          />
          <JudgeDemoPanel />
        </>
      ) : null}
      {loading && snapshot ? <div className="v5-refreshing" role="status">Refreshing profile…</div> : null}
    </section>
  )
}
