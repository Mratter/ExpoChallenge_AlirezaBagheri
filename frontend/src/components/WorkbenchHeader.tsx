import { Activity, Database } from 'lucide-react'

export function WorkbenchHeader({ projectName }: { projectName: string }) {
  return (
    <header className="workbench-header">
      <a className="brand" href="#top" aria-label={`${projectName} workbench home`}>
        <span className="brand-sigil" aria-hidden="true">
          <i /><i /><i /><i /><i />
        </span>
        <span>
          <b>{projectName}</b>
          <small>Live model toolbox</small>
        </span>
      </a>
      <div className="runtime-status" aria-label="Model API connected">
        <Activity size={15} aria-hidden="true" />
        <span>Model API</span>
        <strong>CONNECTED</strong>
        <Database size={14} aria-hidden="true" />
      </div>
    </header>
  )
}
