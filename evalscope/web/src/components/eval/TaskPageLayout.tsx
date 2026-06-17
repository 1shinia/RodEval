import type { ReactNode } from 'react'
import TaskMonitor from '@/components/eval/TaskMonitor'
import Card from '@/components/ui/Card'
import type { EvalInvokeResponse } from '@/api/types'
import { Copy, Check } from 'lucide-react'

interface Props {
  title: string
  configTitle: string
  statusTitle: string
  readyLabel: string
  running: boolean
  progress: number
  result: EvalInvokeResponse | null
  logText: string
  reportUrl: string | null
  copied: boolean
  onCopy: () => void
  onStop: () => void
  children: ReactNode
}

export default function TaskPageLayout({
  title, configTitle, statusTitle, readyLabel,
  running, progress, result, logText, reportUrl,
  copied, onCopy, onStop, children,
}: Props) {
  return (
    <div className="page-enter">
      <h1 className="text-xl font-semibold mb-6">{title}</h1>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title={configTitle}>
          {children}
        </Card>
        <Card title={statusTitle} action={
          <button onClick={onCopy}
            className="flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-colors text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)]"
            title="复制全部日志">
            {copied ? <Check size={13} className="text-[var(--green)]" /> : <Copy size={13} />}
            {copied ? '已复制' : '复制日志'}
          </button>
        }>
          <TaskMonitor
            running={running}
            progress={progress}
            logText={logText}
            result={result}
            reportUrl={reportUrl}
            readyLabel={readyLabel}
            onStop={onStop}
          />
        </Card>
      </div>
    </div>
  )
}
