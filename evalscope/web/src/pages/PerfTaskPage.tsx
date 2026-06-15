import { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import PerfConfigForm from '@/components/perf/PerfConfigForm'
import TaskMonitor from '@/components/eval/TaskMonitor'
import Card from '@/components/ui/Card'
import { submitPerfTask, stopPerfTask, getPerfProgress, getPerfLog, getPerfReportUrl, listPerfTasks, deletePerfTask, type PerfTaskMeta } from '@/api/perf'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from '@/api/types'
import { usePolling } from '@/hooks/usePolling'
import { ExternalLink, History, Copy, Check, Trash2 } from 'lucide-react'

export default function PerfTaskPage() {
  const { t } = useLocale()
  const [taskId, setTaskId] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<EvalInvokeResponse | null>(null)
  const [logText, setLogText] = useState('')
  const [logLine, setLogLine] = useState(0)
  const [progress, setProgress] = useState(0)
  const [copied, setCopied] = useState(false)

  // History state
  const [history, setHistory] = useState<PerfTaskMeta[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true)
    try {
      const res = await listPerfTasks()
      setHistory(res.tasks || [])
    } catch { /* ignore */ }
    finally { setHistoryLoading(false) }
  }, [])

  useEffect(() => { loadHistory() }, [loadHistory])

  const handleSubmit = async (config: Record<string, unknown>) => {
    const id = `perf_${Date.now()}`
    setTaskId(id)
    setRunning(true)
    setLogText('')
    setLogLine(0)
    setProgress(0)
    setResult(null)
    try {
      const res = await submitPerfTask(config, id)
      setResult(res)
    } catch (e) {
      setResult({ status: 'error', task_id: id, error: String(e) })
    } finally {
      setRunning(false)
      // Final fetch to catch any remaining log lines and 100% progress
      try {
        const finalLog = await getPerfLog(id, logLine)
        if (finalLog.text) {
          setLogText((prev) => prev + finalLog.text)
          setLogLine(finalLog.tail_line)
        }
        const finalProg = await getPerfProgress(id)
        setProgress(finalProg.percent ?? 100)
      } catch { /* ignore */ }
      loadHistory()
    }
  }

  const handleStop = async () => {
    if (!taskId) return
    try {
      await stopPerfTask(taskId)
    } catch { /* ignore errors */ }
    setRunning(false)
    setResult({ status: 'stopped', task_id: taskId })
  }

  const progressFn = useCallback(async () => {
    if (!taskId) throw new Error('no task')
    return getPerfProgress(taskId)
  }, [taskId])

  const logFn = useCallback(async () => {
    if (!taskId) throw new Error('no task')
    return getPerfLog(taskId, logLine)
  }, [taskId, logLine])

  usePolling<ProgressResponse>({
    fn: progressFn,
    enabled: running && !!taskId,
    interval: 5000,
    onData: (d) => {
      setProgress(d.percent ?? 0)
      if (d.percent >= 100) setRunning(false)
    },
  })

  usePolling<LogResponse>({
    fn: logFn,
    enabled: running && !!taskId,
    interval: 5000,
    onData: (d) => {
      if (d.text) {
        setLogText((prev) => prev + d.text)
        setLogLine(d.tail_line)
      }
    },
  })

  const reportUrl = useMemo(() => (taskId ? getPerfReportUrl(taskId) : null), [taskId])

  const handleViewReport = (tid: string) => {
    window.open(getPerfReportUrl(tid), '_blank')
  }

  const handleDelete = useCallback(async (tid: string, model: string) => {
    if (!window.confirm(`确定要删除 "${model}" 的压测记录吗？此操作不可撤销。`)) return
    try {
      await deletePerfTask(tid)
      loadHistory()
    } catch (e) {
      alert(e instanceof Error ? e.message : '删除失败')
    }
  }, [loadHistory])

  return (
    <div className="page-enter">
      <h1 className="text-xl font-semibold mb-6">{t('perf.title')}</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Left: Config Form */}
        <Card title={t('perf.config')}>
          <PerfConfigForm onSubmit={handleSubmit} disabled={running} />
        </Card>

        {/* Right: Task Monitor */}
        <Card title={t('perf.status')} action={
          <button onClick={() => {
            const text = [logText, result?.error].filter(Boolean).join('\n')
            if (!text) return
            const ta = document.createElement('textarea')
            ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0'
            document.body.appendChild(ta); ta.select()
            try { document.execCommand('copy') } catch { /* ignore */ }
            document.body.removeChild(ta)
            setCopied(true); setTimeout(() => setCopied(false), 2000)
          }}
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
            readyLabel={t('perf.ready')}
            onStop={handleStop}
          />
        </Card>
      </div>

      {/* History section */}
      {history.length > 0 && (
        <Card title="压测历史">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left py-2 px-3 text-xs text-[var(--text-muted)] font-medium">任务 ID</th>
                  <th className="text-left py-2 px-3 text-xs text-[var(--text-muted)] font-medium">模型</th>
                  <th className="text-left py-2 px-3 text-xs text-[var(--text-muted)] font-medium">API</th>
                  <th className="text-left py-2 px-3 text-xs text-[var(--text-muted)] font-medium">数据集</th>
                  <th className="text-left py-2 px-3 text-xs text-[var(--text-muted)] font-medium">并发配置数</th>
                  <th className="text-left py-2 px-3 text-xs text-[var(--text-muted)] font-medium">时间</th>
                  <th className="text-right py-2 px-3 text-xs text-[var(--text-muted)] font-medium">操作</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.task_id} className="border-b border-[var(--border)] hover:bg-[var(--bg-card2)] transition-colors">
                    <td className="py-2.5 px-3 font-mono text-xs text-[var(--text-muted)] max-w-[180px] truncate" title={item.task_id}>
                      {item.task_id.replace('perf_', '')}
                    </td>
                    <td className="py-2.5 px-3 font-medium text-[var(--text)] max-w-[200px] truncate" title={item.model}>
                      {item.model}
                    </td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)]">{item.api}</td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)] max-w-[150px] truncate" title={item.dataset}>
                      {item.dataset}
                    </td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)]">{item.runs}</td>
                    <td className="py-2.5 px-3 text-xs text-[var(--text-dim)] whitespace-nowrap">{item.timestamp}</td>
                    <td className="py-2.5 px-3 text-right whitespace-nowrap">
                      {item.has_report && (
                        <button
                          onClick={() => handleViewReport(item.task_id)}
                          className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded text-[var(--accent)] hover:bg-[var(--accent-dim)] transition-colors cursor-pointer"
                          title="查看报告"
                        >
                          <ExternalLink size={13} />
                          报告
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(item.task_id, item.model)}
                        className="inline-flex items-center gap-1 px-2 py-1 ml-1 text-xs rounded text-[var(--text-muted)] hover:bg-[var(--danger-bg)] hover:text-[var(--danger)] transition-colors cursor-pointer"
                        title="删除"
                      >
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Empty state */}
      {!historyLoading && history.length === 0 && (
        <div className="flex flex-col items-center justify-center py-12 gap-2 text-[var(--text-dim)]">
          <History size={32} />
          <p className="text-sm">暂无压测记录，运行一次压测后将在此显示</p>
        </div>
      )}
    </div>
  )
}
