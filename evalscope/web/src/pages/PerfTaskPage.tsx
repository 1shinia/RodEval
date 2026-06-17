import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import PerfConfigForm from '@/components/perf/PerfConfigForm'
import TaskMonitor from '@/components/eval/TaskMonitor'
import Card from '@/components/ui/Card'
import { submitPerfTask, stopPerfTask, getPerfProgress, getPerfLog, getPerfReportUrl } from '@/api/perf'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from '@/api/types'
import { usePolling } from '@/hooks/usePolling'
import { Copy, Check } from 'lucide-react'

export default function PerfTaskPage() {
  const { t } = useLocale()
  const queryParams = useQueryParams()
  const urlTaskId = queryParams.get('task')

  const [taskId, setTaskId] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<EvalInvokeResponse | null>(null)
  const [logText, setLogText] = useState('')
  const [logLine, setLogLine] = useState(0)
  const [progress, setProgress] = useState(0)
  const [copied, setCopied] = useState(false)
  const resumedRef = useRef(false)

  // Resume monitoring a task from URL ?task=xxx (e.g. from running tasks indicator)
  useEffect(() => {
    if (urlTaskId && !resumedRef.current) {
      resumedRef.current = true
      setTaskId(urlTaskId)
      setRunning(true)
      // Fetch initial log immediately so it shows even if the task already finished
      getPerfLog(urlTaskId, 0).then((d) => {
        if (d.text) { setLogText(d.text); setLogLine(d.tail_line) }
      }).catch(() => {})
      // Check progress to see if task is still running
      getPerfProgress(urlTaskId).then((d) => {
        setProgress(d.percent ?? 0)
        if (d.percent >= 100) {
          setRunning(false)
          setResult({ status: 'completed', task_id: urlTaskId })
        }
      }).catch(() => { setRunning(false) })
    }
  }, [urlTaskId])

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
      try {
        const finalLog = await getPerfLog(id, logLine)
        if (finalLog.text) {
          setLogText((prev) => prev + finalLog.text)
          setLogLine(finalLog.tail_line)
        }
        const finalProg = await getPerfProgress(id)
        setProgress(finalProg.percent ?? 100)
      } catch { /* ignore */ }
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
      if (d.percent >= 100) {
        setRunning(false)
        setResult((prev) => prev ?? { status: 'completed', task_id: taskId })
      }
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

  return (
    <div className="page-enter">
      <h1 className="text-xl font-semibold mb-6">{t('perf.title')}</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
    </div>
  )
}
