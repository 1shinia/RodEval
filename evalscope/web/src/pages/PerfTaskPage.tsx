import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import PerfConfigForm from '@/components/perf/PerfConfigForm'
import TaskPageLayout from '@/components/eval/TaskPageLayout'
import { useTaskRunner } from '@/hooks/useTaskRunner'
import { submitPerfTask, stopPerfTask, getPerfProgress, getPerfLog, getPerfReportUrl, resumePerfTask, launchBatchPerf, getBatchStatus, stopBatchPerf } from '@/api/perf'
import type { BatchStatus } from '@/api/perf'
import { toast } from '@/components/common/Toast'

const perfApi = {
  submit: submitPerfTask,
  stop: stopPerfTask,
  getProgress: getPerfProgress,
  getLog: getPerfLog,
  getReportUrl: getPerfReportUrl,
  resume: resumePerfTask,
}

export default function PerfTaskPage() {
  const { t } = useLocale()
  const apiKeyRef = useRef('')

  const api = useMemo(() => perfApi, [])
  const { running, progress, result, logText, reportUrl, copied, taskId,
    handleSubmit, handleStop, handleResume: rawResume, copyLog } = useTaskRunner({ api, taskPrefix: 'perf' })

  const onApiKeyChange = useCallback((key: string) => { apiKeyRef.current = key }, [])
  const handleResume = useCallback((id: string) => { rawResume(id, apiKeyRef.current || undefined) }, [rawResume])

  // Batch state
  const [batchRunning, setBatchRunning] = useState(false)
  const [batchState, setBatchState] = useState<BatchStatus | null>(null)
  const [batchLogText, setBatchLogText] = useState('')
  const [selectedTaskId, setSelectedTaskId] = useState('')
  const [selectedTaskLog, setSelectedTaskLog] = useState('')
  const batchIdRef = useRef<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const lastLogTaskIdRef = useRef<string>('')

  const clearBatchPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const fetchTaskLog = useCallback(async (tid: string) => {
    try {
      const log = await getPerfLog(tid)
      setSelectedTaskLog(log.text || '')
    } catch {
      setSelectedTaskLog('')
    }
  }, [])

  const getDisplayLog = useCallback(() => {
    if (running) return logText
    if (batchRunning) return batchLogText
    return selectedTaskId ? selectedTaskLog : logText
  }, [running, logText, batchRunning, batchLogText, selectedTaskId, selectedTaskLog])

  const copyCurrentLog = useCallback(async () => {
    const text = getDisplayLog()
    if (!text) return
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
      } else {
        const ta = document.createElement('textarea')
        ta.value = text
        ta.style.position = 'fixed'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        document.body.removeChild(ta)
      }
      toast.success('日志已复制')
    } catch {
      toast.error('复制失败')
    }
  }, [getDisplayLog])

  const handleSelectTask = useCallback((tid: string) => {
    setSelectedTaskId(tid)
    fetchTaskLog(tid)
  }, [fetchTaskLog])

  const handleBatchStop = useCallback(async () => {
    const bid = batchIdRef.current
    if (!bid) return
    try {
      await stopBatchPerf(bid)
      toast.info('正在停止批量测试...')
    } catch (e) {
      toast.error(String(e))
    }
  }, [])

  const handleBatchSubmit = useCallback(async (batchId: string, sharedConfig: Record<string, unknown>) => {
    setBatchRunning(true)
    setBatchState(null)
    setBatchLogText('')
    setSelectedTaskId('')
    setSelectedTaskLog('')
    batchIdRef.current = batchId
    clearBatchPoll()

    try {
      const launched = await launchBatchPerf(batchId, sharedConfig)
      toast.info(`批量测试已启动，共 ${launched.total} 个模型`)

      pollRef.current = setInterval(async () => {
        try {
          const st = await getBatchStatus(batchId)
          setBatchState(st)

          // Fetch log for current task (only during running)
          if (st.current_task_id && st.current_task_id !== lastLogTaskIdRef.current) {
            lastLogTaskIdRef.current = st.current_task_id
            setBatchLogText('')
          }
          if (st.current_task_id) {
            try {
              const log = await getPerfLog(st.current_task_id)
              setBatchLogText(log.text || '')
            } catch { /* ignore */ }
          }

          if (st.status !== 'running') {
            clearBatchPoll()
            setBatchRunning(false)

            // Auto-select last result and load its log
            const results = st.results || []
            if (results.length > 0) {
              const last = results[results.length - 1]
              setSelectedTaskId(last.task_id)
              fetchTaskLog(last.task_id)
            }

            if (st.status === 'completed') {
              if (st.errors > 0) {
                toast.warning(`批量测试完成：${st.completed} 成功，${st.errors} 失败`)
              } else {
                toast.success(`批量测试完成：${st.completed} 个模型全部成功`)
              }
            } else if (st.status === 'cancelled') {
              toast.info(`批量测试已取消：${st.completed} 完成`)
            }
          }
        } catch (e) {
          // Batch state lost (e.g. server restart) or network error
          clearBatchPoll()
          setBatchRunning(false)
          toast.warning('批量状态丢失（服务可能已重启）')
        }
      }, 3000)
    } catch (e) {
      toast.error(String(e))
      setBatchRunning(false)
    }
  }, [clearBatchPoll, fetchTaskLog])

  useEffect(() => {
    return () => clearBatchPoll()
  }, [clearBatchPoll])

  return (
    <TaskPageLayout
      title={t('perf.title')}
      configTitle={t('perf.config')}
      statusTitle={t('perf.status')}
      readyLabel={t('perf.ready')}
      running={running || batchRunning}
      progress={running ? progress : 0}
      result={result}
      logText={running ? logText : (batchRunning ? batchLogText : (selectedTaskId ? selectedTaskLog : logText))}
      reportUrl={reportUrl}
      copied={copied}
      onCopy={copyCurrentLog}
      onStop={running ? handleStop : handleBatchStop}
      onResume={handleResume}
      taskId={taskId}
    >
      <PerfConfigForm
        onSubmit={handleSubmit}
        disabled={running || batchRunning}
        onApiKeyChange={onApiKeyChange}
        onBatchSubmit={handleBatchSubmit}
      />

      {/* Batch progress */}
      {batchRunning && batchState && batchState.status === 'running' && (
        <div className="mt-4 p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-card2)]">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium">
              批量测试中：{batchState.completed}/{batchState.total}
            </h3>
            <button
              onClick={handleBatchStop}
              className="px-2 py-1 text-xs rounded border border-[var(--danger)] text-[var(--danger)] hover:bg-[var(--danger)]/10 transition-colors"
            >
              停止
            </button>
          </div>
          <div className="w-full bg-[var(--bg)] rounded-full h-2 mb-2">
            <div
              className="bg-[var(--accent)] h-2 rounded-full transition-all duration-500"
              style={{ width: `${batchState.total > 0 ? (batchState.completed / batchState.total) * 100 : 0}%` }}
            />
          </div>
          {batchState.current_model && (
            <p className="text-xs text-[var(--text-muted)]">
              当前: {batchState.current_model}
            </p>
          )}
          {batchState.errors > 0 && (
            <p className="text-xs text-[var(--danger)] mt-1">{batchState.errors} 个失败</p>
          )}
        </div>
      )}

      {/* Batch result summary */}
      {batchState && batchState.status !== 'running' && (
        <div className="mt-4 p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-card2)]">
          <h3 className="text-sm font-medium mb-2">
            {batchState.status === 'completed' ? '批量测试完成' : '批量测试已取消'}：
            {batchState.completed} 成功
            {batchState.errors > 0 && <span className="text-[var(--danger)]">，{batchState.errors} 失败</span>}
          </h3>
          <p className="text-xs text-[var(--text-muted)] mb-2">点击模型查看对应日志</p>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {batchState.results.map((r) => (
              <div
                key={r.task_id}
                onClick={() => handleSelectTask(r.task_id)}
                className={`flex items-center gap-2 text-xs rounded px-1.5 py-0.5 -mx-1.5 cursor-pointer transition-colors ${
                  r.task_id === selectedTaskId
                    ? 'bg-[var(--accent)]/10 ring-1 ring-[var(--accent-dim)]'
                    : 'hover:bg-[var(--bg)]'
                }`}
              >
                {r.status === 'error' ? (
                  <span className="text-[var(--danger)]">✗</span>
                ) : (
                  <span className="text-[var(--green)]">✓</span>
                )}
                <span className="text-[var(--text)]">{r.name}</span>
                <span className="text-[var(--text-muted)]">({r.model})</span>
                {r.status === 'error' && r.error && (
                  <span className="text-[var(--danger)] truncate max-w-48" title={r.error}>{r.error}</span>
                )}
                <span className="text-[var(--text-dim)] ml-auto">{r.task_id}</span>
              </div>
            ))}
            {batchState.error_details.filter((e) => !batchState.results.some((r) => r.name === e.name)).map((e, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="text-[var(--danger)]">✗</span>
                <span className="text-[var(--text)]">{e.name}</span>
                <span className="text-[var(--text-muted)]">({e.model})</span>
                <span className="text-[var(--danger)] ml-auto">{e.error}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </TaskPageLayout>
  )
}
