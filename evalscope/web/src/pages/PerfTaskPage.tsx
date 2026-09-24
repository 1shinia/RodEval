import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import PerfConfigForm from '@/components/perf/PerfConfigForm'
import TaskPageLayout from '@/components/eval/TaskPageLayout'
import { useTaskRunner } from '@/hooks/useTaskRunner'
import { submitPerfTask, launchPerfTask, stopPerfTask, getPerfProgress, getPerfLog, getPerfReportUrl, resumePerfTask,
  launchBatchPerf, resumeBatchPerf, getBatchStatus, stopBatchPerf, uploadBatchCsv } from '@/api/perf'
import type { BatchStatus } from '@/api/perf'
import { toast } from '@/components/common/Toast'
import { TASK_STATUSES, isActiveTaskStatus, normalizeTaskStatus } from '@/hooks/taskLifecycle'

const perfApi = {
  submit: submitPerfTask,
  launch: launchPerfTask,
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
  const { running, progress, progressError, result, logText, reportUrl, copied, taskId,
    handleSubmit, handleStop, handleResume: rawResume } = useTaskRunner({ api, taskPrefix: 'perf' })

  const onApiKeyChange = useCallback((key: string) => { apiKeyRef.current = key }, [])
  const handleResume = useCallback((id: string) => { rawResume(id, apiKeyRef.current || undefined) }, [rawResume])

  // Batch state
  const [batchRunning, setBatchRunning] = useState(false)
  const [batchState, setBatchState] = useState<BatchStatus | null>(null)
  const [batchLogText, setBatchLogText] = useState('')
  const [selectedTaskId, setSelectedTaskId] = useState('')
  const [selectedTaskLog, setSelectedTaskLog] = useState('')
  const batchIdRef = useRef<string | null>(null)
  const batchFileRef = useRef<File | null>(null)
  const batchConfigRef = useRef<Record<string, unknown>>({})
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollGenerationRef = useRef(0)
  const logGenerationRef = useRef(0)
  const lastLogTaskIdRef = useRef<string>('')

  const clearBatchPoll = useCallback(() => {
    pollGenerationRef.current += 1
    if (pollRef.current) { clearTimeout(pollRef.current); pollRef.current = null }
  }, [])

  const fetchTaskLog = useCallback(async (tid: string) => {
    const generation = ++logGenerationRef.current
    try {
      const log = await getPerfLog(tid)
      if (generation === logGenerationRef.current) setSelectedTaskLog(log.text || '')
    } catch {
      if (generation === logGenerationRef.current) setSelectedTaskLog('')
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
      toast.success(t('perf.logCopied'))
    } catch {
      toast.error(t('perf.copyFailed'))
    }
  }, [getDisplayLog, t])

  const handleSelectTask = useCallback((tid: string) => {
    setSelectedTaskId(tid)
    void fetchTaskLog(tid)
  }, [fetchTaskLog])

  const handleModeChange = useCallback((mode: 'single' | 'batch') => {
    if (mode === 'single') {
      batchIdRef.current = null
      clearBatchPoll()
      setBatchRunning(false)
      setBatchState(null)
      setBatchLogText('')
      setSelectedTaskId('')
      setSelectedTaskLog('')
    }
  }, [clearBatchPoll])

  const handleBatchStop = useCallback(async () => {
    const bid = batchIdRef.current
    if (!bid) return
    try {
      await stopBatchPerf(bid)
      toast.info(t('perf.batchStopping'))
    } catch (e) {
      toast.error(String(e))
    }
  }, [t])

  const monitorBatch = useCallback((batchId: string) => {
    clearBatchPoll()
    const generation = pollGenerationRef.current
    const isCurrent = () => generation === pollGenerationRef.current
    const schedule = (delay: number) => {
      if (isCurrent()) pollRef.current = setTimeout(poll, delay)
    }
    const poll = async () => {
      try {
        const st = await getBatchStatus(batchId)
        if (!isCurrent()) return
        setBatchState(st)
        const active = isActiveTaskStatus(st.status)
        setBatchRunning(active)
        if (st.current_task_id) {
          lastLogTaskIdRef.current = st.current_task_id
          try {
            const log = await getPerfLog(st.current_task_id)
            if (isCurrent()) setBatchLogText(log.text || '')
          } catch { /* status remains authoritative */ }
        }
        if (!active) {
          clearBatchPoll()
          const results = st.results || []
          if (results.length > 0) {
            const last = results[results.length - 1]
            setSelectedTaskId(last.task_id)
            void fetchTaskLog(last.task_id)
          }
          if (st.status === TASK_STATUSES.COMPLETED) {
            sessionStorage.removeItem('perfBatchId')
            toast.success(t('perf.batchCompleted', { n: st.completed }))
          } else if (st.status === TASK_STATUSES.PARTIAL_SUCCESS) {
            sessionStorage.removeItem('perfBatchId')
            toast.warning(t('perf.batchPartial', { completed: st.completed, errors: st.errors }))
          } else if (st.status === TASK_STATUSES.FAILED) {
            toast.error(t('perf.batchFailed'))
          } else if (st.status === TASK_STATUSES.STOPPED) {
            toast.info(t('perf.batchStopped', { n: st.completed }))
          }
          return
        }
        schedule(3000)
      } catch {
        if (isCurrent()) {
          toast.warning(t('perf.batchStatusRetry'))
          schedule(5000)
        }
      }
    }
    void poll()
  }, [clearBatchPoll, fetchTaskLog, t])

  useEffect(() => {
    const batchId = sessionStorage.getItem('perfBatchId')
    if (!batchId) return
    batchIdRef.current = batchId
    monitorBatch(batchId)
  }, [monitorBatch, t])

  const handleBatchSubmit = useCallback(async (batchId: string, sharedConfig: Record<string, unknown>) => {
    setBatchRunning(true)
    setBatchState(null)
    setBatchLogText('')
    setSelectedTaskId('')
    setSelectedTaskLog('')
    batchIdRef.current = batchId
    batchConfigRef.current = sharedConfig
    sessionStorage.setItem('perfBatchId', batchId)

    try {
      const launched = await launchBatchPerf(batchId, sharedConfig)
      toast.info(t('perf.batchStarted', { n: launched.total }))
      monitorBatch(batchId)
    } catch (e) {
      toast.error(String(e))
      setBatchRunning(false)
    }
  }, [monitorBatch, t])

  const handleBatchResume = useCallback(async (file: File, sharedConfig: Record<string, unknown>) => {
    const batchId = batchIdRef.current
    if (!batchId) return
    try {
      const uploaded = await uploadBatchCsv(file)
      await resumeBatchPerf(batchId, uploaded.batch_id, sharedConfig)
      batchFileRef.current = file
      batchConfigRef.current = sharedConfig
      setBatchRunning(true)
      toast.info(t('perf.batchResumed'))
      monitorBatch(batchId)
    } catch (e) {
      toast.error(String(e))
    }
  }, [monitorBatch, t])

  useEffect(() => () => {
    logGenerationRef.current += 1
    clearBatchPoll()
  }, [clearBatchPoll])

  return (
    <TaskPageLayout
      title={t('perf.title')}
      configTitle={t('perf.config')}
      statusTitle={t('perf.status')}
      readyLabel={t('perf.ready')}
      running={running || batchRunning}
      progress={running ? progress : 0}
      progressError={running ? progressError : null}
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
        onBatchResume={handleBatchResume}
        batchResumable={Boolean(batchState?.status === TASK_STATUSES.STOPPED && batchState.resumable)}
        onModeChange={handleModeChange}
      />

      {/* Batch progress */}
      {batchRunning && batchState && batchState.status === 'running' && (
        <div className="mt-4 p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-card2)]">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium">
              {t('perf.batchProgress', { completed: batchState.completed, total: batchState.total })}
            </h3>
            <button
              onClick={handleBatchStop}
              className="px-2 py-1 text-xs rounded border border-[var(--danger)] text-[var(--danger)] hover:bg-[var(--danger)]/10 transition-colors"
            >
              {t('perf.stop')}
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
              {t('perf.currentModel', { model: batchState.current_model })}
            </p>
          )}
          {batchState.errors > 0 && (
            <p className="text-xs text-[var(--danger)] mt-1">{t('perf.failureCount', { n: batchState.errors })}</p>
          )}
        </div>
      )}

      {/* Batch result summary */}
      {batchState && !isActiveTaskStatus(batchState.status) && (
        <div className="mt-4 p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-card2)]">
          <h3 className="text-sm font-medium mb-2">
            {batchState.status === TASK_STATUSES.COMPLETED
              ? t('perf.batchResultCompleted')
              : batchState.status === TASK_STATUSES.PARTIAL_SUCCESS
                ? t('perf.batchResultPartial')
                : batchState.status === TASK_STATUSES.FAILED
                  ? t('perf.batchResultFailed')
                  : t('perf.batchResultStopped')}：
            {t('perf.successCount', { n: batchState.completed })}
            {batchState.errors > 0 && <span className="text-[var(--danger)]">，{t('perf.failureCount', { n: batchState.errors })}</span>}
          </h3>
          <p className="text-xs text-[var(--text-muted)] mb-2">{t('perf.clickModelLogs')}</p>
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
                {normalizeTaskStatus(r.status) === TASK_STATUSES.FAILED ? (
                  <span className="text-[var(--danger)]">✗</span>
                ) : (
                  <span className="text-[var(--green)]">✓</span>
                )}
                <span className="text-[var(--text)]">{r.name}</span>
                <span className="text-[var(--text-muted)]">({r.model})</span>
                {normalizeTaskStatus(r.status) === TASK_STATUSES.FAILED && r.error && (
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
