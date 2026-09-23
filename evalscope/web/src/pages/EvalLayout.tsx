import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Outlet, NavLink, useLocation, Navigate } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import TaskPageLayout from '@/components/eval/TaskPageLayout'
import { useTaskRunner } from '@/hooks/useTaskRunner'
import { submitEvalTask, stopEvalTask, getEvalProgress, getEvalLog, getEvalReportUrl, resumeEvalTask,
  launchEvalBatch, resumeEvalBatch, getEvalBatchStatus, stopEvalBatch, uploadEvalBatchCsv, launchEvalTask } from '@/api/eval'
import type { EvalBatchStatus } from '@/api/eval'
import { isActiveTaskStatus, normalizeTaskStatus, TASK_STATUSES } from '@/hooks/taskLifecycle'
import { toast } from '@/components/common/Toast'

type EvalMode = 'llm' | 'rag' | 'aigc' | 'audio'

export interface EvalTabContext {
  onSubmit: ReturnType<typeof useTaskRunner>['handleSubmit']
  disabled: boolean
  onApiKeyChange: (key: string) => void
  initialDataset: string | null
  evalMode: EvalMode
  // Batch
  isBatch: boolean
  batchRunning: boolean
  batchState: EvalBatchStatus | null
  batchInfo: { batch_id: string; model_count: number; models: string[] } | null
  batchError: string
  batchUploading: boolean
  selectedTaskId: string
  onSelectTask: (taskId: string) => void
  onBatchSubmit: (batchId: string, sharedConfig: Record<string, unknown>) => void
  onBatchResume: (file: File, sharedConfig: Record<string, unknown>) => Promise<void>
  onBatchStop: () => void
  onBatchUpload: (file: File) => Promise<void>
  setBatchMode: (v: boolean) => void
}

const MODES: { mode: EvalMode; label: string }[] = [
  { mode: 'llm', label: 'eval.evalModeLLM' },
  { mode: 'rag', label: 'eval.evalModeRAG' },
  { mode: 'aigc', label: 'eval.evalModeAIGC' },
  { mode: 'audio', label: 'eval.evalModeAudio' },
]

export default function EvalLayout() {
  const { t } = useLocale()
  const location = useLocation()
  const queryParams = useQueryParams()
  const initialDataset = queryParams.get('dataset') ?? null
  const apiKeyRef = useRef('')

  const segments = location.pathname.split('/')
  const evalMode = (segments[segments.length - 1] || 'llm') as EvalMode

  const api = useMemo(
    () => ({
      submit: submitEvalTask,
      launch: launchEvalTask,
      stop: stopEvalTask,
      getProgress: getEvalProgress,
      getLog: getEvalLog,
      getReportUrl: (taskId: string) =>
        evalMode === 'aigc'
          ? `/reports/aigc/${encodeURIComponent(taskId)}`
          : evalMode === 'audio'
            ? `/reports/audio/${encodeURIComponent(taskId)}`
            : getEvalReportUrl(taskId),
      resume: resumeEvalTask,
    }),
    [evalMode],
  )

  const {
    running,
    progress,
    progressError,
    result,
    logText,
    reportUrl,
    copied,
    taskId,
    handleSubmit,
    handleStop,
    handleResume: rawResume,
    sseState,
  } = useTaskRunner({ api, taskPrefix: 'eval' })

  const onApiKeyChange = useCallback((key: string) => { apiKeyRef.current = key }, [])
  const handleResume = useCallback((id: string) => { rawResume(id, apiKeyRef.current || undefined) }, [rawResume])

  // ── Batch state ──
  const [isBatch, setIsBatch] = useState(() => Boolean(sessionStorage.getItem('evalBatchId')))
  const [batchRunning, setBatchRunning] = useState(false)
  const [batchState, setBatchState] = useState<EvalBatchStatus | null>(null)
  const [batchLogText, setBatchLogText] = useState('')
  const [selectedTaskId, setSelectedTaskId] = useState('')
  const [selectedTaskLog, setSelectedTaskLog] = useState('')
  const batchIdRef = useRef<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollGenerationRef = useRef(0)
  const logGenerationRef = useRef(0)
  const lastLogTaskIdRef = useRef<string>('')
  const batchFileRef = useRef<File | null>(null)
  const batchConfigRef = useRef<Record<string, unknown>>({})
  const [batchInfo, setBatchInfo] = useState<{ batch_id: string; model_count: number; models: string[] } | null>(null)
  const [batchError, setBatchError] = useState('')
  const [batchUploading, setBatchUploading] = useState(false)

  const clearBatchPoll = useCallback(() => {
    pollGenerationRef.current += 1
    if (pollRef.current) {
      clearTimeout(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const fetchTaskLog = useCallback(async (tid: string) => {
    const generation = ++logGenerationRef.current
    try {
      const log = await getEvalLog(tid)
      if (generation === logGenerationRef.current) {
        setSelectedTaskLog(log.text || '')
      }
    } catch {
      if (generation === logGenerationRef.current) {
        setSelectedTaskLog('')
      }
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

  const handleBatchUpload = useCallback(async (file: File) => {
    batchFileRef.current = file
    setBatchUploading(true)
    setBatchError('')
    try {
      const info = await uploadEvalBatchCsv(file)
      setBatchInfo({ batch_id: info.batch_id, model_count: info.model_count, models: info.models })
    } catch (e) {
      setBatchError(String(e))
      setBatchInfo(null)
    } finally {
      setBatchUploading(false)
    }
  }, [])

  const monitorBatch = useCallback((batchId: string) => {
    clearBatchPoll()
    const generation = pollGenerationRef.current
    const isCurrent = () => generation === pollGenerationRef.current

    const poll = async () => {
      try {
        const state = await getEvalBatchStatus(batchId)
        if (!isCurrent()) return

        const status = normalizeTaskStatus(state.status)
        const active = isActiveTaskStatus(status)
        setBatchState(state)
        setBatchRunning(active)

        if (state.current_task_id) {
          lastLogTaskIdRef.current = state.current_task_id
          try {
            const log = await getEvalLog(state.current_task_id)
            if (!isCurrent()) return
            setBatchLogText(log.text || '')
          } catch {
            // The batch status is authoritative; a transient log failure must not stop polling.
          }
        }

        if (!isCurrent()) return
        if (active) {
          pollRef.current = setTimeout(poll, 3000)
          return
        }

        clearBatchPoll()
        const results = state.results || []
        if (results.length > 0) {
          const lastTaskId = results[results.length - 1].task_id
          setSelectedTaskId(lastTaskId)
          void fetchTaskLog(lastTaskId)
        }

        if (status === TASK_STATUSES.COMPLETED) {
          sessionStorage.removeItem('evalBatchId')
          toast.success(`批量评估完成：${state.completed} 个模型全部成功`)
        } else if (status === TASK_STATUSES.PARTIAL_SUCCESS) {
          sessionStorage.removeItem('evalBatchId')
          toast.warning(`批量评估部分完成：${state.completed} 成功，${state.errors} 失败`)
        } else if (status === TASK_STATUSES.FAILED || status === TASK_STATUSES.ORPHANED) {
          toast.error('批量评估失败，请查看错误详情')
        } else if (status === TASK_STATUSES.STOPPED) {
          toast.info(`批量评估已停止：${state.completed} 完成，可从断点继续`)
        }
      } catch {
        if (isCurrent()) {
          toast.warning('批量状态暂不可用，正在重试')
          pollRef.current = setTimeout(poll, 5000)
        }
      }
    }

    void poll()
  }, [clearBatchPoll, fetchTaskLog])

  const onBatchSubmit = useCallback(async (batchId: string, sharedConfig: Record<string, unknown>) => {
    setBatchRunning(true)
    setBatchState(null)
    setBatchLogText('')
    setSelectedTaskId('')
    setSelectedTaskLog('')
    batchIdRef.current = batchId
    batchConfigRef.current = sharedConfig
    sessionStorage.setItem('evalBatchId', batchId)

    try {
      const launched = await launchEvalBatch(batchId, sharedConfig)
      toast.info(`批量评估已启动，共 ${launched.total} 个模型`)
      monitorBatch(batchId)
    } catch (e) {
      toast.error(String(e))
      setBatchRunning(false)
    }
  }, [monitorBatch])

  const onBatchResume = useCallback(async (file: File, sharedConfig: Record<string, unknown>) => {
    const batchId = batchIdRef.current
    if (!batchId) return
    setBatchUploading(true)
    setBatchError('')
    try {
      const uploaded = await uploadEvalBatchCsv(file)
      await resumeEvalBatch(batchId, uploaded.batch_id, sharedConfig)
      batchFileRef.current = file
      batchConfigRef.current = sharedConfig
      setBatchRunning(true)
      toast.info('批量评估已从断点继续')
      monitorBatch(batchId)
    } catch (e) {
      setBatchError(String(e))
      toast.error(String(e))
    } finally {
      setBatchUploading(false)
    }
  }, [monitorBatch])

  useEffect(() => {
    const batchId = sessionStorage.getItem('evalBatchId')
    if (!batchId) return
    batchIdRef.current = batchId
    monitorBatch(batchId)
  }, [monitorBatch])

  useEffect(() => () => {
    logGenerationRef.current += 1
    clearBatchPoll()
  }, [clearBatchPoll])

  const handleSelectTask = useCallback((tid: string) => {
    setSelectedTaskId(tid)
    void fetchTaskLog(tid)
  }, [fetchTaskLog])

  const onBatchStop = useCallback(async () => {
    const batchId = batchIdRef.current
    if (!batchId) return
    try {
      await stopEvalBatch(batchId)
      toast.info('正在停止批量评估...')
    } catch (e) {
      toast.error(String(e))
    }
  }, [])

  const setBatchMode = useCallback((value: boolean) => {
    setIsBatch(value)
    if (!value) {
      setBatchInfo(null)
      setBatchError('')
      setBatchLogText('')
      setBatchState(null)
      setSelectedTaskId('')
      setSelectedTaskLog('')
    }
  }, [])

  if (location.pathname === '/eval') {
    return <Navigate to="/eval/llm" replace />
  }

  const context: EvalTabContext = {
    onSubmit: handleSubmit,
    disabled: running || batchRunning,
    onApiKeyChange,
    initialDataset,
    evalMode,
    isBatch,
    batchRunning,
    batchState,
    batchInfo,
    batchError,
    batchUploading,
    selectedTaskId,
    onSelectTask: handleSelectTask,
    onBatchSubmit,
    onBatchResume,
    onBatchStop,
    onBatchUpload: handleBatchUpload,
    setBatchMode,
  }

  return (
    <TaskPageLayout
      title={t('eval.title')}
      configTitle={t('eval.config')}
      statusTitle={t('eval.status')}
      readyLabel={t('eval.ready')}
      running={running || batchRunning}
      progress={running ? progress : 0}
      progressError={running ? progressError : null}
      result={result}
      logText={running ? logText : (batchRunning ? batchLogText : (selectedTaskId ? selectedTaskLog : logText))}
      reportUrl={reportUrl}
      copied={copied}
      onCopy={copyCurrentLog}
      onStop={running ? handleStop : onBatchStop}
      onResume={handleResume}
      taskId={taskId}
      sseState={sseState}
    >
      <div className="flex items-center gap-4 mb-4 pb-4 border-b border-[var(--border-md)]">
        <span className="text-sm font-medium text-[var(--text)]">{t('eval.evalMode')}</span>
        <div className="flex gap-1 rounded-lg bg-[var(--bg-card2)] p-1">
          {MODES.map(({ mode, label }) => (
            <NavLink
              key={mode}
              to={`/eval/${mode}`}
              className={({ isActive }) =>
                `px-3 py-1.5 text-sm rounded-md transition-colors ${
                  isActive
                    ? 'bg-[var(--accent)] text-white shadow-sm'
                    : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card)]'
                }`
              }
            >
              {t(label)}
            </NavLink>
          ))}
        </div>
      </div>

      <Outlet context={context} />
    </TaskPageLayout>
  )
}
