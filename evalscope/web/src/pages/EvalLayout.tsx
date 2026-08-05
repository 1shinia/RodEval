import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Outlet, NavLink, useLocation, Navigate } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import TaskPageLayout from '@/components/eval/TaskPageLayout'
import { useTaskRunner } from '@/hooks/useTaskRunner'
import { submitEvalTask, stopEvalTask, getEvalProgress, getEvalLog, getEvalReportUrl, resumeEvalTask,
  launchEvalBatch, getEvalBatchStatus, stopEvalBatch, uploadEvalBatchCsv } from '@/api/eval'
import type { EvalBatchStatus } from '@/api/eval'
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
  const initialDataset = queryParams.get('dataset')
  const apiKeyRef = useRef('')

  const segments = location.pathname.split('/')
  const evalMode = (segments[segments.length - 1] || 'llm') as EvalMode

  const api = useMemo(
    () => ({
      submit: submitEvalTask,
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
    result,
    logText,
    reportUrl,
    copied,
    taskId,
    handleSubmit,
    handleStop,
    handleResume: rawResume,
    copyLog,
    sseState,
  } = useTaskRunner({ api, taskPrefix: 'eval' })

  const onApiKeyChange = useCallback((key: string) => { apiKeyRef.current = key }, [])
  const handleResume = useCallback((id: string) => { rawResume(id, apiKeyRef.current || undefined) }, [rawResume])

  // ── Batch state ──
  const [isBatch, setIsBatch] = useState(false)
  const [batchRunning, setBatchRunning] = useState(false)
  const [batchState, setBatchState] = useState<EvalBatchStatus | null>(null)
  const [batchLogText, setBatchLogText] = useState('')
  const [selectedTaskId, setSelectedTaskId] = useState('')
  const [selectedTaskLog, setSelectedTaskLog] = useState('')
  const batchIdRef = useRef<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const lastLogTaskIdRef = useRef<string>('')
  const batchFileRef = useRef<File | null>(null)
  const [batchInfo, setBatchInfo] = useState<{ batch_id: string; model_count: number; models: string[] } | null>(null)
  const [batchError, setBatchError] = useState('')
  const [batchUploading, setBatchUploading] = useState(false)

  const clearBatchPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  useEffect(() => () => clearBatchPoll(), [clearBatchPoll])

  const fetchTaskLog = useCallback(async (tid: string) => {
    try {
      const log = await getEvalLog(tid)
      setSelectedTaskLog(log.text || '')
    } catch { setSelectedTaskLog('') }
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
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0'
        document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta)
      }
      toast.success('日志已复制')
    } catch { toast.error('复制失败') }
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

  const onBatchSubmit = useCallback(async (batchId: string, sharedConfig: Record<string, unknown>) => {
    setBatchRunning(true)
    setBatchState(null)
    setBatchLogText('')
    setSelectedTaskId('')
    setSelectedTaskLog('')
    batchIdRef.current = batchId
    clearBatchPoll()

    try {
      const launched = await launchEvalBatch(batchId, sharedConfig)
      toast.info(`批量评估已启动，共 ${launched.total} 个模型`)

      pollRef.current = setInterval(async () => {
        try {
          const st = await getEvalBatchStatus(batchId)
          setBatchState(st)
          if (st.current_task_id && st.current_task_id !== lastLogTaskIdRef.current) {
            lastLogTaskIdRef.current = st.current_task_id
            setBatchLogText('')
          }
          if (st.current_task_id) {
            try { const log = await getEvalLog(st.current_task_id); setBatchLogText(log.text || '') } catch { /* */ }
          }
          if (st.status !== 'running') {
            clearBatchPoll(); setBatchRunning(false)
            const results = st.results || []
            if (results.length > 0) {
              setSelectedTaskId(results[results.length - 1].task_id)
              fetchTaskLog(results[results.length - 1].task_id)
            }
            if (st.status === 'completed') {
              if (st.errors > 0) toast.warning(`批量评估完成：${st.completed} 成功，${st.errors} 失败`)
              else toast.success(`批量评估完成：${st.completed} 个模型全部成功`)
            } else if (st.status === 'cancelled') toast.info(`批量评估已取消：${st.completed} 完成`)
          }
        } catch { /* */ }
      }, 3000)
    } catch (e) {
      toast.error(String(e)); setBatchRunning(false)
    }
  }, [clearBatchPoll, fetchTaskLog])

  const handleSelectTask = useCallback((tid: string) => {
    setSelectedTaskId(tid)
    fetchTaskLog(tid)
  }, [fetchTaskLog])

  const onBatchStop = useCallback(async () => {
    const bid = batchIdRef.current
    if (!bid) return
    try { await stopEvalBatch(bid); toast.info('正在停止批量评估...') } catch (e) { toast.error(String(e)) }
  }, [])

  const setBatchMode = useCallback((v: boolean) => setIsBatch(v), [])

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
