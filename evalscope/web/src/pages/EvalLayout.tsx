import { useCallback, useMemo, useRef } from 'react'
import { Outlet, NavLink, useLocation, Navigate } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import TaskPageLayout from '@/components/eval/TaskPageLayout'
import { useTaskRunner } from '@/hooks/useTaskRunner'
import { submitEvalTask, stopEvalTask, getEvalProgress, getEvalLog, getEvalReportUrl, resumeEvalTask } from '@/api/eval'

type EvalMode = 'llm' | 'rag' | 'aigc' | 'audio'

export interface EvalTabContext {
  onSubmit: ReturnType<typeof useTaskRunner>['handleSubmit']
  disabled: boolean
  onApiKeyChange: (key: string) => void
  initialDataset: string | null
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

  // Derive current mode from URL path
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

  const onApiKeyChange = useCallback((key: string) => {
    apiKeyRef.current = key
  }, [])

  const handleResume = useCallback(
    (id: string) => {
      rawResume(id, apiKeyRef.current || undefined)
    },
    [rawResume],
  )

  // If at /eval without a mode, redirect to /eval/llm
  if (location.pathname === '/eval') {
    return <Navigate to="/eval/llm" replace />
  }

  const context: EvalTabContext = {
    onSubmit: handleSubmit,
    disabled: running,
    onApiKeyChange,
    initialDataset,
  }

  return (
    <TaskPageLayout
      title={t('eval.title')}
      configTitle={t('eval.config')}
      statusTitle={t('eval.status')}
      readyLabel={t('eval.ready')}
      running={running}
      progress={progress}
      result={result}
      logText={logText}
      reportUrl={reportUrl}
      copied={copied}
      onCopy={copyLog}
      onStop={handleStop}
      onResume={handleResume}
      taskId={taskId}
      sseState={sseState}
    >
      {/* Eval Mode Selector */}
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
