import { useCallback, useMemo, useRef, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import EvalConfigForm from '@/components/eval/EvalConfigForm'
import RAGEvalForm from '@/components/eval/RAGEvalForm'
import TaskPageLayout from '@/components/eval/TaskPageLayout'
import { useTaskRunner } from '@/hooks/useTaskRunner'
import { submitEvalTask, stopEvalTask, getEvalProgress, getEvalLog, getEvalReportUrl, resumeEvalTask } from '@/api/eval'

type EvalMode = 'llm' | 'embedding' | 'reranker'

const evalApi = {
  submit: submitEvalTask,
  stop: stopEvalTask,
  getProgress: getEvalProgress,
  getLog: getEvalLog,
  getReportUrl: getEvalReportUrl,
  resume: resumeEvalTask,
}

export default function EvalTaskPage() {
  const { t } = useLocale()
  const queryParams = useQueryParams()
  const initialDataset = queryParams.get('dataset')
  const apiKeyRef = useRef('')
  const [evalMode, setEvalMode] = useState<EvalMode>('llm')

  const api = useMemo(() => evalApi, [])
  const { running, progress, result, logText, reportUrl, copied, taskId,
    handleSubmit, handleStop, handleResume: rawResume, copyLog, sseState } = useTaskRunner({ api, taskPrefix: 'eval' })

  const onApiKeyChange = useCallback((key: string) => { apiKeyRef.current = key }, [])
  const handleResume = useCallback((id: string) => { rawResume(id, apiKeyRef.current || undefined) }, [rawResume])

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
          {([
            ['llm', 'eval.evalModeLLM'],
            ['embedding', 'eval.evalModeEmbedding'],
            ['reranker', 'eval.evalModeReranker'],
          ] as const).map(([mode, label]) => (
            <button key={mode} type="button"
              onClick={() => setEvalMode(mode)}
              className={`px-3 py-1.5 text-sm rounded-md transition-colors cursor-pointer ${
                evalMode === mode
                  ? 'bg-[var(--accent)] text-white shadow-sm'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card)]'
              }`}>
              {t(label)}
            </button>
          ))}
        </div>
      </div>

      {evalMode === 'llm' && (
        <EvalConfigForm onSubmit={handleSubmit} disabled={running} initialDataset={initialDataset} onApiKeyChange={onApiKeyChange} />
      )}
      {evalMode === 'embedding' && (
        <RAGEvalForm onSubmit={handleSubmit} disabled={running} evalMode="embedding" />
      )}
      {evalMode === 'reranker' && (
        <RAGEvalForm onSubmit={handleSubmit} disabled={running} evalMode="reranker" />
      )}
    </TaskPageLayout>
  )
}
