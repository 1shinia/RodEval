import { useCallback, useMemo, useRef } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import EvalConfigForm from '@/components/eval/EvalConfigForm'
import TaskPageLayout from '@/components/eval/TaskPageLayout'
import { useTaskRunner } from '@/hooks/useTaskRunner'
import { submitEvalTask, stopEvalTask, getEvalProgress, getEvalLog, getEvalReportUrl, resumeEvalTask } from '@/api/eval'

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

  const api = useMemo(() => evalApi, [])
  const { running, progress, result, logText, reportUrl, copied, taskId,
    handleSubmit, handleStop, handleResume: rawResume, copyLog } = useTaskRunner({ api, taskPrefix: 'eval' })

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
    >
      <EvalConfigForm onSubmit={handleSubmit} disabled={running} initialDataset={initialDataset} onApiKeyChange={onApiKeyChange} />
    </TaskPageLayout>
  )
}
