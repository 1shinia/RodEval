import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import EvalConfigForm from '@/components/eval/EvalConfigForm'
import TaskPageLayout from '@/components/eval/TaskPageLayout'
import { useTaskRunner } from '@/hooks/useTaskRunner'
import { submitEvalTask, stopEvalTask, getEvalProgress, getEvalLog, getEvalReportUrl } from '@/api/eval'

const evalApi = {
  submit: submitEvalTask,
  stop: stopEvalTask,
  getProgress: getEvalProgress,
  getLog: getEvalLog,
  getReportUrl: getEvalReportUrl,
}

export default function EvalTaskPage() {
  const { t } = useLocale()
  const queryParams = useQueryParams()
  const initialDataset = queryParams.get('dataset')

  const api = useMemo(() => evalApi, [])
  const { running, progress, result, logText, reportUrl, copied,
    handleSubmit, handleStop, copyLog } = useTaskRunner({ api, taskPrefix: 'eval' })

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
    >
      <EvalConfigForm onSubmit={handleSubmit} disabled={running} initialDataset={initialDataset} />
    </TaskPageLayout>
  )
}
