import { useCallback, useMemo, useRef } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import PerfConfigForm from '@/components/perf/PerfConfigForm'
import TaskPageLayout from '@/components/eval/TaskPageLayout'
import { useTaskRunner } from '@/hooks/useTaskRunner'
import { submitPerfTask, stopPerfTask, getPerfProgress, getPerfLog, getPerfReportUrl, resumePerfTask } from '@/api/perf'

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

  return (
    <TaskPageLayout
      title={t('perf.title')}
      configTitle={t('perf.config')}
      statusTitle={t('perf.status')}
      readyLabel={t('perf.ready')}
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
      <PerfConfigForm onSubmit={handleSubmit} disabled={running} onApiKeyChange={onApiKeyChange} />
    </TaskPageLayout>
  )
}
