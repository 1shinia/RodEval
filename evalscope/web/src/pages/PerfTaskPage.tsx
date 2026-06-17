import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import PerfConfigForm from '@/components/perf/PerfConfigForm'
import TaskPageLayout from '@/components/eval/TaskPageLayout'
import { useTaskRunner } from '@/hooks/useTaskRunner'
import { submitPerfTask, stopPerfTask, getPerfProgress, getPerfLog, getPerfReportUrl } from '@/api/perf'

const perfApi = {
  submit: submitPerfTask,
  stop: stopPerfTask,
  getProgress: getPerfProgress,
  getLog: getPerfLog,
  getReportUrl: getPerfReportUrl,
}

export default function PerfTaskPage() {
  const { t } = useLocale()

  const api = useMemo(() => perfApi, [])
  const { running, progress, result, logText, reportUrl, copied,
    handleSubmit, handleStop, copyLog } = useTaskRunner({ api, taskPrefix: 'perf' })

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
    >
      <PerfConfigForm onSubmit={handleSubmit} disabled={running} />
    </TaskPageLayout>
  )
}
