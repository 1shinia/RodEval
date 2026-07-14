import type { EvalInvokeResponse } from '@/api/types'
import LogViewer from '@/components/common/LogViewer'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import { useLocale } from '@/contexts/LocaleContext'
import { ExternalLink, CheckCircle2, XCircle, Loader2, Square, OctagonX, PlayCircle, WifiOff } from 'lucide-react'

interface Props {
  running: boolean
  progress: number
  logText: string
  result: EvalInvokeResponse | null
  reportUrl: string | null
  readyLabel: string
  onStop?: () => void
  onResume?: (taskId: string) => void
  taskId?: string | null
  sseState?: { status: string; message: string }
}

export default function TaskMonitor({ running, progress, logText, result, reportUrl, readyLabel, onStop, onResume, taskId, sseState }: Props) {
  const { t } = useLocale()
  const isReconnecting = sseState?.status === 'reconnecting'

  return (
    <div className="space-y-4">
      {/* Status */}
      <div className="flex items-center gap-2 text-sm">
        {running && (
          <>
            <Loader2 size={16} className="animate-spin text-[var(--accent)]" />
            <Badge variant="warning">{t('eval.statusRunning')}{progress > 0 ? ` ${Math.round(progress)}%` : '...'}</Badge>
          </>
        )}
        {!running && result?.status === 'stopped' && (
          <>
            <OctagonX size={16} className="text-[var(--yellow)]" />
            <Badge variant="warning">{t('eval.statusStopped')}</Badge>
          </>
        )}
        {!running && result?.status === 'error' && (
          <>
            <XCircle size={16} className="text-[var(--danger)]" />
            <Badge variant="danger">{result.error}</Badge>
          </>
        )}
        {!running && result && result.status !== 'error' && result.status !== 'stopped' && (
          <>
            <CheckCircle2 size={16} className="text-[var(--green)]" />
            <Badge variant="success">{t('eval.statusCompleted')}</Badge>
          </>
        )}
        {!running && !result && (
          <span className="text-[var(--text-muted)]">{readyLabel}</span>
        )}

        {/* SSE connection state indicator */}
        {isReconnecting && (
          <div className="flex items-center gap-1.5 ml-2 px-2 py-0.5 rounded bg-[var(--warning-bg)] text-[var(--warning-color)] text-xs">
            <WifiOff size={12} />
            <span>{sseState!.message}</span>
          </div>
        )}
      </div>

      {/* Progress bar */}
      {(running || (result && result.status !== 'error' && result.status !== 'stopped')) && (
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-deep)' }}>
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${Math.min(progress, 100)}%`, background: 'var(--accent)' }}
          />
        </div>
      )}

      {/* Stop button */}
      {running && onStop && (
        <Button
          variant="outline"
          size="sm"
          onClick={onStop}
          className="text-[var(--danger)] border-[var(--danger)] hover:bg-[var(--danger-bg)]"
        >
          <Square size={14} />
          {t('common.stop')}
        </Button>
      )}

      {/* Resume button */}
      {!running && onResume && taskId && result && (result.status === 'stopped' || result.status === 'error') && (
        <Button
          variant="primary"
          size="sm"
          onClick={() => onResume(taskId)}
          className="btn-glow"
        >
          <PlayCircle size={14} />
          续跑
        </Button>
      )}

      {/* Log */}
      {logText && <LogViewer content={logText} />}

      {/* Report link */}
      {!running && result && result.status !== 'error' && result.status !== 'stopped' && reportUrl && (
        <Button
          variant="primary"
          size="sm"
          onClick={() => window.open(reportUrl, '_blank')}
          className="btn-glow"
        >
          <ExternalLink size={14} />
          {t('common.openNewTab')}
        </Button>
      )}
    </div>
  )
}
