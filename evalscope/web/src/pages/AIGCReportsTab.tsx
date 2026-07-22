import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, Image as ImageIcon } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import Skeleton from '@/components/ui/Skeleton'
import { EmptyState } from '@/pages/ReportsLayout'

interface AIGCReportSummary {
  task_id: string
  model_name: string
  model_type: string
  total_images: number
  clip_score_mean?: number
  fid?: number
  inception_score?: number
  created_at: string
}

export default function AIGCReportsTab() {
  const { t } = useLocale()
  const navigate = useNavigate()

  // ---- AIGC reports state ----
  const [aigcReports, setAigcReports] = useState<AIGCReportSummary[]>([])
  const [aigcLoading, setAigcLoading] = useState(false)
  const [aigcError, setAigcError] = useState<string | null>(null)

  // Fetch AIGC reports
  const fetchAIGCReports = useCallback(async () => {
    setAigcLoading(true)
    setAigcError(null)
    try {
      const response = await fetch('/api/v1/aigc/reports')
      if (!response.ok) {
        throw new Error(`Failed to load reports: ${response.statusText}`)
      }
      const data = await response.json()
      setAigcReports(data.reports || [])
    } catch (err) {
      setAigcError(err instanceof Error ? err.message : 'Failed to load reports')
    } finally {
      setAigcLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAIGCReports()
  }, [fetchAIGCReports])

  return (
    <>
      {aigcError && (
        <div className="px-4 py-3 rounded-[var(--radius)] bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger)]">
          {aigcError}
        </div>
      )}

      {aigcLoading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height={64} className="rounded-[var(--radius)]" />
          ))}
        </div>
      ) : aigcReports.length === 0 ? (
        <EmptyState icon={<ImageIcon size={40} />} title={t('aigc.noReports')} subtitle={t('aigc.noReportsHint')} />
      ) : (
        <div className="rounded-[var(--radius)] border border-[var(--border)] overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-[var(--bg-deep)] border-b border-[var(--border)]">
                <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.taskId')}</th>
                <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.modelName')}</th>
                <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.modelType')}</th>
                <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.totalImages')}</th>
                <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.clipScoreMean')}</th>
                <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.fid')}</th>
                <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.createdAt')}</th>
                <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">{t('common.actions')}</th>
              </tr>
            </thead>
            <tbody>
              {aigcReports.map((report) => (
                <tr key={report.task_id} className="border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--bg-card2)] transition-colors">
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)]">{report.task_id}</td>
                  <td className="px-4 py-3 text-[var(--text)]">{report.model_name}</td>
                  <td className="px-4 py-3 text-[var(--text-muted)]">{report.model_type}</td>
                  <td className="px-4 py-3 text-right text-[var(--text)]">{report.total_images}</td>
                  <td className="px-4 py-3 text-right font-mono">
                    {report.clip_score_mean != null ? (
                      <span className="text-[var(--accent)]">{report.clip_score_mean.toFixed(4)}</span>
                    ) : (
                      <span className="text-[var(--text-dim)]">-</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right font-mono">
                    {report.fid != null ? (
                      <span className="text-[var(--text)]">{report.fid.toFixed(2)}</span>
                    ) : (
                      <span className="text-[var(--text-dim)]">-</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-[var(--text-muted)] text-xs">
                    {new Date(report.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => navigate(`/reports/aigc/${encodeURIComponent(report.task_id)}`)}
                    >
                      <Eye size={14} />
                      {t('common.view')}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}
