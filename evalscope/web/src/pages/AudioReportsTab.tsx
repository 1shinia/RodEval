import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, Mic } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import Skeleton from '@/components/ui/Skeleton'
import { EmptyState } from '@/pages/ReportsLayout'

interface AudioReportSummary {
  task_id: string
  tool: string
  model_name: string
  wer?: number
  cer?: number
  reference?: string
  hypothesis?: string
  num_samples?: number
  total_elapsed?: number
  language?: string
  created_at: string
}

export default function AudioReportsTab() {
  const { t } = useLocale()
  const navigate = useNavigate()

  const [reports, setReports] = useState<AudioReportSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchReports = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/v1/audio/reports')
      if (!response.ok) {
        throw new Error(`Failed to load reports: ${response.statusText}`)
      }
      const data = await response.json()
      setReports(data.reports || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load reports')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchReports()
  }, [fetchReports])

  return (
    <>
      {error && (
        <div className="px-4 py-3 rounded-[var(--radius)] bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger)]">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full rounded-lg" />
          ))}
        </div>
      ) : reports.length === 0 ? (
        <EmptyState
          icon={<Mic size={40} className="text-[var(--text-muted)]" />}
          title="暂无音频评估报告"
          subtitle="完成 ASR 或 TTS 评估后，报告将显示在这里"
        />
      ) : (
        <div className="rounded-[var(--radius)] border border-[var(--border)] overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--border)] bg-[var(--bg-card2)] text-left">
                <th className="px-4 py-3 font-medium text-[var(--text)]">任务 ID</th>
                <th className="px-4 py-3 font-medium text-[var(--text)]">功能</th>
                <th className="px-4 py-3 font-medium text-[var(--text)]">模型</th>
                <th className="px-4 py-3 font-medium text-[var(--text)]">评估得分</th>
                <th className="px-4 py-3 font-medium text-[var(--text)]">创建时间</th>
                <th className="px-4 py-3 font-medium text-[var(--text)] w-20">操作</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((r) => (
                <tr key={r.task_id} className="border-b border-[var(--border)] hover:bg-[var(--bg-card2)]">
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)] max-w-[180px] truncate" title={r.task_id}>
                    {r.task_id.length > 24 ? r.task_id.slice(0, 24) + '...' : r.task_id}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      r.tool === 'asr'
                        ? 'bg-blue-500/10 text-blue-400'
                        : 'bg-green-500/10 text-green-400'
                    }`}>
                      {r.tool === 'asr' ? 'ASR' : 'TTS'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-[var(--text)]">{r.model_name}</td>
                  <td className="px-4 py-3 text-right font-mono text-xs">
                    <div className="flex flex-col gap-0.5 items-end">
                      {r.wer != null && (
                        <span className="text-[var(--accent)]">WER: {r.wer.toFixed(4)}</span>
                      )}
                      {r.cer != null && (
                        <span className="text-[var(--text)]">CER: {r.cer.toFixed(4)}</span>
                      )}
                      {r.num_samples != null && (
                        <span className="text-[var(--text-dim)]">{r.num_samples} 条样本</span>
                      )}
                      {r.wer == null && r.cer == null && r.num_samples == null && (
                        <span className="text-[var(--text-dim)]">-</span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-[var(--text-muted)]">
                    {r.created_at ? new Date(r.created_at).toLocaleString() : '-'}
                  </td>
                  <td className="px-4 py-3">
                    <Button variant="ghost" size="sm" onClick={() => navigate(`/reports/audio/${encodeURIComponent(r.task_id)}`)}>
                      <Eye size={14} className="mr-1" />
                      查看
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
