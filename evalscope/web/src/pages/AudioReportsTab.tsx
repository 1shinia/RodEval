import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, Mic, Trash2 } from 'lucide-react'
import { toast } from '@/components/common/Toast'
import Button from '@/components/ui/Button'
import Skeleton from '@/components/ui/Skeleton'
import ReportFiltersBar, { type ReportFilters } from '@/components/reports/ReportFilters'
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

const PAGE_SIZE = 20

const defaultFilters: ReportFilters = {
  search: '',
  models: [],
  datasets: [],
  scoreMin: 0,
  scoreMax: 1,
  sortBy: 'time',
  sortOrder: 'desc',
}

function primaryScore(r: AudioReportSummary): number {
  return r.wer ?? r.cer ?? 0
}

export default function AudioReportsTab() {
  const navigate = useNavigate()

  const [allReports, setAllReports] = useState<AudioReportSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState<ReportFilters>(defaultFilters)
  const [page, setPage] = useState(1)

  const fetchReports = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/v1/audio/reports')
      if (!response.ok) throw new Error(`Failed to load reports: ${response.statusText}`)
      const data = await response.json()
      setAllReports(data.reports || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load reports')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleDelete = useCallback(async (taskId: string) => {
    if (!window.confirm(`确定要删除此报告吗？\n\n${taskId}`)) return
    try {
      const response = await fetch(`/api/v1/audio/reports/${encodeURIComponent(taskId)}`, { method: 'DELETE' })
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error((data as any).error || `HTTP ${response.status}`)
      }
      toast.success('已删除')
      fetchReports()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '删除失败')
    }
  }, [fetchReports])

  useEffect(() => { fetchReports() }, [fetchReports])

  const availableModels = useMemo(
    () => [...new Set(allReports.map((r) => r.model_name).filter(Boolean))].sort(),
    [allReports],
  )
  const availableDatasets = useMemo(
    () => [...new Set(allReports.map((r) => r.tool).filter(Boolean))].sort(),
    [allReports],
  )

  const filtered = useMemo(() => {
    let list = [...allReports]
    const q = filters.search.toLowerCase()
    if (q) {
      list = list.filter((r) => r.task_id.toLowerCase().includes(q) || r.model_name.toLowerCase().includes(q))
    }
    if (filters.models.length) {
      const ms = new Set(filters.models.map((m) => m.toLowerCase()))
      list = list.filter((r) => ms.has(r.model_name.toLowerCase()))
    }
    if (filters.datasets.length) {
      const ds = new Set(filters.datasets.map((d) => d.toLowerCase()))
      list = list.filter((r) => ds.has(r.tool.toLowerCase()))
    }
    list = list.filter((r) => {
      const s = primaryScore(r)
      return s >= filters.scoreMin && s <= filters.scoreMax
    })
    const desc = filters.sortOrder === 'desc'
    list.sort((a, b) => {
      const da = new Date(a.created_at).getTime()
      const db = new Date(b.created_at).getTime()
      return desc ? db - da : da - db
    })
    return list
  }, [allReports, filters])

  const total = filtered.length
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))
  const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  useEffect(() => { setPage(1) }, [filters])

  return (
    <>
      {error && (
        <div className="px-4 py-3 rounded-[var(--radius)] bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger)]">{error}</div>
      )}

      <ReportFiltersBar
        filters={filters}
        availableModels={availableModels}
        availableDatasets={availableDatasets}
        onChange={setFilters}
        datasetLabel="类型"
      />

      {loading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-16 w-full rounded-lg" />)}
        </div>
      ) : allReports.length === 0 ? (
        <EmptyState icon={<Mic size={40} />} title="暂无音频评估报告" subtitle="完成 ASR 或 TTS 评估后，报告将显示在这里" />
      ) : paged.length === 0 ? (
        <div className="text-center text-[var(--text-muted)] py-8">无匹配结果</div>
      ) : (
        <>
          <div className="rounded-[var(--radius)] border border-[var(--border)] overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border)] bg-[var(--bg-card2)] text-left">
                  <th className="px-4 py-3 font-medium text-[var(--text)]">任务 ID</th>
                  <th className="px-4 py-3 font-medium text-[var(--text)]">类型</th>
                  <th className="px-4 py-3 font-medium text-[var(--text)]">模型</th>
                  <th className="px-4 py-3 font-medium text-[var(--text)]">评估得分</th>
                  <th className="px-4 py-3 font-medium text-[var(--text)]">创建时间</th>
                  <th className="px-4 py-3 font-medium text-[var(--text)] w-20">操作</th>
                </tr>
              </thead>
              <tbody>
                {paged.map((r) => (
                  <tr key={r.task_id} className="border-b border-[var(--border)] hover:bg-[var(--bg-card2)]">
                    <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)] max-w-[180px] truncate" title={r.task_id}>
                      {r.task_id.length > 24 ? r.task_id.slice(0, 24) + '...' : r.task_id}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${r.tool === 'asr' ? 'bg-blue-500/10 text-blue-400' : 'bg-green-500/10 text-green-400'}`}>
                        {r.tool === 'asr' ? 'ASR' : 'TTS'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-[var(--text)]">{r.model_name}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs">
                      <div className="flex flex-col gap-0.5 items-end">
                        {r.wer != null && <span className="text-[var(--accent)]">WER: {r.wer.toFixed(4)}</span>}
                        {r.cer != null && <span className="text-[var(--text)]">CER: {r.cer.toFixed(4)}</span>}
                        {r.num_samples != null && <span className="text-[var(--text-dim)]">{r.num_samples} 条样本</span>}
                        {r.wer == null && r.cer == null && r.num_samples == null && <span className="text-[var(--text-dim)]">-</span>}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-[var(--text-muted)]">{r.created_at ? new Date(r.created_at).toLocaleString() : '-'}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1">
                        <Button variant="ghost" size="sm" onClick={() => navigate(`/reports/audio/${encodeURIComponent(r.task_id)}`)}>
                          <Eye size={14} className="mr-1" />查看
                        </Button>
                        <button type="button" onClick={(e) => { e.stopPropagation(); handleDelete(r.task_id) }} className="p-1.5 rounded cursor-pointer opacity-40 hover:opacity-100 hover:bg-[var(--danger-bg)] hover:text-[var(--danger)] transition-all" title="删除">
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 pt-2">
              <Button variant="ghost" size="sm" disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>←</Button>
              <span className="text-sm text-[var(--text-muted)]">{page} / {totalPages}</span>
              <Button variant="ghost" size="sm" disabled={page >= totalPages} onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>→</Button>
            </div>
          )}
        </>
      )}
    </>
  )
}
