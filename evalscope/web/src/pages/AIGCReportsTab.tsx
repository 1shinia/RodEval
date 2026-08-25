import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, Image as ImageIcon, Trash2 } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { toast } from '@/components/common/Toast'
import { api, apiDelete } from '@/api/client'
import Button from '@/components/ui/Button'
import Skeleton from '@/components/ui/Skeleton'
import ReportFiltersBar, { type ReportFilters } from '@/components/reports/ReportFilters'
import CompareBar from '@/components/reports/CompareBar'
import { EmptyState } from '@/pages/ReportsLayout'

interface AIGCReportSummary {
  task_id: string
  model_name: string
  model_type: string
  total_images: number
  clip_score_mean?: number
  lpips_mean?: number
  has_errors?: number
  error_note?: string
  fvd?: number
  inception_score?: number
  created_at: string
}

const PAGE_SIZE = 20

const defaultFilters: ReportFilters = {
  search: '',
  models: [],
  datasets: [],
  scoreMin: 0,
  scoreMax: 100,
  sortBy: 'time',
  sortOrder: 'desc',
}

function primaryScore(r: AIGCReportSummary): number {
  return r.clip_score_mean ?? r.lpips_mean ?? r.fvd ?? 0
}

export default function AIGCReportsTab() {
  const { t } = useLocale()
  const navigate = useNavigate()

  const [allReports, setAllReports] = useState<AIGCReportSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState<ReportFilters>(defaultFilters)
  const [page, setPage] = useState(1)

  const [selected, setSelected] = useState<Set<string>>(new Set())

  const fetchReports = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api<{ reports: AIGCReportSummary[] }>('/api/v1/aigc/reports')
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
      await apiDelete<{ ok: boolean }>(`/api/v1/aigc/reports/${encodeURIComponent(taskId)}`)
      toast.success('已删除')
      setSelected((prev) => { const next = new Set(prev); next.delete(taskId); return next })
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
    () => [...new Set(allReports.map((r) => r.model_type).filter(Boolean))].sort(),
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
      list = list.filter((r) => ds.has(r.model_type.toLowerCase()))
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

  const allPageNames = useMemo(() => paged.map((r) => r.task_id), [paged])
  const allPageSelected = allPageNames.length > 0 && allPageNames.every((n) => selected.has(n))

  const toggleSelect = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const toggleSelectAll = useCallback(() => {
    if (selected.size >= filtered.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(filtered.map((r) => r.task_id)))
    }
  }, [selected.size, filtered])

  const togglePageSelect = useCallback(() => {
    setSelected((prev) => {
      const next = new Set(prev)
      const allSelected = allPageNames.every((n) => prev.has(n))
      if (allSelected) {
        for (const n of allPageNames) next.delete(n)
      } else {
        for (const n of allPageNames) next.add(n)
      }
      return next
    })
  }, [allPageNames])

  const handleClear = useCallback(() => setSelected(new Set()), [])

  return (
    <>
      <CompareBar
        selected={[...selected]}
        totalCount={filtered.length}
        rootPath=""
        backend="AIGCEval"
        onSelectAll={toggleSelectAll}
        onClear={handleClear}
      />

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
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-16 w-full rounded-lg" />)}
        </div>
      ) : allReports.length === 0 ? (
        <EmptyState icon={<ImageIcon size={40} />} title={t('aigc.noReports')} subtitle={t('aigc.noReportsHint')} />
      ) : paged.length === 0 ? (
        <div className="text-center text-[var(--text-muted)] py-8">无匹配结果</div>
      ) : (
        <>
          <div className="rounded-[var(--radius)] border border-[var(--border)] overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-[var(--bg-deep)] border-b border-[var(--border)]">
                  <th className="w-10 px-3 py-3">
                    <input
                      type="checkbox"
                      checked={allPageSelected}
                      onChange={togglePageSelect}
                      className="w-4 h-4 rounded accent-[var(--accent)] cursor-pointer"
                    />
                  </th>
                  <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.taskId')}</th>
                  <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">类型</th>
                  <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.modelName')}</th>
                  <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.totalImages')}</th>
                  <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">评估得分</th>
                  <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">{t('aigc.createdAt')}</th>
                  <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">{t('common.actions')}</th>
                </tr>
              </thead>
              <tbody>
                {paged.map((report) => (
                  <tr key={report.task_id} className={`border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--bg-card2)] transition-colors ${selected.has(report.task_id) ? 'bg-[var(--accent-dim)]' : ''}`}>
                    <td className="px-3 py-3">
                      <input
                        type="checkbox"
                        checked={selected.has(report.task_id)}
                        onChange={() => toggleSelect(report.task_id)}
                        className="w-4 h-4 rounded accent-[var(--accent)] cursor-pointer"
                      />
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)]">{report.task_id}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        report.model_type === 'txt2img' ? 'bg-purple-500/10 text-purple-400' :
                        report.model_type === 'txt2video' ? 'bg-blue-500/10 text-blue-400' :
                        'bg-orange-500/10 text-orange-400'
                      }`}>
                        {report.model_type === 'txt2img' ? '文生图' : report.model_type === 'txt2video' ? '文生视频' : '图生图'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-[var(--text)]">{report.model_name}</td>
                    <td className="px-4 py-3 text-right text-[var(--text)]">{report.total_images}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs">
                      <div className="flex flex-col gap-0.5 items-end">
                        {report.clip_score_mean != null && <span className="text-[var(--accent)]">CLIP: {report.clip_score_mean.toFixed(2)}</span>}
                        {report.lpips_mean != null && <span className="text-[var(--text)]">LPIPS: {report.lpips_mean.toFixed(4)}</span>}
                        {report.fvd != null && <span className="text-[var(--warning-color)]">FVD: {report.fvd.toFixed(2)}</span>}
                        {report.clip_score_mean == null && report.lpips_mean == null && report.fvd == null && <span className="text-[var(--text-dim)]">-</span>}
                        {report.has_errors ? (
                          <span
                            title={report.error_note || '失败/不完整结果'}
                            className="inline-flex items-center rounded px-1 py-px text-[10px] font-medium text-white bg-amber-500/90 cursor-help"
                          >
                            失败
                          </span>
                        ) : null}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-[var(--text-muted)] text-xs">{new Date(report.created_at).toLocaleString()}</td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <Button variant="outline" size="sm" onClick={() => navigate(`/reports/aigc/${encodeURIComponent(report.task_id)}`)}>
                          <Eye size={14} />{t('common.view')}
                        </Button>
                        <button type="button" onClick={(e) => { e.stopPropagation(); handleDelete(report.task_id) }} className="p-1.5 rounded cursor-pointer opacity-40 hover:opacity-100 hover:bg-[var(--danger-bg)] hover:text-[var(--danger)] transition-all" title="删除">
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
              {Array.from({ length: totalPages }, (_, i) => i + 1)
                .filter((p) => p === 1 || p === totalPages || Math.abs(p - page) <= 2)
                .reduce<(number | 'ellipsis')[]>((acc, p, idx, arr) => {
                  if (idx > 0 && p - (arr[idx - 1] as number) > 1) acc.push('ellipsis')
                  acc.push(p)
                  return acc
                }, [])
                .map((item, idx) =>
                  item === 'ellipsis' ? (
                    <span key={`e${idx}`} className="px-1 text-[var(--text-dim)]">...</span>
                  ) : (
                    <Button key={item} variant={item === page ? 'primary' : 'ghost'} size="sm" onClick={() => setPage(item as number)} className="!min-w-[32px]">{item}</Button>
                  ),
                )}
              <Button variant="ghost" size="sm" disabled={page >= totalPages} onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>→</Button>
            </div>
          )}
        </>
      )}
    </>
  )
}
