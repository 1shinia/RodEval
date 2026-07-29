import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { BookOpen, Eye, Trash2 } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { toast } from '@/components/common/Toast'
import * as reportsApi from '@/api/reports'
import type { ReportSummary } from '@/api/types'
import Button from '@/components/ui/Button'
import Skeleton from '@/components/ui/Skeleton'
import ReportFiltersBar, { type ReportFilters } from '@/components/reports/ReportFilters'
import CompareBar from '@/components/reports/CompareBar'
import { EmptyState } from '@/pages/ReportsLayout'

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

function extractTaskId(name: string): string {
  const idx = name.indexOf('@@')
  return idx > 0 ? name.slice(0, idx) : name
}

export default function RAGReportsTab() {
  const { t } = useLocale()
  const navigate = useNavigate()
  const { rootPath } = useReports()

  const [filters, setFilters] = useState<ReportFilters>(defaultFilters)
  const [page, setPage] = useState(1)
  const [reports, setReports] = useState<ReportSummary[]>([])
  const [total, setTotal] = useState(0)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [selectAllLoading, setSelectAllLoading] = useState(false)
  const allPageNames = useMemo(() => reports.map((r) => r.name), [reports])

  const [debouncedSearch, setDebouncedSearch] = useState('')
  const searchTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    searchTimer.current = setTimeout(() => setDebouncedSearch(filters.search), 300)
    return () => clearTimeout(searchTimer.current)
  }, [filters.search])

  const fetchReports = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await reportsApi.listReports({
        rootPath,
        search: debouncedSearch || undefined,
        models: filters.models.length ? filters.models : undefined,
        datasets: filters.datasets.length ? filters.datasets : undefined,
        scoreMin: filters.scoreMin > 0 ? filters.scoreMin : undefined,
        scoreMax: filters.scoreMax < 1 ? filters.scoreMax : undefined,
        sortBy: filters.sortBy,
        sortOrder: filters.sortOrder,
        page,
        pageSize: PAGE_SIZE,
        backend: 'RAGEval',
      })
      setReports(res.reports)
      setTotal(res.total)
      setAvailableModels(res.filters.available_models)
      setAvailableDatasets(res.filters.available_datasets)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load reports'
      setError(msg)
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }, [rootPath, debouncedSearch, filters.models, filters.datasets, filters.scoreMin, filters.scoreMax, filters.sortBy, filters.sortOrder, page])

  useEffect(() => { fetchReports() }, [fetchReports])
  useEffect(() => { setPage(1) }, [debouncedSearch, filters.models, filters.datasets, filters.scoreMin, filters.scoreMax, filters.sortBy, filters.sortOrder])

  const handleDelete = useCallback(async (name: string) => {
    if (!window.confirm(t('reports.confirmDelete', { name }))) return
    try {
      await reportsApi.deleteReport(rootPath, name)
      toast.success(t('common.deleteSuccess'))
      setSelected((prev) => { const next = new Set(prev); next.delete(name); return next })
      fetchReports()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t('common.deleteFailed'))
    }
  }, [rootPath, fetchReports, t])

  const toggleSelect = useCallback((name: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }, [])

  const toggleSelectAll = useCallback(async () => {
    if (selected.size >= total) {
      setSelected(new Set())
      return
    }
    setSelectAllLoading(true)
    try {
      const res = await reportsApi.listReports({
        rootPath,
        page: 1,
        pageSize: 10000,
        sortBy: filters.sortBy,
        sortOrder: filters.sortOrder,
        backend: 'RAGEval',
      })
      setSelected(new Set(res.reports.map((r: ReportSummary) => r.name)))
    } catch {
      setSelected((prev) => {
        const next = new Set(prev)
        for (const n of allPageNames) next.add(n)
        return next
      })
    } finally {
      setSelectAllLoading(false)
    }
  }, [selected.size, total, rootPath, filters.sortBy, filters.sortOrder, allPageNames])

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

  const allPageSelected = allPageNames.length > 0 && allPageNames.every((n) => selected.has(n))

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <>
      <CompareBar
        selected={[...selected]}
        totalCount={total}
        rootPath={rootPath}
        backend="RAGEval"
        onSelectAll={toggleSelectAll}
        onClear={handleClear}
        loading={selectAllLoading}
      />

      <ReportFiltersBar filters={filters} availableModels={availableModels} availableDatasets={availableDatasets} onChange={setFilters} />

      {error && (
        <div className="px-4 py-3 rounded-[var(--radius)] bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger)]">{error}</div>
      )}

      {loading ? (
        <div className="flex flex-col gap-2">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-10 w-full rounded-lg" />)}</div>
      ) : reports.length === 0 ? (
        <EmptyState icon={<BookOpen size={40} />} title="暂无 RAG 评估报告" subtitle="完成 RAG 评估后，报告将显示在这里" />
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
                  <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">任务 ID</th>
                  <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">模型</th>
                  <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">数据集</th>
                  <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">样本数</th>
                  <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">评估得分</th>
                  <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">创建时间</th>
                  <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">操作</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((report) => (
                  <tr key={report.name} className={`border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--bg-card2)] transition-colors ${selected.has(report.name) ? 'bg-[var(--accent-dim)]' : ''}`}>
                    <td className="px-3 py-3">
                      <input
                        type="checkbox"
                        checked={selected.has(report.name)}
                        onChange={() => toggleSelect(report.name)}
                        className="w-4 h-4 rounded accent-[var(--accent)] cursor-pointer"
                      />
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)]">{extractTaskId(report.name)}</td>
                    <td className="px-4 py-3 text-[var(--text)]">{report.model_name}</td>
                    <td className="px-4 py-3 text-[var(--text-muted)]">{report.dataset_name}</td>
                    <td className="px-4 py-3 text-right text-[var(--text)]">{report.num_samples}</td>
                    <td className="px-4 py-3 text-right font-mono text-sm text-[var(--accent)]">{report.score.toFixed(4)}</td>
                    <td className="px-4 py-3 text-[var(--text-muted)] text-xs">{report.timestamp ? new Date(report.timestamp).toLocaleString() : '-'}</td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <Button variant="outline" size="sm" onClick={() => navigate(`/reports/${encodeURIComponent(report.name)}?root_path=${encodeURIComponent(rootPath)}`)}>
                          <Eye size={14} />
                          查看
                        </Button>
                        <button type="button" onClick={(e) => { e.stopPropagation(); handleDelete(report.name) }} className="p-1.5 rounded cursor-pointer opacity-40 hover:opacity-100 hover:bg-[var(--danger-bg)] hover:text-[var(--danger)] transition-all" title="删除">
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
