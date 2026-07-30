import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listPerfTasks, deletePerfTask, getPerfReportUrl, type PerfTaskMeta } from '@/api/perf'
import { toast } from '@/components/common/Toast'
import { api } from '@/api/client'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Card from '@/components/ui/Card'
import ServerBadge from '@/components/ui/ServerBadge'
import CompareBar from '@/components/reports/CompareBar'
import { ExternalLink, History, Search, Trash2 } from 'lucide-react'
import Button from '@/components/ui/Button'

const PAGE_SIZE = 20

export default function PerfReportsPage() {
  const { t } = useLocale()
  const [history, setHistory] = useState<PerfTaskMeta[]>([])
  const [loading, setLoading] = useState(true)
  const [rootPath, setRootPath] = useState('')
  const [serverAddress, setServerAddress] = useState('')
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const searchTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const [filterModel, setFilterModel] = useState('')
  const [filterDataset, setFilterDataset] = useState('')
  const [sortOrder, setSortOrder] = useState('desc')
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([])

  // Pagination
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  // Selection (compare)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [selectAllLoading, setSelectAllLoading] = useState(false)
  const allPageIds = useMemo(() => history.map((item) => item.task_id), [history])

  useEffect(() => {
    searchTimer.current = setTimeout(() => {
      setDebouncedSearch(search)
      setPage(1)
    }, 300)
    return () => clearTimeout(searchTimer.current)
  }, [search])

  // Fetch server address from config
  useEffect(() => {
    api<{ server_address?: string }>('/api/v1/config')
      .then((cfg) => {
        if (cfg.server_address) setServerAddress(cfg.server_address)
      })
      .catch(() => {/* ignore */})
  }, [])

  const loadHistory = useCallback(async (p: number) => {
    setLoading(true)
    try {
      const params: Record<string, string> = {}
      if (rootPath) params.root_path = rootPath
      if (debouncedSearch) params.search = debouncedSearch
      if (filterModel) params.model = filterModel
      if (filterDataset) params.dataset = filterDataset
      params.sort_by = 'time'
      params.sort_order = sortOrder
      params.page = String(p)
      params.page_size = String(PAGE_SIZE)
      const res = await listPerfTasks(params)
      setHistory(res.tasks || [])
      setTotal(res.total || 0)
      if (res.root_path && !rootPath) setRootPath(res.root_path)
      if (res.filters) {
        setAvailableModels(res.filters.available_models)
        setAvailableDatasets(res.filters.available_datasets)
      }
    } catch {
      toast.error(t('common.loadFailed'))
    }
    finally { setLoading(false) }
  }, [rootPath, debouncedSearch, filterModel, filterDataset, sortOrder])

  useEffect(() => { loadHistory(page) }, [loadHistory, page])

  // Reset page when filters change
  useEffect(() => { setPage(1) }, [filterModel, filterDataset, sortOrder, debouncedSearch])

  const handleViewReport = (tid: string) => window.open(getPerfReportUrl(tid), '_blank')

  const handleDelete = useCallback(async (tid: string) => {
    if (!window.confirm(t('perf.confirmDelete'))) return
    try {
      await deletePerfTask(tid)
      toast.success(t('common.deleteSuccess'))
      setSelected((prev) => { const next = new Set(prev); next.delete(tid); return next })
      loadHistory(page)
    }
    catch (e) { toast.error(e instanceof Error ? e.message : t('common.deleteFailed')) }
  }, [loadHistory, page, t])

  // ---- Selection helpers ----
  const toggleSelect = useCallback((taskId: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      next.has(taskId) ? next.delete(taskId) : next.add(taskId)
      return next
    })
  }, [])

  const togglePageSelect = useCallback(() => {
    setSelected((prev) => {
      const next = new Set(prev)
      const allSelected = allPageIds.every((id) => prev.has(id))
      if (allSelected) {
        for (const id of allPageIds) next.delete(id)
      } else {
        for (const id of allPageIds) next.add(id)
      }
      return next
    })
  }, [allPageIds])

  const toggleSelectAll = useCallback(async () => {
    if (selected.size >= total) {
      setSelected(new Set())
      return
    }
    setSelectAllLoading(true)
    try {
      const params: Record<string, string> = { page: '1', page_size: '10000' }
      if (rootPath) params.root_path = rootPath
      const res = await listPerfTasks(params)
      setSelected(new Set((res.tasks || []).map((t) => t.task_id)))
    } catch {
      setSelected((prev) => {
        const next = new Set(prev)
        for (const id of allPageIds) next.add(id)
        return next
      })
    } finally {
      setSelectAllLoading(false)
    }
  }, [selected.size, total, rootPath, allPageIds])

  const handleClear = useCallback(() => setSelected(new Set()), [])

  const allPageSelected = allPageIds.length > 0 && allPageIds.every((id) => selected.has(id))

  return (
    <div className="page-enter flex flex-col gap-5">
      {/* Header with Breadcrumb and Server Address */}
      <div className="flex items-center justify-between">
        <Breadcrumb items={[{ label: t('nav.perfReports') }]} />
        <ServerBadge address={serverAddress} />
      </div>

      {/* Compare bar */}
      <CompareBar
        selected={[...selected]}
        totalCount={total}
        rootPath={rootPath}
        backend="Perf"
        onSelectAll={toggleSelectAll}
        onClear={handleClear}
        loading={selectAllLoading}
        compareUrl={`/compare?root_path=${encodeURIComponent(rootPath)}`}
      />

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[180px] max-w-[300px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-dim)]" />
          <input type="text" value={search} onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:border-[var(--accent)]"
            placeholder={t('perf.search')} />
        </div>
        <select value={filterModel} onChange={(e) => setFilterModel(e.target.value)}
          className="px-2 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] cursor-pointer max-w-[140px] truncate focus:outline-none focus:border-[var(--accent)]">
          <option value="">{t('perf.allModels')}</option>
          {availableModels.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        <select value={filterDataset} onChange={(e) => setFilterDataset(e.target.value)}
          className="px-2 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] cursor-pointer max-w-[120px] truncate focus:outline-none focus:border-[var(--accent)]">
          <option value="">{t('perf.allDatasets')}</option>
          {availableDatasets.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <button onClick={() => setSortOrder((o) => (o === 'desc' ? 'asc' : 'desc'))}
          className="px-2 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)] cursor-pointer">
          {sortOrder === 'desc' ? t('perf.sortNewest') : t('perf.sortOldest')}
        </button>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-12 gap-2 text-[var(--text-dim)]">
          <div className="w-8 h-8 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
          <p className="text-sm">{t('perf.loading')}</p>
        </div>
      ) : history.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 gap-3 text-[var(--text-dim)]">
          <History size={40} />
          <h3 className="text-lg font-semibold text-[var(--text)]">{t('perf.noRecords')}</h3>
          <p className="text-sm text-[var(--text-muted)]">{t('perf.noRecordsHint')}</p>
        </div>
      ) : (
        <Card title={t('perf.historyTitle')}>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="w-10 py-2.5 px-3">
                    <input
                      type="checkbox"
                      checked={allPageSelected}
                      onChange={togglePageSelect}
                      className="w-4 h-4 rounded accent-[var(--accent)] cursor-pointer"
                    />
                  </th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">{t('perf.taskId')}</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">{t('perf.model')}</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">{t('perf.datasetLabel')}</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">{t('perf.api')}</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">{t('perf.concurrencyCount')}</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">{t('perf.time')}</th>
                  <th className="text-right py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">{t('perf.actions')}</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.task_id} className={`border-b border-[var(--border)] hover:bg-[var(--bg-card2)] transition-colors ${selected.has(item.task_id) ? 'bg-[var(--accent-dim)]' : ''}`}>
                    <td className="py-2.5 px-3">
                      <input
                        type="checkbox"
                        checked={selected.has(item.task_id)}
                        onChange={() => toggleSelect(item.task_id)}
                        className="w-4 h-4 rounded accent-[var(--accent)] cursor-pointer"
                      />
                    </td>
                    <td className="py-2.5 px-3 font-mono text-xs text-[var(--text-muted)] max-w-[180px] truncate" title={item.task_id}>{item.task_id.replace('perf_', '')}</td>
                    <td className="py-2.5 px-3 font-medium text-[var(--text)] max-w-[280px] truncate" title={item.model}>{item.model}</td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)] max-w-[150px] truncate" title={item.dataset}>{item.dataset}</td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)]">{item.api}</td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)]">{item.runs}</td>
                    <td className="py-2.5 px-3 text-xs text-[var(--text-dim)] whitespace-nowrap">{item.timestamp}</td>
                    <td className="py-2.5 px-3 text-right whitespace-nowrap">
                      {item.has_report && (
                        <button onClick={() => handleViewReport(item.task_id)}
                          className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded text-[var(--accent)] hover:bg-[var(--accent-dim)] transition-colors cursor-pointer">
                          <ExternalLink size={13} />{t('perf.report')}
                        </button>
                      )}
                      <button onClick={() => handleDelete(item.task_id)}
                        className="inline-flex items-center gap-1 px-2 py-1 ml-1 text-xs rounded text-[var(--text-muted)] hover:bg-[var(--danger-bg)] hover:text-[var(--danger)] transition-colors cursor-pointer">
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
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
        </Card>
      )}
    </div>
  )
}
