import { useCallback, useEffect, useRef, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listPerfTasks, deletePerfTask, getPerfReportUrl, type PerfTaskMeta } from '@/api/perf'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Card from '@/components/ui/Card'
import { ExternalLink, FolderOpen, History, Search, Trash2 } from 'lucide-react'

export default function PerfReportsPage() {
  const { t } = useLocale()
  const [history, setHistory] = useState<PerfTaskMeta[]>([])
  const [loading, setLoading] = useState(true)
  const [rootPath, setRootPath] = useState('')

  // Filters
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const searchTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const [filterModel, setFilterModel] = useState('')
  const [filterDataset, setFilterDataset] = useState('')
  const [sortOrder, setSortOrder] = useState('desc')
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([])

  useEffect(() => {
    searchTimer.current = setTimeout(() => setDebouncedSearch(search), 300)
    return () => clearTimeout(searchTimer.current)
  }, [search])

  const loadHistory = useCallback(async () => {
    setLoading(true)
    try {
      const params: Record<string, string> = {}
      if (rootPath) params.root_path = rootPath
      if (debouncedSearch) params.search = debouncedSearch
      if (filterModel) params.model = filterModel
      if (filterDataset) params.dataset = filterDataset
      params.sort_by = 'time'
      params.sort_order = sortOrder
      const res = await listPerfTasks(params)
      setHistory(res.tasks || [])
      if (res.root_path && !rootPath) setRootPath(res.root_path)
      if (res.filters) {
        setAvailableModels(res.filters.available_models)
        setAvailableDatasets(res.filters.available_datasets)
      }
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }, [rootPath, debouncedSearch, filterModel, filterDataset, sortOrder])

  useEffect(() => { loadHistory() }, [loadHistory])

  const handleViewReport = (tid: string) => {
    window.open(getPerfReportUrl(tid), '_blank')
  }

  const handleDelete = useCallback(async (tid: string, model: string) => {
    if (!window.confirm(`确定要删除 "${model}" 的压测记录吗？此操作不可撤销。`)) return
    try {
      await deletePerfTask(tid)
      loadHistory()
    } catch (e) {
      alert(e instanceof Error ? e.message : '删除失败')
    }
  }, [loadHistory])

  return (
    <div className="page-enter flex flex-col gap-5">
      <Breadcrumb items={[{ label: t('nav.perfReports') }]} />

      {/* Path bar */}
      <div className="flex items-center gap-2">
        <FolderOpen size={16} className="text-[var(--text-muted)] shrink-0" />
        <input
          type="text"
          value={rootPath}
          onChange={(e) => setRootPath(e.target.value)}
          className="flex-1 px-3 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:border-[var(--accent)] transition-all duration-[var(--transition)]"
          placeholder="输出目录路径"
        />
      </div>

      {/* Filter bar */}
      {history.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative flex-1 min-w-[180px] max-w-[300px]">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-dim)]" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-3 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:border-[var(--accent)] transition-all duration-[var(--transition)]"
              placeholder="搜索..."
            />
          </div>
          <select value={filterModel} onChange={(e) => setFilterModel(e.target.value)}
            className="px-2 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] cursor-pointer max-w-[140px] truncate focus:outline-none focus:border-[var(--accent)]">
            <option value="">全部模型</option>
            {availableModels.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          <select value={filterDataset} onChange={(e) => setFilterDataset(e.target.value)}
            className="px-2 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] cursor-pointer max-w-[120px] truncate focus:outline-none focus:border-[var(--accent)]">
            <option value="">全部数据集</option>
            {availableDatasets.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <button onClick={() => setSortOrder((o) => (o === 'desc' ? 'asc' : 'desc'))}
            className="px-2 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)] cursor-pointer"
            title={sortOrder === 'desc' ? '降序' : '升序'}>
            {sortOrder === 'desc' ? '↓ 最新' : '↑ 最早'}
          </button>
        </div>
      )}

      {/* Content */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-12 gap-2 text-[var(--text-dim)]">
          <div className="w-8 h-8 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
          <p className="text-sm">加载中...</p>
        </div>
      ) : history.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 gap-3 text-[var(--text-dim)]">
          <History size={40} />
          <h3 className="text-lg font-semibold text-[var(--text)]">暂无压测记录</h3>
          <p className="text-sm text-[var(--text-muted)]">运行一次性能测试后将在此显示</p>
        </div>
      ) : (
        <Card title="压测历史">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">任务 ID</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">模型</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">数据集</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">API</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">并发配置数</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">时间</th>
                  <th className="text-right py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">操作</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.task_id} className="border-b border-[var(--border)] hover:bg-[var(--bg-card2)] transition-colors">
                    <td className="py-2.5 px-3 font-mono text-xs text-[var(--text-muted)] max-w-[180px] truncate" title={item.task_id}>
                      {item.task_id.replace('perf_', '')}
                    </td>
                    <td className="py-2.5 px-3 font-medium text-[var(--text)] max-w-[280px] truncate" title={item.model}>
                      {item.model}
                    </td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)] max-w-[150px] truncate" title={item.dataset}>
                      {item.dataset}
                    </td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)]">{item.api}</td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)]">{item.runs}</td>
                    <td className="py-2.5 px-3 text-xs text-[var(--text-dim)] whitespace-nowrap">{item.timestamp}</td>
                    <td className="py-2.5 px-3 text-right whitespace-nowrap">
                      {item.has_report && (
                        <button onClick={() => handleViewReport(item.task_id)}
                          className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded text-[var(--accent)] hover:bg-[var(--accent-dim)] transition-colors cursor-pointer" title="查看报告">
                          <ExternalLink size={13} />报告
                        </button>
                      )}
                      <button onClick={() => handleDelete(item.task_id, item.model)}
                        className="inline-flex items-center gap-1 px-2 py-1 ml-1 text-xs rounded text-[var(--text-muted)] hover:bg-[var(--danger-bg)] hover:text-[var(--danger)] transition-colors cursor-pointer" title="删除">
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
