import { useCallback, useEffect, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listPerfTasks, deletePerfTask, getPerfReportUrl, type PerfTaskMeta } from '@/api/perf'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Card from '@/components/ui/Card'
import { ExternalLink, History, Trash2 } from 'lucide-react'

export default function PerfReportsPage() {
  const { t } = useLocale()
  const [history, setHistory] = useState<PerfTaskMeta[]>([])
  const [loading, setLoading] = useState(true)

  const loadHistory = useCallback(async () => {
    setLoading(true)
    try {
      const res = await listPerfTasks()
      setHistory(res.tasks || [])
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }, [])

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
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">API</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">数据集</th>
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
                    <td className="py-2.5 px-3 font-medium text-[var(--text)] max-w-[200px] truncate" title={item.model}>
                      {item.model}
                    </td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)]">{item.api}</td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)] max-w-[150px] truncate" title={item.dataset}>
                      {item.dataset}
                    </td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)]">{item.runs}</td>
                    <td className="py-2.5 px-3 text-xs text-[var(--text-dim)] whitespace-nowrap">{item.timestamp}</td>
                    <td className="py-2.5 px-3 text-right whitespace-nowrap">
                      {item.has_report && (
                        <button
                          onClick={() => handleViewReport(item.task_id)}
                          className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded text-[var(--accent)] hover:bg-[var(--accent-dim)] transition-colors cursor-pointer"
                          title="查看报告"
                        >
                          <ExternalLink size={13} />
                          报告
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(item.task_id, item.model)}
                        className="inline-flex items-center gap-1 px-2 py-1 ml-1 text-xs rounded text-[var(--text-muted)] hover:bg-[var(--danger-bg)] hover:text-[var(--danger)] transition-colors cursor-pointer"
                        title="删除"
                      >
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
