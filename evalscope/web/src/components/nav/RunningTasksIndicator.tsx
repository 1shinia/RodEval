import { useEffect, useState, useRef } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { api } from '@/api/client'
import { Activity, X, ExternalLink, Loader2 } from 'lucide-react'

interface RunningTask {
  task_id: string
  task_type: 'eval' | 'perf'
  model: string
  start_time: number
  elapsed_seconds: number
}

interface RunningTasksResponse {
  tasks: RunningTask[]
  count: number
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return `${h}h ${m}m`
}

export default function RunningTasksIndicator() {
  const { t } = useLocale()
  const [tasks, setTasks] = useState<RunningTask[]>([])
  const [open, setOpen] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)

  // Poll every 3 seconds
  useEffect(() => {
    let cancelled = false

    const fetchTasks = async () => {
      try {
        const data = await api<RunningTasksResponse>('/api/v1/tasks/running')
        if (!cancelled) setTasks(data.tasks)
      } catch {
        // Silently ignore errors
      }
    }

    fetchTasks()
    const interval = setInterval(fetchTasks, 3000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  // Update elapsed time every second
  useEffect(() => {
    if (tasks.length === 0) return
    const interval = setInterval(() => {
      setTasks(prev =>
        prev.map(task => ({
          ...task,
          elapsed_seconds: (Date.now() / 1000) - task.start_time,
        }))
      )
    }, 1000)
    return () => clearInterval(interval)
  }, [tasks.length])

  // Close panel on outside click
  useEffect(() => {
    if (!open) return
    const handleClick = (e: MouseEvent) => {
      if (
        panelRef.current &&
        !panelRef.current.contains(e.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(e.target as Node)
      ) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  const count = tasks.length

  return (
    <div className="relative">
      <button
        ref={buttonRef}
        onClick={() => setOpen(!open)}
        className={`relative flex items-center justify-center w-9 h-9 rounded-lg transition-all duration-200 ${
          count > 0
            ? 'text-[var(--accent)] hover:bg-[var(--bg-card2)]'
            : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)]'
        }`}
        title={count > 0 ? `${count} ${t('tasks.running')}` : t('tasks.noRunning')}
      >
        {count > 0 ? (
          <Activity size={18} className="animate-pulse" />
        ) : (
          <Activity size={18} />
        )}
        {count > 0 && (
          <span className="absolute -top-0.5 -right-0.5 flex items-center justify-center w-4 h-4 text-[10px] font-bold rounded-full bg-[var(--accent)] text-white">
            {count}
          </span>
        )}
      </button>

      {open && (
        <div
          ref={panelRef}
          className="absolute right-0 top-full mt-2 w-80 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] shadow-xl z-50"
        >
          <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
            <span className="text-sm font-semibold text-[var(--text)]">
              {t('tasks.running')} ({count})
            </span>
            <button
              onClick={() => setOpen(false)}
              className="w-6 h-6 flex items-center justify-center rounded text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)]"
            >
              <X size={14} />
            </button>
          </div>

          <div className="max-h-64 overflow-y-auto">
            {tasks.length === 0 ? (
              <div className="px-4 py-6 text-center text-sm text-[var(--text-muted)]">
                {t('tasks.noRunning')}
              </div>
            ) : (
              <div className="divide-y divide-[var(--border)]">
                {tasks.map(task => (
                  <div key={task.task_id} className="px-4 py-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Loader2 size={14} className="animate-spin text-[var(--accent)]" />
                      <span className="text-sm font-medium text-[var(--text)]">
                        {task.task_type === 'eval' ? t('tasks.eval') : t('tasks.perf')}
                      </span>
                      <span className="text-xs text-[var(--text-muted)] ml-auto">
                        {t('tasks.elapsed')}: {formatElapsed(task.elapsed_seconds)}
                      </span>
                    </div>
                    <div className="text-xs text-[var(--text-muted)] truncate">
                      {t('tasks.model')}: {task.model || task.task_id}
                    </div>
                    <div className="flex gap-2 mt-2">
                      <a
                        href={task.task_type === 'eval' ? `/eval?task=${task.task_id}` : `/perf?task=${task.task_id}`}
                        className="inline-flex items-center gap-1 text-xs text-[var(--accent)] hover:underline"
                      >
                        <ExternalLink size={12} />
                        {t('tasks.viewLog')}
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
