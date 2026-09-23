import { useEffect, useState, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { listRunningTasks, stopEvalTask } from '@/api/eval'
import { stopPerfTask } from '@/api/perf'
import type { RunningTask } from '@/api/eval'
import { Activity, Square } from 'lucide-react'

export default function RunningTasksIndicator() {
  const [tasks, setTasks] = useState<RunningTask[]>([])
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const mountedRef = useRef(true)
  const navigate = useNavigate()

  const fetch = useCallback(async () => {
    try {
      const res = await listRunningTasks()
      if (mountedRef.current) setTasks(res.tasks || [])
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null
    const poll = async () => {
      await fetch()
      if (!cancelled) timer = setTimeout(poll, 5000)
    }
    void poll()
    return () => {
      mountedRef.current = false
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [fetch])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleStop = useCallback(async (e: React.MouseEvent, task: RunningTask) => {
    e.stopPropagation()
    try {
      if (task.task_type === 'perf') await stopPerfTask(task.task_id)
      else await stopEvalTask(task.task_id)
      fetch()
    } catch { /* ignore */ }
  }, [fetch])

  const handleClick = useCallback((task: RunningTask) => {
    const path = task.task_type === 'perf' ? '/perf' : '/eval/llm'
    navigate(`${path}?task=${encodeURIComponent(task.task_id)}`)
    setOpen(false)
  }, [navigate])

  if (tasks.length === 0) return null

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 px-2 py-1 text-xs rounded hover:bg-[var(--bg-card)] text-[var(--accent)] transition-colors"
        title={`${tasks.length} 个任务运行中`}
      >
        <Activity size={14} className="animate-pulse" />
        <span className="font-mono">{tasks.length}</span>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-56 rounded-lg border border-[var(--border)] bg-[var(--bg-card)] shadow-lg z-50 py-1">
          {tasks.map((t) => (
            <div
              key={t.task_id}
              onClick={() => handleClick(t)}
              className="flex items-center justify-between px-3 py-1.5 text-xs hover:bg-[var(--bg-card2)] cursor-pointer transition-colors"
            >
              <span className="text-[var(--text)] truncate flex-1">{t.model || t.task_id}</span>
              <span className="text-[var(--text-dim)] mx-1">{Math.floor(t.elapsed_seconds / 60)}m</span>
              <button
                onClick={(e) => handleStop(e, t)}
                className="p-0.5 rounded hover:bg-[var(--danger)]/20 text-[var(--text-muted)] hover:text-[var(--danger)]"
                title="停止"
              >
                <Square size={10} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
