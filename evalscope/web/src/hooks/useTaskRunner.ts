import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQueryParams } from '@/hooks/useQueryParams'
import { useSSE } from '@/hooks/useSSE'
import { toast } from '@/components/common/Toast'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from '@/api/types'

export interface TaskApi {
  submit: (config: Record<string, unknown>, taskId: string) => Promise<EvalInvokeResponse>
  stop: (taskId: string) => Promise<unknown>
  getProgress: (taskId: string) => Promise<ProgressResponse>
  getLog: (taskId: string, startLine?: number, page?: number) => Promise<LogResponse>
  getReportUrl: (taskId: string) => string
  resume?: (taskId: string, apiKey?: string) => Promise<EvalInvokeResponse>
}

export interface UseTaskRunnerOptions {
  api: TaskApi
  taskPrefix: string
}

export function useTaskRunner({ api, taskPrefix }: UseTaskRunnerOptions) {
  const queryParams = useQueryParams()
  const urlTaskId = queryParams.get('task')

  const [taskId, setTaskId] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<EvalInvokeResponse | null>(null)
  const [logText, setLogText] = useState('')
  const [progress, setProgress] = useState(0)
  const [copied, setCopied] = useState(false)
  const resumedRef = useRef(false)

  // Resume monitoring a task from URL ?task=xxx (e.g. from running tasks indicator)
  useEffect(() => {
    if (urlTaskId && !resumedRef.current) {
      resumedRef.current = true
      setTaskId(urlTaskId)
      setRunning(true)

      const resume = async () => {
        let logAcc = ''
        let nextLine = 0
        try {
          const d = await api.getLog(urlTaskId, 0)
          if (d.text) { logAcc = d.text; nextLine = d.tail_line }
        } catch { /* ignore */ }

        let done = false
        try {
          const p = await api.getProgress(urlTaskId)
          setProgress(p.percent ?? 0)
          if (p.percent >= 100) done = true
        } catch { done = true }

        // If task already completed, fetch ALL remaining log pages
        if (done) {
          try {
            let safety = 0
            while (nextLine > 0 && safety < 50) {
              const more = await api.getLog(urlTaskId, nextLine)
              if (!more.text || more.tail_line <= nextLine) break
              logAcc += more.text
              nextLine = more.tail_line
              if (nextLine >= more.total_lines) break
              safety++
            }
          } catch { /* ignore */ }
          setLogText(logAcc)
          setRunning(false)
          setResult({ status: 'ok', task_id: urlTaskId })
        } else {
          setLogText(logAcc)
        }
      }
      resume()
    }
  }, [urlTaskId, api])

  const handleSubmit = async (config: Record<string, unknown>) => {
    const id = `${taskPrefix}_${Date.now()}`
    setTaskId(id)
    setRunning(true)
    setLogText('')
    setProgress(0)
    setResult(null)
    setCopied(false)
    try {
      const res = await api.submit(config, id)
      setResult(res)
    } catch (e) {
      setResult({ status: 'error', task_id: id, error: String(e) })
      toast.error(String(e))
    } finally {
      setRunning(false)
      // Fetch complete final log + progress
      try {
        const finalLog = await api.getLog(id, 0, 999999)
        if (finalLog.text) {
          setLogText(finalLog.text)
        }
        const finalProg = await api.getProgress(id)
        setProgress(finalProg.percent ?? 100)
      } catch { /* ignore */ }
    }
  }

  const handleStop = async () => {
    if (!taskId) return
    try { await api.stop(taskId) } catch { toast.warning('Stop request failed') }
    setRunning(false)
    setResult({ status: 'stopped', task_id: taskId })
  }

  const handleResume = async (existingTaskId: string, apiKey?: string) => {
    setTaskId(existingTaskId)
    setRunning(true)
    setLogText('')
    setProgress(0)
    setResult(null)
    setCopied(false)
    try {
      const res = await api.resume!(existingTaskId, apiKey)
      setResult(res)
    } catch (e) {
      setResult({ status: 'error', task_id: existingTaskId, error: String(e) })
      toast.error(String(e))
    } finally {
      setRunning(false)
      // Fetch complete final log + progress
      try {
        const finalLog = await api.getLog(existingTaskId, 0, 999999)
        if (finalLog.text) {
          setLogText(finalLog.text)
        }
        const finalProg = await api.getProgress(existingTaskId)
        setProgress(finalProg.percent ?? 100)
      } catch { /* ignore */ }
    }
  }

  // Build SSE URLs for real-time streaming
  const progressStreamUrl = useMemo(() => {
    if (!taskId) return null
    return `/api/v1/${taskPrefix}/progress/stream?task_id=${taskId}`
  }, [taskId, taskPrefix])

  const logStreamUrl = useMemo(() => {
    if (!taskId) return null
    return `/api/v1/${taskPrefix}/log/stream?task_id=${taskId}`
  }, [taskId, taskPrefix])

  useSSE<ProgressResponse>({
    url: progressStreamUrl,
    enabled: running && !!taskId,
    onData: (d) => {
      setProgress(d.percent ?? 0)
      if (d.percent >= 100) {
        setRunning(false)
        setResult((prev) => prev ?? { status: 'ok', task_id: taskId! })
      }
    },
  })

  useSSE<LogResponse>({
    url: logStreamUrl,
    enabled: running && !!taskId,
    onData: (d) => { if (d.text) { setLogText((prev) => prev + d.text) } },
  })

  const reportUrl = useMemo(() => (taskId ? api.getReportUrl(taskId) : null), [taskId, api])

  const copyLog = useCallback(() => {
    const text = [logText, result?.error].filter(Boolean).join('\n')
    if (!text) return
    const ta = document.createElement('textarea')
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0'
    document.body.appendChild(ta); ta.select()
    try { document.execCommand('copy') } catch { /* ignore */ }
    document.body.removeChild(ta)
    setCopied(true); setTimeout(() => setCopied(false), 2000)
  }, [logText, result?.error])

  return {
    running, progress, result, logText, reportUrl, copied, taskId,
    handleSubmit, handleStop, handleResume, copyLog,
  }
}
