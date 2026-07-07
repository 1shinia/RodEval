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

      const resume = async () => {
        let logAcc = ''
        let nextLine = 0

        // Check completion first so we know whether to fetch full log or tail
        let done = false
        try {
          const p = await api.getProgress(urlTaskId)
          setProgress(p.percent ?? 0)
          // Also treat error/stopped tasks as done to prevent SSE log re-streaming
          const status = typeof p.status === 'string' ? p.status : ''
          if ((p.percent ?? 0) >= 100 || status === 'error' || status === 'stopped') {
            done = true
          }
        } catch { done = true }

        if (done) {
          // Completed/error/stopped: fetch full log from beginning
          try {
            const d = await api.getLog(urlTaskId, 0)
            if (d.text) { logAcc = d.text; nextLine = d.tail_line }
          } catch { /* ignore */ }
          // Fetch all remaining pages
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
          // Keep running=false – prevents SSE from connecting and duplicating log content
          setResult({ status: 'ok', task_id: urlTaskId })
        } else {
          // Running: fetch tail, then enable SSE for real-time streaming
          try {
            const d = await api.getLog(urlTaskId)
            if (d.text) { logAcc = d.text; nextLine = d.tail_line }
          } catch { /* ignore */ }
          setLogText(logAcc)
          setRunning(true)  // Only now enable SSE – task is confirmed running
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
    let shouldFinalize = true
    try {
      const res = await api.submit(config, id)
      setResult(res)
    } catch (e) {
      const msg = String(e)
      const isNetworkIssue =
        msg.includes('AbortError') ||
        msg.includes('abort') ||
        msg.includes('timeout') ||
        msg.includes('network') ||
        msg.includes('NetworkError') ||
        msg.includes('Failed to fetch')
      if (isNetworkIssue) {
        toast.warning('请求中断，但任务可能在后台继续运行。请等待日志更新或刷新页面查看。')
        shouldFinalize = false  // keep running=true, SSE stays alive
        return
      }
      setResult({ status: 'error', task_id: id, error: msg })
      toast.error(msg)
    } finally {
      if (!shouldFinalize) return
      setRunning(false)
      // Fetch complete final log + progress
      try {
        const finalLog = await api.getLog(id, 0, 999999)
        if (finalLog.text) setLogText(finalLog.text)
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

  const progressSSE = useSSE<ProgressResponse>({
    url: progressStreamUrl,
    enabled: running && !!taskId,
    onData: (d) => {
      setProgress(d.percent ?? 0)
      if ((d.percent ?? 0) >= 100 && d.status === 'completed') {
        setRunning(false)
        setResult((prev) => prev ?? { status: 'ok', task_id: taskId! })
      }
    },
  })

  const logSSE = useSSE<LogResponse>({
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
    sseState: logSSE.connectionState || progressSSE.connectionState,
  }
}
