import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQueryParams } from '@/hooks/useQueryParams'
import { useSSE } from '@/hooks/useSSE'
import { toast } from '@/components/common/Toast'
import { createTaskId } from '@/utils/taskId'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from '@/api/types'
import { classifyProgress, progressRetryDelay, shouldShowProgressError } from './taskLifecycle'

export interface TaskApi {
  submit: (config: Record<string, unknown>, taskId: string) => Promise<EvalInvokeResponse>
  launch?: (config: Record<string, unknown>, taskId: string) => Promise<{ task_id: string; status: string }>
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
  const [progressError, setProgressError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const resumedRef = useRef(false)
  const suppressProgressPollRef = useRef(false)

  // --- localStorage persistence: survive page refresh ---
  const STORAGE_KEY = `evalscope_last_${taskPrefix}`

  const saveTaskId = useCallback((id: string) => {
    try { localStorage.setItem(STORAGE_KEY, id) } catch { /* quota / private mode */ }
  }, [STORAGE_KEY])

  const clearTaskId = useCallback(() => {
    try { localStorage.removeItem(STORAGE_KEY) } catch { /* ignore */ }
  }, [STORAGE_KEY])

  // On mount: restore persisted task if no URL param and no active task
  useEffect(() => {
    if (urlTaskId || taskId) return
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        queueMicrotask(() => {
          setTaskId(saved)
          setProgressError(null)
          setRunning(true)
        })
        window.history.replaceState(null, '', `?task=${saved}`)
      }
    } catch { /* ignore */ }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Resume monitoring a task from URL ?task=xxx (e.g. from running tasks indicator)
  useEffect(() => {
    if (urlTaskId && !resumedRef.current) {
      resumedRef.current = true
      setTaskId(urlTaskId)

      const resume = async () => {
        try {
          const d = await api.getLog(urlTaskId)
          if (d.text) setLogText(d.text)
        } catch { /* Progress polling remains the source of task status. */ }
        setRunning(true)
      }
      resume()
    }
  }, [urlTaskId, api])

  const handleSubmit = async (config: Record<string, unknown>) => {
    suppressProgressPollRef.current = false
    const id = createTaskId(taskPrefix)
    saveTaskId(id)
    setTaskId(id)
    setLogText('')
    setProgress(0)
    setProgressError(null)
    setResult(null)
    setCopied(false)

    // Use non-blocking /launch if available, otherwise blocking /invoke
    const launchFn = api.launch || api.submit

    setRunning(true)
    try {
      const res = await launchFn(config, id)
      // For blocking /invoke, res is the full result (eval completed).
      // For non-blocking /launch, the shared lifecycle poll handles status.
      if (!res || (res as { status: string }).status !== 'launched') {
        setResult(res as EvalInvokeResponse)
        setRunning(false)
        clearTaskId()
        try {
          const finalLog = await api.getLog(id, 0, 999999)
          if (finalLog.text) setLogText(finalLog.text)
          const finalProg = await api.getProgress(id)
          setProgress(finalProg.percent ?? 100)
        } catch { /* ignore */ }
      }
    } catch (e) {
      const msg = String(e)
      setRunning(false)
      clearTaskId()
      setResult({ status: 'error', task_id: id, error: msg } as EvalInvokeResponse)
      toast.error(msg)
    }
  }

  const handleStop = async () => {
    if (!taskId) return
    try { await api.stop(taskId) } catch { toast.warning('Stop request failed') }
    clearTaskId()
    setRunning(false)
    suppressProgressPollRef.current = false
    setProgressError(null)
    setResult({ status: 'stopped', task_id: taskId })
  }

  const handleResume = async (existingTaskId: string, apiKey?: string) => {
    // /resume/invoke is blocking. Its old progress file can still contain the
    // previous terminal status during startup, so do not classify it mid-run.
    suppressProgressPollRef.current = true
    setTaskId(existingTaskId)
    setRunning(true)
    setLogText('')
    setProgress(0)
    setProgressError(null)
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
      suppressProgressPollRef.current = false
      clearTaskId()
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

  // Build SSE URL for log streaming (only persistent SSE per task)
  const logStreamUrl = useMemo(() => {
    if (!taskId) return null
    return `/api/v1/${taskPrefix}/log/stream?task_id=${taskId}`
  }, [taskId, taskPrefix])

  // Single lifecycle poll for launch and URL restore.  Transport failures do
  // not become business success; they preserve state and retry with backoff.
  useEffect(() => {
    if (!running || !taskId || suppressProgressPollRef.current) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null
    let failures = 0

    const schedule = (delay: number) => {
      if (!cancelled) timer = setTimeout(poll, delay)
    }

    const poll = async () => {
      try {
        const d = await api.getProgress(taskId)
        if (cancelled) return
        failures = 0
        setProgressError(null)
        setProgress(d.percent ?? 0)
        const outcome = classifyProgress(d)
        if (outcome) {
          setRunning(false)
          clearTaskId()
          setResult({ ...outcome, task_id: taskId })
          try {
            const finalLog = await api.getLog(taskId, 0, 999999)
            if (!cancelled && finalLog.text) setLogText(finalLog.text)
          } catch { /* Status remains authoritative if the final log is unavailable. */ }
          return
        }
        schedule(3000)
      } catch {
        if (cancelled) return
        failures += 1
        if (shouldShowProgressError(failures)) {
          setProgressError('任务状态暂不可用，正在重试')
        }
        schedule(progressRetryDelay(failures))
      }
    }

    poll()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [running, taskId, api, clearTaskId])

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
    running, progress, progressError, result, logText, reportUrl, copied, taskId,
    handleSubmit, handleStop, handleResume, copyLog,
    sseState: logSSE.connectionState,
  }
}
