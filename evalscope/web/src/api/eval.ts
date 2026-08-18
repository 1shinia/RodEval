import { apiPost, api, getAuthHeaders } from './client'
import type { BenchmarksResponse, EvalInvokeResponse, LogResponse, ProgressResponse } from './types'

export interface RunningTask {
  task_id: string
  task_type: string
  model: string
  user_id: number
  start_time: number
  elapsed_seconds: number
}

export async function listRunningTasks(): Promise<{ tasks: RunningTask[] }> {
  return api<{ tasks: RunningTask[] }>('/api/v1/tasks/running')
}

export async function submitEvalTask(
  payload: Record<string, unknown>,
  taskId: string,
): Promise<EvalInvokeResponse> {
  // Eval tasks can run for hours — no client-side timeout (0 = disable).
  // Progress is tracked via SSE; the HTTP response only arrives on completion.
  return apiPost<EvalInvokeResponse>('/api/v1/eval/invoke', payload, { 'EvalScope-Task-Id': taskId }, 0)
}

export async function launchEvalTask(
  payload: Record<string, unknown>,
  taskId: string,
): Promise<{ task_id: string; status: string }> {
  // Non-blocking launch — returns immediately, use SSE + polling for progress.
  return apiPost<{ task_id: string; status: string }>(
    '/api/v1/eval/launch', payload, { 'EvalScope-Task-Id': taskId },
  )
}

export async function getEvalProgress(taskId: string): Promise<ProgressResponse> {
  return api<ProgressResponse>('/api/v1/eval/progress', { task_id: taskId })
}

export async function getEvalLog(taskId: string, startLine?: number, page = 500): Promise<LogResponse> {
  const params: Record<string, string> = { task_id: taskId, page: String(page) }
  if (startLine !== undefined) params.start_line = String(startLine)
  return api<LogResponse>('/api/v1/eval/log', params)
}

export function getEvalReportUrl(taskId: string): string {
  return `/api/v1/eval/report?task_id=${encodeURIComponent(taskId)}`
}

export async function stopEvalTask(taskId: string): Promise<{ status: string; task_id: string }> {
  return apiPost<{ status: string; task_id: string }>(`/api/v1/eval/stop?task_id=${encodeURIComponent(taskId)}`, {})
}

export async function resumeEvalTask(taskId: string, apiKey?: string): Promise<EvalInvokeResponse> {
  const body: Record<string, string> = { task_id: taskId }
  if (apiKey) body.api_key = apiKey
  // Resume can also run for hours — no client-side timeout.
  return apiPost<EvalInvokeResponse>('/api/v1/eval/resume/invoke', body, undefined, 0)
}

export async function listBenchmarks(type?: 'text' | 'multimodal' | 'aigc', all?: boolean): Promise<BenchmarksResponse> {
  const params: Record<string, string> = {}
  if (type) params.type = type
  if (all) params.all = 'true'
  return api<BenchmarksResponse>('/api/v1/eval/benchmarks', params)
}

// ── Batch evaluation ──

export interface EvalBatchUploadResponse {
  batch_id: string
  model_count: number
  models: string[]
  preview: {
    name: string
    model: string
    base_url: string
    api_key: string
    api: string
  }[]
}

export interface EvalBatchStatus {
  batch_id: string
  status: string
  total: number
  completed: number
  errors: number
  current_model: string
  current_task_id: string
  results: { task_id: string; name: string; model: string; eval_backend: string; status: string; error?: string }[]
  error_details: { name: string; model: string; error: string }[]
}

export function getEvalTemplateDownloadUrl(): string {
  return '/api/v1/eval/batch/template'
}

export async function uploadEvalBatchCsv(file: File): Promise<EvalBatchUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/v1/eval/batch/upload', { method: 'POST', body: form, headers: getAuthHeaders() })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(body.error || res.statusText)
  }
  return res.json()
}

export async function launchEvalBatch(
  batchId: string,
  sharedConfig: Record<string, unknown>,
): Promise<{ batch_id: string; total: number; status: string }> {
  return apiPost<{ batch_id: string; total: number; status: string }>(
    '/api/v1/eval/batch/launch', { batch_id: batchId, ...sharedConfig },
  )
}

export async function getEvalBatchStatus(batchId: string): Promise<EvalBatchStatus> {
  return api<EvalBatchStatus>(`/api/v1/eval/batch/status/${batchId}`)
}

export async function stopEvalBatch(batchId: string): Promise<{ batch_id: string; status: string }> {
  return apiPost<{ batch_id: string; status: string }>(`/api/v1/eval/batch/stop/${batchId}`, {})
}
