import { api, apiPost, apiDelete, getAuthHeaders } from './client'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from './types'

export interface PerfTaskMeta {
  task_id: string
  model: string
  api: string
  dataset: string
  runs: number
  has_report: boolean
  timestamp: string
}

export interface ListPerfTasksResponse {
  tasks: PerfTaskMeta[]
  total: number
  page: number
  page_size: number
  root_path?: string
  filters?: {
    available_models: string[]
    available_datasets: string[]
  }
  error?: string
}

export async function listPerfTasks(params?: Record<string, string>): Promise<ListPerfTasksResponse> {
  const q: Record<string, string> = {}
  if (params) Object.assign(q, params)
  return api<ListPerfTasksResponse>('/api/v1/perf/list', q)
}

export async function submitPerfTask(
  payload: Record<string, unknown>,
  taskId: string,
): Promise<EvalInvokeResponse> {
  // Perf tasks can run for hours — no client-side timeout (0 = disable).
  // Progress is tracked via SSE; the HTTP response only arrives on completion.
  return apiPost<EvalInvokeResponse>('/api/v1/perf/invoke', payload, { 'EvalScope-Task-Id': taskId }, 0)
}

export async function getPerfProgress(taskId: string): Promise<ProgressResponse> {
  return api<ProgressResponse>('/api/v1/perf/progress', { task_id: taskId })
}

export async function getPerfLog(taskId: string, startLine?: number, page = 500): Promise<LogResponse> {
  const params: Record<string, string> = { task_id: taskId, page: String(page) }
  if (startLine !== undefined) params.start_line = String(startLine)
  return api<LogResponse>('/api/v1/perf/log', params)
}

export function getPerfReportUrl(taskId: string): string {
  return `/api/v1/perf/report?task_id=${encodeURIComponent(taskId)}`
}

export async function stopPerfTask(taskId: string): Promise<{ status: string; task_id: string }> {
  return apiPost<{ status: string; task_id: string }>(`/api/v1/perf/stop?task_id=${encodeURIComponent(taskId)}`, {})
}

export async function resumePerfTask(taskId: string, apiKey?: string): Promise<EvalInvokeResponse> {
  const body: Record<string, string> = { task_id: taskId }
  if (apiKey) body.api_key = apiKey
  // Resume can also run for hours — no client-side timeout.
  return apiPost<EvalInvokeResponse>('/api/v1/perf/resume/invoke', body, undefined, 0)
}

export async function deletePerfTask(taskId: string): Promise<{ ok: boolean }> {
  return apiDelete<{ ok: boolean }>('/api/v1/perf/delete', { task_id: taskId })
}

// Perf compare types
export interface PerfRunSummary {
  [key: string]: number
}

export interface PerfTaskCompare {
  task_id: string
  model: string
  dataset: string
  api: string
  runs: {
    run_name: string
    summary: Record<string, number>
    percentiles: Record<string, number>[]
    throughput: Record<string, unknown>
  }[]
}

export interface PerfCompareResponse {
  meta: {
    generated_at: string
    task_count: number
  }
  tasks: PerfTaskCompare[]
}

export async function comparePerfReports(taskIds: string[]): Promise<PerfCompareResponse> {
  return api<PerfCompareResponse>('/api/v1/perf/compare', {
    task_ids: taskIds.join(','),
  })
}

export interface SavedCompareReport {
  id: number
  name: string
  task_ids: string
  created_at: string
  task_count: number
  backend: string
  root_path: string
}

export async function saveCompareReport(name: string, taskIds: string[], backend?: string, rootPath?: string) {
  return apiPost<{ id: number }>('/api/v1/perf/compare/save', {
    name,
    task_ids: taskIds,
    backend: backend || 'Perf',
    root_path: rootPath || '',
  })
}

export async function listSavedCompareReports() {
  return api<{ reports: SavedCompareReport[] }>('/api/v1/perf/compare/saved')
}

export async function deleteCompareReport(id: number) {
  return apiDelete<{ ok: boolean }>('/api/v1/perf/compare/saved/' + id)
}

// Batch model testing
export function getTemplateDownloadUrl(): string {
  return '/api/v1/perf/template'
}

export interface BatchUploadResponse {
  batch_id: string
  model_count: number
  models: string[]
  preview: {
    name: string
    base_url: string
    api_key: string
    api: string
    model: string
    concurrency: string
    number: string
    max_tokens: number
    stream: boolean
    prompt: string
  }[]
}

export async function uploadBatchCsv(file: File): Promise<BatchUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/v1/perf/batch/upload', { method: 'POST', body: form, headers: getAuthHeaders() })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(body.error || `HTTP ${res.status}`)
  }
  return res.json()
}

export interface BatchRunResult {
  task_id: string
  name: string
  model: string
  status: string
  error?: string
}

export interface BatchRunResponse {
  batch_id: string
  completed: number
  errors: number
  results: BatchRunResult[]
  error_details: { name: string; model: string; error: string }[]
}

export async function launchBatchPerf(
  batchId: string,
  sharedConfig: Record<string, unknown>,
): Promise<{ batch_id: string; total: number; status: string }> {
  return apiPost<{ batch_id: string; total: number; status: string }>(
    '/api/v1/perf/batch/launch', { batch_id: batchId, ...sharedConfig },
  )
}

export interface BatchStatus {
  batch_id: string
  status: string          // 'running' | 'completed' | 'cancelled' | 'error'
  total: number
  completed: number
  errors: number
  current_model: string
  current_task_id: string
  results: BatchRunResult[]
  error_details: { name: string; model: string; error: string }[]
}

export async function getBatchStatus(batchId: string): Promise<BatchStatus> {
  return api<BatchStatus>(`/api/v1/perf/batch/status/${batchId}`)
}

export async function stopBatchPerf(batchId: string): Promise<{ batch_id: string; status: string }> {
  return apiPost<{ batch_id: string; status: string }>(`/api/v1/perf/batch/stop/${batchId}`, {})
}

// Legacy — kept for backward compatibility but forwards to launch
export async function runBatchPerf(
  batchId: string,
  sharedConfig: Record<string, unknown>,
): Promise<BatchRunResponse> {
  return apiPost<BatchRunResponse>('/api/v1/perf/batch/launch', { batch_id: batchId, ...sharedConfig }, undefined, 0)
}
