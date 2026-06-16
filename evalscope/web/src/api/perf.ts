import { apiPost, api } from './client'
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
  return apiPost<EvalInvokeResponse>('/api/v1/perf/invoke', payload, { 'EvalScope-Task-Id': taskId })
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

export async function deletePerfTask(taskId: string): Promise<{ ok: boolean }> {
  const BASE = '/api/v1/perf'
  const res = await fetch(`${BASE}/delete`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(body.error || `HTTP ${res.status}`)
  }
  return res.json()
}
