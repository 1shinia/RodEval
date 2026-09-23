import type { ProgressResponse } from '@/api/types'

export const TASK_STATUSES = {
  QUEUED: 'queued',
  STARTING: 'starting',
  RUNNING: 'running',
  CANCELLING: 'cancelling',
  COMPLETED: 'completed',
  PARTIAL_SUCCESS: 'partial_success',
  FAILED: 'failed',
  STOPPED: 'stopped',
  ORPHANED: 'orphaned',
} as const

export type TaskStatus = typeof TASK_STATUSES[keyof typeof TASK_STATUSES]
export const ACTIVE_TASK_STATUSES = new Set<TaskStatus>([
  TASK_STATUSES.QUEUED,
  TASK_STATUSES.STARTING,
  TASK_STATUSES.RUNNING,
  TASK_STATUSES.CANCELLING,
])
export const TERMINAL_TASK_STATUSES = new Set<TaskStatus>([
  TASK_STATUSES.COMPLETED,
  TASK_STATUSES.PARTIAL_SUCCESS,
  TASK_STATUSES.FAILED,
  TASK_STATUSES.STOPPED,
  TASK_STATUSES.ORPHANED,
])

const LEGACY_STATUS_ALIASES: Record<string, TaskStatus> = {
  cancelled: TASK_STATUSES.STOPPED,
  error: TASK_STATUSES.FAILED,
  ok: TASK_STATUSES.COMPLETED,
  success: TASK_STATUSES.COMPLETED,
  completed_with_warnings: TASK_STATUSES.PARTIAL_SUCCESS,
}


export function normalizeTaskStatus(status?: string | null): TaskStatus | null {
  const value = status?.toLowerCase()
  if (!value) return null
  if ((Object.values(TASK_STATUSES) as string[]).includes(value)) return value as TaskStatus
  return LEGACY_STATUS_ALIASES[value] ?? null
}

export function isActiveTaskStatus(status?: string | null): boolean {
  const normalized = normalizeTaskStatus(status)
  return normalized ? ACTIVE_TASK_STATUSES.has(normalized) : false
}

export function isTerminalTaskStatus(status?: string | null): boolean {
  const normalized = normalizeTaskStatus(status)
  return normalized ? TERMINAL_TASK_STATUSES.has(normalized) : false
}

export type ProgressOutcome =
  | { status: 'ok' }
  | { status: 'partial_success'; error?: string }
  | { status: 'error'; error: string }
  | { status: 'stopped' }

export function classifyProgress(progress: ProgressResponse): ProgressOutcome | null {
  const status = normalizeTaskStatus(typeof progress.status === 'string' ? progress.status : null)

  if (status === TASK_STATUSES.COMPLETED) {
    return { status: 'ok' }
  }
  if (status === TASK_STATUSES.PARTIAL_SUCCESS) {
    return { status: 'partial_success', error: '任务完成，但存在失败项' }
  }
  if (status === TASK_STATUSES.FAILED || status === TASK_STATUSES.ORPHANED) {
    return { status: 'error', error: status === TASK_STATUSES.ORPHANED ? '服务重启后任务已中断' : '任务执行失败' }
  }
  if (status === TASK_STATUSES.STOPPED) {
    return { status: 'stopped' }
  }
  return null
}

export function progressRetryDelay(failures: number): number {
  return Math.min(3000 * (2 ** Math.max(0, failures - 1)), 30000)
}
