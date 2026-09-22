import type { ProgressResponse } from '@/api/types'

export type ProgressOutcome =
  | { status: 'ok' }
  | { status: 'error'; error: string }
  | { status: 'stopped' }

const FAILED_STATUSES = new Set(['error', 'failed'])
const STOPPED_STATUSES = new Set(['stopped', 'cancelled'])

export function classifyProgress(progress: ProgressResponse): ProgressOutcome | null {
  const status = typeof progress.status === 'string' ? progress.status.toLowerCase() : ''

  if (status === 'completed' || (!status && (progress.percent ?? 0) >= 100)) {
    return { status: 'ok' }
  }
  if (FAILED_STATUSES.has(status)) {
    return { status: 'error', error: '任务执行失败' }
  }
  if (status === 'orphaned') {
    return { status: 'error', error: '服务重启后任务已中断' }
  }
  if (STOPPED_STATUSES.has(status)) {
    return { status: 'stopped' }
  }
  return null
}

export function progressRetryDelay(failures: number): number {
  return Math.min(3000 * (2 ** Math.max(0, failures - 1)), 30000)
}
