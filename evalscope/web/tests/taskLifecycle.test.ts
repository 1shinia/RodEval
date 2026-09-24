import assert from 'node:assert/strict'
import test from 'node:test'

import { classifyProgress, progressRetryDelay, shouldShowProgressError } from '../src/hooks/taskLifecycle.ts'

test('classifies every backend terminal status consistently', () => {
  assert.deepEqual(classifyProgress({ percent: 100, status: 'completed' }), { status: 'ok' })
  assert.deepEqual(classifyProgress({ percent: 42, status: 'error' }), { status: 'error', error: '任务执行失败' })
  assert.deepEqual(classifyProgress({ percent: 42, status: 'failed' }), { status: 'error', error: '任务执行失败' })
  assert.deepEqual(classifyProgress({ percent: 42, status: 'orphaned' }), { status: 'error', error: '服务重启后任务已中断' })
  assert.deepEqual(classifyProgress({ percent: 42, status: 'stopped' }), { status: 'stopped' })
  assert.deepEqual(classifyProgress({ percent: 42, status: 'cancelled' }), { status: 'stopped' })
})

test('keeps active and unknown states non-terminal', () => {
  assert.equal(classifyProgress({ percent: 99, status: 'running' }), null)
  assert.equal(classifyProgress({ percent: 0, status: 'queued' }), null)
  assert.equal(classifyProgress({ percent: 100, status: 'running' }), null)
  assert.equal(classifyProgress({ percent: 30, status: 'unexpected' }), null)
})

test('preserves legacy status-less 100 percent completion', () => {
  assert.deepEqual(classifyProgress({ percent: 100 }), { status: 'ok' })
})

test('shows progress errors only after repeated polling failures', () => {
  assert.equal(shouldShowProgressError(1), false)
  assert.equal(shouldShowProgressError(2), true)
  assert.equal(shouldShowProgressError(3), true)
})

test('backs off progress retries with a bounded delay', () => {
  assert.deepEqual([1, 2, 3, 4, 5, 6].map(progressRetryDelay), [3000, 6000, 12000, 24000, 30000, 30000])
})
