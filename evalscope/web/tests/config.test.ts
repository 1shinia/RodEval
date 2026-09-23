import assert from 'node:assert/strict'
import test from 'node:test'

import { createConfigLoader } from '../src/api/config.ts'
import { createInFlightDeduper } from '../src/api/requestDeduper.ts'

test('deduplicates concurrent and repeated config loads', async () => {
  let calls = 0
  const load = createConfigLoader(async () => {
    calls += 1
    return new Response(JSON.stringify({ outputs_root: '/tmp/outputs' }), {
      headers: { 'Content-Type': 'application/json' },
    })
  })

  const [first, second] = await Promise.all([load(), load()])
  const third = await load()

  assert.equal(calls, 1)
  assert.equal(first.outputs_root, '/tmp/outputs')
  assert.strictEqual(first, second)
  assert.strictEqual(second, third)
})

test('retries config loading after a failed request', async () => {
  let calls = 0
  const load = createConfigLoader(async () => {
    calls += 1
    if (calls === 1) return new Response('', { status: 503 })
    return new Response(JSON.stringify({ registration_mode: 'invite' }), {
      headers: { 'Content-Type': 'application/json' },
    })
  })

  await assert.rejects(load(), /Failed to load application config/)
  assert.equal((await load()).registration_mode, 'invite')
  assert.equal(calls, 2)
})

test('deduplicates only concurrent requests with the same key', async () => {
  let calls = 0
  const run = createInFlightDeduper(async (key: string) => {
    calls += 1
    await Promise.resolve()
    return `${key}:${calls}`
  })

  const [first, second] = await Promise.all([run('reports'), run('reports')])
  const third = await run('reports')

  assert.equal(calls, 2)
  assert.equal(first, second)
  assert.notEqual(second, third)
})
