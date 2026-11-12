import { test } from 'node:test'
import assert from 'node:assert/strict'
import * as draft from '../src/components/eval/ragDraft.ts'
function store() {
  const data = new Map()
  return { getItem: k => data.get(k) ?? null, setItem: (k,v) => data.set(k,v), removeItem: k => data.delete(k), key: i => [...data.keys()][i], get length() { return data.size } }
}
test('reload restores configuration but excludes every credential', () => {
  const s = store()
  draft.saveRagDraft(s, 1, { ragModelPath: 'example', ragTool: 'reranker', ragApiKey: 'secret', ragasLlmKey: 'secret', ragasEmbKey: 'secret', clipApiKey: 'secret' }, 100)
  assert.equal(draft.loadRagDraft(s, 1, 101).ragModelPath, 'example')
  assert.equal(draft.loadRagDraft(s, 1, 101).ragTool, 'reranker')
  assert.ok(!s.getItem(s.key(0)).includes('secret'))
})
test('24-hour expiration removes draft', () => {
  const s = store(); draft.saveRagDraft(s, 1, { ragModelPath: 'old' }, 100)
  assert.equal(draft.loadRagDraft(s, 1, 100 + draft.RAG_DRAFT_TTL).ragModelPath, '')
  assert.equal(s.length, 0)
})
test('account switch removes other users draft', () => {
  const s = store(); draft.saveRagDraft(s, 1, { ragModelPath: 'private' }, 100)
  assert.equal(draft.loadRagDraft(s, 2, 101).ragModelPath, '')
  assert.equal(s.length, 0)
})
test('reset/logout clear only RAG drafts', () => {
  const s = store(); s.setItem('task-id', 'running'); draft.saveRagDraft(s, 1, {}, 100)
  draft.clearRagDrafts(s)
  assert.equal(s.getItem('task-id'), 'running'); assert.equal(s.length, 1)
})
test('corrupt, wrong-schema, future and unavailable storage never break editing', () => {
  for (const value of ['{', 'null', JSON.stringify({version: 2}), JSON.stringify({version: 1, updatedAt: 200})]) {
    const s = store(); s.setItem(draft.RAG_DRAFT_PREFIX + 1, value)
    assert.equal(draft.loadRagDraft(s, 1, 100).ragModelPath, '')
  }
  const blocked = { getItem() { throw Error('blocked') }, setItem() { throw Error('blocked') }, removeItem() { throw Error('blocked') } }
  assert.equal(draft.loadRagDraft(blocked, 1).ragModelPath, '')
  assert.doesNotThrow(() => draft.saveRagDraft(blocked, 1, {}))
})
test('invalid fields and hidden tools cannot be restored', () => {
  const s = store(); draft.saveRagDraft(s, 1, {ragTool: 'ragas', ragModelPath: 123, ragasMetrics: [123]}, 100)
  assert.deepEqual(draft.loadRagDraft(s, 1, 101), draft.RAG_DEFAULTS)
})
