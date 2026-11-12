import {test} from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
const form = readFileSync(new URL('../src/components/eval/RAGEvalForm.tsx', import.meta.url), 'utf8')
test('RAG form integrates draft lifecycle and non-submitting reset', () => {
 assert.match(form, /useRagDraft/)
 assert.match(form, /type="button"[^>]*onClick=\{resetDraft\}/)
})
