// RAG-only, short-lived form drafts. Credentials are intentionally not in this schema.
export const RAG_DRAFT_PREFIX = 'evalperf:rag-draft:'
export const RAG_DRAFT_TTL = 24 * 60 * 60 * 1000
export const RAG_DEFAULTS = {
  ragTool: 'embedding' as 'embedding' | 'reranker',
  ragModelPath: '', ragApiBase: '', ragTaskTypes: '', ragTaskNames: '',
  ragLanguages: '', ragLimit: '', ragRandomSample: false,
  truncateInput: false, maxInputChars: '2000', ragBatchSize: '',
  ragPrompt: '', ragDataHub: 'huggingface',
  ragasTestset: '', ragasLlmModel: '', ragasLlmBase: '',
  ragasEmbModel: '', ragasEmbProv: 'huggingface', ragasEmbBase: '',
  ragasMetrics: ['answer_relevancy', 'faithfulness'],
  clipModelPath: '', clipApiBase: '', clipDatasets: '', clipBatchSize: '128', clipLimit: '',
}
export type RagDraft = typeof RAG_DEFAULTS

function sanitize(value: unknown): RagDraft {
  const result = { ...RAG_DEFAULTS, ragasMetrics: [...RAG_DEFAULTS.ragasMetrics] }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return result
  const source = value as Record<string, unknown>
  for (const key of Object.keys(result) as (keyof RagDraft)[]) {
    const field = source[key]
    if (key === 'ragTool') {
      if (field === 'embedding' || field === 'reranker') result.ragTool = field
    } else if (key === 'ragasMetrics') {
      if (Array.isArray(field) && field.every(x => typeof x === 'string')) result.ragasMetrics = [...field]
    } else if (typeof field === typeof result[key]) {
      Object.assign(result, { [key]: field })
    }
  }
  return result
}

export function clearRagDrafts(storage: Storage) {
  try {
    for (let i = storage.length - 1; i >= 0; i--) {
      const key = storage.key(i)
      if (key?.startsWith(RAG_DRAFT_PREFIX)) storage.removeItem(key)
    }
  } catch { /* Storage can be disabled by browser policy. */ }
}

export function loadRagDraft(storage: Storage, userId: number | undefined, now = Date.now()): RagDraft {
  if (userId === undefined) return sanitize(null)
  const key = RAG_DRAFT_PREFIX + userId
  try {
    // Never keep another account's RAG draft in this tab.
    for (let i = storage.length - 1; i >= 0; i--) {
      const other = storage.key(i)
      if (other?.startsWith(RAG_DRAFT_PREFIX) && other !== key) storage.removeItem(other)
    }
    const raw = storage.getItem(key)
    if (!raw) return sanitize(null)
    const saved = JSON.parse(raw)
    if (saved.version !== 1 || typeof saved.updatedAt !== 'number' || !Number.isFinite(saved.updatedAt)
      || now < saved.updatedAt || now - saved.updatedAt >= RAG_DRAFT_TTL) {
      storage.removeItem(key)
      return sanitize(null)
    }
    return sanitize(saved.values)
  } catch {
    try { storage.removeItem(key) } catch { /* unavailable */ }
    return sanitize(null)
  }
}

export function saveRagDraft(storage: Storage, userId: number | undefined, values: unknown, now = Date.now()) {
  if (userId === undefined) return
  try {
    storage.setItem(RAG_DRAFT_PREFIX + userId, JSON.stringify({ version: 1, updatedAt: now, values: sanitize(values) }))
  } catch { /* Form editing must still work when storage is unavailable. */ }
}
