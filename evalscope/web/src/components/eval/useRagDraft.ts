import { useRef, useState, type SetStateAction } from 'react'
import { loadRagDraft, saveRagDraft, clearRagDrafts, RAG_DEFAULTS, type RagDraft } from './ragDraft'

export function useRagDraft(userId: number | undefined) {
  const [values, setValues] = useState(() => {
    try { return loadRagDraft(window.sessionStorage, userId) } catch { return { ...RAG_DEFAULTS } }
  })
  const current = useRef(values)
  function field<K extends keyof RagDraft>(key: K): [RagDraft[K], (value: SetStateAction<RagDraft[K]>) => void] {
    return [values[key], value => {
      const next = { ...current.current, [key]: typeof value === 'function' ? value(current.current[key]) : value }
      current.current = next
      setValues(next)
      try { saveRagDraft(window.sessionStorage, userId, next) } catch { /* unavailable */ }
    }]
  }
  function reset() {
    const next = { ...RAG_DEFAULTS, ragasMetrics: [...RAG_DEFAULTS.ragasMetrics] }
    current.current = next
    setValues(next)
    try { clearRagDrafts(window.sessionStorage) } catch { /* unavailable */ }
  }
  return { field, reset }
}
