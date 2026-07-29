import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

interface CompareSelection {
  reports: string[]
  rootPath: string
  backend: string
}

interface CompareCtx {
  selection: CompareSelection | null
  setCompareSelection: (sel: CompareSelection) => void
  clearCompareSelection: () => void
}

const CompareContext = createContext<CompareCtx>(null!)

export function CompareProvider({ children }: { children: ReactNode }) {
  const [selection, setSelection] = useState<CompareSelection | null>(() => {
    try {
      const raw = sessionStorage.getItem('compare_selection')
      if (raw) return JSON.parse(raw) as CompareSelection
    } catch { /* ignore */ }
    return null
  })

  const setCompareSelection = useCallback((sel: CompareSelection) => {
    sessionStorage.setItem('compare_selection', JSON.stringify(sel))
    setSelection(sel)
  }, [])

  const clearCompareSelection = useCallback(() => {
    sessionStorage.removeItem('compare_selection')
    setSelection(null)
  }, [])

  const value = useMemo<CompareCtx>(
    () => ({ selection, setCompareSelection, clearCompareSelection }),
    [selection, setCompareSelection, clearCompareSelection],
  )

  return <CompareContext.Provider value={value}>{children}</CompareContext.Provider>
}

export function useCompare() {
  return useContext(CompareContext)
}
