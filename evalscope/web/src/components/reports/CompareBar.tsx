import { GitCompareArrows } from 'lucide-react'
import Button from '@/components/ui/Button'
import { useLocale } from '@/contexts/LocaleContext'
import { useCompare } from '@/contexts/CompareContext'

interface CompareBarProps {
  selected: string[]
  totalCount: number
  rootPath: string
  backend: string
  onSelectAll: () => Promise<void> | void
  onClear: () => void
  loading?: boolean
  compareUrl?: string
}

export default function CompareBar({
  selected,
  totalCount,
  rootPath,
  backend,
  onSelectAll,
  onClear,
  loading,
  compareUrl,
}: CompareBarProps) {
  const { t } = useLocale()
  const { setCompareSelection } = useCompare()

  const handleCompare = () => {
    setCompareSelection({ reports: selected, rootPath, backend })
    window.open(compareUrl || `/compare?root_path=${encodeURIComponent(rootPath)}`, '_blank')
  }

  const hasSelection = selected.length > 0

  return (
    <div className="sticky top-0 z-20 flex items-center gap-3 px-4 py-2.5 rounded-[var(--radius)] border border-[var(--accent)] bg-[var(--accent-dim)] shadow-[var(--shadow)]">
      <span className="text-sm font-medium text-[var(--accent)]">
        {t('reports.selected')} {selected.length} / {totalCount}
      </span>
      <button
        type="button"
        onClick={onSelectAll}
        disabled={loading}
        className="text-xs text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors cursor-pointer disabled:opacity-50"
      >
        {selected.length >= totalCount ? t('reports.deselectAll') : `${t('reports.selectAll')} (${totalCount})`}
      </button>
      {hasSelection && (
        <button
          type="button"
          onClick={onClear}
          className="text-xs text-[var(--text-muted)] hover:text-[var(--danger)] transition-colors cursor-pointer"
        >
          {t('reports.clearSelection')}
        </button>
      )}
      <div className="flex-1" />
      <Button size="sm" onClick={handleCompare} disabled={!hasSelection}>
        <GitCompareArrows size={14} />
        {t('reports.compare')}
      </Button>
    </div>
  )
}
