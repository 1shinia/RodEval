import { GitCompareArrows } from 'lucide-react'
import Button from '@/components/ui/Button'
import { useCompare } from '@/contexts/CompareContext'

interface CompareBarProps {
  selected: string[]
  totalCount: number
  rootPath: string
  backend: string
  onSelectAll: () => Promise<void> | void
  onClear: () => void
  loading?: boolean
}

export default function CompareBar({
  selected,
  totalCount,
  rootPath,
  backend,
  onSelectAll,
  onClear,
  loading,
}: CompareBarProps) {
  const { setCompareSelection } = useCompare()

  const handleCompare = () => {
    setCompareSelection({ reports: selected, rootPath, backend })
    window.location.href = `/compare?root_path=${encodeURIComponent(rootPath)}`
  }

  const hasSelection = selected.length > 0

  return (
    <div className="sticky top-0 z-20 flex items-center gap-3 px-4 py-2.5 rounded-[var(--radius)] border border-[var(--accent)] bg-[var(--accent-dim)] shadow-[var(--shadow)]">
      <span className="text-sm font-medium text-[var(--accent)]">
        已选 {selected.length} / {totalCount}
      </span>
      <button
        type="button"
        onClick={onSelectAll}
        disabled={loading}
        className="text-xs text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors cursor-pointer disabled:opacity-50"
      >
        {selected.length >= totalCount ? '取消全选' : `全选 (${totalCount})`}
      </button>
      {hasSelection && (
        <button
          type="button"
          onClick={onClear}
          className="text-xs text-[var(--text-muted)] hover:text-[var(--danger)] transition-colors cursor-pointer"
        >
          清除
        </button>
      )}
      <div className="flex-1" />
      <Button size="sm" onClick={handleCompare} disabled={!hasSelection}>
        <GitCompareArrows size={14} />
        对比
      </Button>
    </div>
  )
}
