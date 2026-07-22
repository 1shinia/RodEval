import { Outlet, NavLink } from 'react-router-dom'
import { FileText, Image as ImageIcon } from 'lucide-react'
import type { ReactNode } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import Breadcrumb from '@/components/ui/Breadcrumb'
import ServerBadge from '@/components/ui/ServerBadge'

export default function ReportsLayout() {
  const { t } = useLocale()
  const { serverAddress } = useReports()

  const tabClass = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all duration-200 ${
      isActive
        ? 'border-[var(--accent)] text-[var(--accent)]'
        : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
    }`

  return (
    <div className="page-enter flex flex-col gap-5">
      {/* Header with Breadcrumb and Server Address */}
      <div className="flex items-center justify-between">
        <Breadcrumb items={[{ label: t('reports.title') }]} />
        <ServerBadge address={serverAddress} />
      </div>

      {/* Tab bar */}
      <div className="flex items-center gap-1 border-b border-[var(--border)]">
        <NavLink to="/reports/llm" className={tabClass}>
          <FileText size={16} />
          {t('reports.tabLLM')}
        </NavLink>
        <NavLink to="/reports/aigc" className={tabClass}>
          <ImageIcon size={16} />
          {t('reports.tabAIGC')}
        </NavLink>
      </div>

      <Outlet />
    </div>
  )
}

/** Shared empty state component used by both LLM and AIGC tabs */
export function EmptyState({
  icon,
  title,
  subtitle,
}: {
  icon: ReactNode
  title: string
  subtitle: string
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3">
      {/* text-dim allowed: empty-state icon (DESIGN.md §Text) */}
      <div className="text-[var(--text-dim)]">{icon}</div>
      <h3 className="text-lg font-semibold text-[var(--text)]">{title}</h3>
      <p className="text-sm text-[var(--text-muted)]">{subtitle}</p>
    </div>
  )
}
