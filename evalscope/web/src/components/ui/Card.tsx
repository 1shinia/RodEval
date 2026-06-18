import { useState, type ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { ChevronDown } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import Eyebrow from './Eyebrow'

interface CardProps {
  children: ReactNode
  className?: string
  title?: string
  badge?: ReactNode
  action?: ReactNode
  collapsible?: boolean
}

export default function Card({ children, className, title, badge, action, collapsible }: CardProps) {
  const [collapsed, setCollapsed] = useState(false)
  const { t } = useLocale()
  return (
    <div className={cn('rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] shadow-[var(--shadow-sm)]', className)}>
      {title && (
        <div className={cn('flex items-center justify-between px-5 py-3 border-b border-[var(--border)]', collapsible && 'cursor-pointer select-none')}
          onClick={collapsible ? () => setCollapsed((c) => !c) : undefined}
          role={collapsible ? 'button' : undefined}
          aria-expanded={collapsible ? !collapsed : undefined}
          aria-label={collapsible ? t('common.expand') : undefined}>
          <div className="flex items-center gap-2"><Eyebrow>{title}</Eyebrow>{badge}</div>
          <div className="flex items-center gap-2">
            {action}
            {collapsible && <ChevronDown size={14} className={cn('text-[var(--text-dim)] transition-transform duration-200', collapsed && '-rotate-90')} />}
          </div>
        </div>
      )}
      {!collapsed && <div className="p-5">{children}</div>}
    </div>
  )
}
