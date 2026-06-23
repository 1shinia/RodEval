import { Server } from 'lucide-react'

interface ServerBadgeProps {
  address: string
}

/**
 * ServerBadge — displays the backend server address in a subtle badge.
 */
export default function ServerBadge({ address }: ServerBadgeProps) {
  if (!address) return null

  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text-dim)] type-body-xs">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
      <Server size={12} strokeWidth={2} />
      <span>{address}</span>
    </span>
  )
}
