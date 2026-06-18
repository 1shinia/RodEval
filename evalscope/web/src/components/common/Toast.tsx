import { useEffect, useState, type ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { X, CheckCircle2, XCircle, AlertTriangle, Info } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'

type ToastType = 'success' | 'error' | 'warning' | 'info'

interface Toast {
  id: number
  type: ToastType
  message: string
}

let nextId = 0

const typeConfig: Record<ToastType, { icon: ReactNode; className: string }> = {
  success: {
    icon: <CheckCircle2 size={16} />,
    className: 'border-[var(--success)] bg-[var(--success-bg)]',
  },
  error: {
    icon: <XCircle size={16} />,
    className: 'border-[var(--danger)] bg-[var(--danger-bg)]',
  },
  warning: {
    icon: <AlertTriangle size={16} />,
    className: 'border-[var(--yellow)] bg-[var(--warning-bg)]',
  },
  info: {
    icon: <Info size={16} />,
    className: 'border-[var(--accent)] bg-[var(--accent-dim)]',
  },
}

const typeIconColor: Record<ToastType, string> = {
  success: 'text-[var(--success)]',
  error: 'text-[var(--danger)]',
  warning: 'text-[var(--yellow)]',
  info: 'text-[var(--accent)]',
}

export interface ToastApi {
  success: (msg: string) => void
  error: (msg: string) => void
  warning: (msg: string) => void
  info: (msg: string) => void
}

let externalAdd: ((t: Toast) => void) | null = null

/** Imperative API — call from anywhere without hook */
export const toast: ToastApi = {
  success: (msg) => externalAdd?.({ id: nextId++, type: 'success', message: msg }),
  error: (msg) => externalAdd?.({ id: nextId++, type: 'error', message: msg }),
  warning: (msg) => externalAdd?.({ id: nextId++, type: 'warning', message: msg }),
  info: (msg) => externalAdd?.({ id: nextId++, type: 'info', message: msg }),
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState<Toast[]>([])
  const { t } = useLocale()

  useEffect(() => {
    externalAdd = (t: Toast) => {
      setToasts((prev) => [...prev, t])
      setTimeout(() => {
        setToasts((prev) => prev.filter((x) => x.id !== t.id))
      }, 4000)
    }
    return () => { externalAdd = null }
  }, [])

  const dismiss = (id: number) => {
    setToasts((prev) => prev.filter((x) => x.id !== id))
  }

  if (toasts.length === 0) return null

  return (
    <div className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-2 pointer-events-none">
      {toasts.map((toast) => {
        const cfg = typeConfig[toast.type]
        return (
          <div
            key={toast.id}
            className={cn(
              'pointer-events-auto flex items-center gap-3 min-w-[280px] max-w-[420px]',
              'px-4 py-3 rounded-[var(--radius)] border shadow-[var(--shadow)]',
              'animate-[toast-in_0.25s_ease-out]',
              cfg.className,
            )}
          >
            <span className={cn('shrink-0', typeIconColor[toast.type])}>{cfg.icon}</span>
            <span className="text-sm text-[var(--text)] flex-1 break-words">{toast.message}</span>
            <button
              onClick={() => dismiss(toast.id)}
              aria-label={t('common.close')}
              className="shrink-0 text-[var(--text-muted)] hover:text-[var(--text)] transition-colors cursor-pointer"
            >
              <X size={14} />
            </button>
          </div>
        )
      })}
    </div>
  )
}
