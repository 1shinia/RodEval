import { useEffect, useMemo, useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { useLocale } from '@/contexts/LocaleContext'

interface Props {
  content: string
  maxHeight?: string
  lineHeight?: number
}

export default function LogViewer({ content, maxHeight = '500px', lineHeight = 20 }: Props) {
  const { t } = useLocale()
  const parentRef = useRef<HTMLDivElement>(null)

  const lines = useMemo(() => {
    if (!content) return [t('common.loading')]
    return content.split('\n')
  }, [content, t])

  const height = useMemo(() => {
    const num = parseInt(maxHeight, 10)
    return isNaN(num) ? 500 : num
  }, [maxHeight])

  const virtualizer = useVirtualizer({
    count: lines.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => lineHeight,
    overscan: 10,
  })

  // Auto-scroll to bottom when content changes
  useEffect(() => {
    if (lines.length > 0) {
      virtualizer.scrollToIndex(lines.length - 1, { align: 'end' })
    }
  }, [lines.length, virtualizer])

  if (lines.length <= 50) {
    return (
      <pre
        className="text-xs p-4 rounded-[var(--radius-sm)] overflow-auto border border-[var(--border)]"
        style={{
          maxHeight,
          background: 'var(--bg-deep)',
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
        }}
      >
        {content || t('common.loading')}
      </pre>
    )
  }

  return (
    <div
      ref={parentRef}
      className="rounded-[var(--radius-sm)] border border-[var(--border)] overflow-auto"
      style={{ height, background: 'var(--bg-deep)' }}
    >
      <div
        style={{
          height: virtualizer.getTotalSize(),
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((item) => (
          <div
            key={item.key}
            style={{
              position: 'absolute',
              top: item.start,
              left: 0,
              width: '100%',
              height: item.size,
            }}
            className="whitespace-pre font-mono text-xs leading-5 text-[var(--text-muted)] px-4"
          >
            {lines[item.index]}
          </div>
        ))}
      </div>
    </div>
  )
}
