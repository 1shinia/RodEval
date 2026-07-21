import { useEffect, useState, useMemo } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { getAIGCReport } from '@/api/reports'
import { toast } from '@/components/common/Toast'
import type { AIGCReportResponse, AIGCSampleResult } from '@/api/types'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Skeleton from '@/components/ui/Skeleton'
import Lightbox from '@/components/aigc/Lightbox'

export default function AIGCReportDetailPage() {
  const { taskId } = useParams<{ taskId: string }>()
  const [searchParams] = useSearchParams()
  const { t } = useLocale()

  const [data, setData] = useState<AIGCReportResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null)

  // Load report on mount
  useEffect(() => {
    if (!taskId) return
    let cancelled = false
    setLoading(true)
    setError('')

    getAIGCReport(taskId)
      .then((res) => {
        if (!cancelled) setData(res)
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err)
        if (!cancelled) setError(msg)
        toast.error(msg)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => { cancelled = true }
  }, [taskId])

  const samples = data?.per_sample ?? []

  // Aggregate metrics display
  const metricsSummary = useMemo(() => {
    if (!data?.metrics) return []
    return Object.entries(data.metrics).map(([key, value]) => ({
      label: key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
      value: typeof value === 'number' ? value.toFixed(4) : String(value),
    }))
  }, [data])

  if (loading) {
    return (
      <div className="page-enter p-6 flex flex-col gap-4">
        <Skeleton width={300} height={20} />
        <Skeleton width="100%" height={120} />
        <Skeleton lines={8} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="page-enter p-6">
        <Breadcrumb
          items={[
            { label: t('reports.title'), href: '/reports' },
            { label: t('aigc.reportDetail') || 'AIGC 报告详情' },
          ]}
        />
        <div className="mt-6 p-6 rounded-[var(--radius)] border border-[var(--danger)] bg-[var(--danger-bg)] text-[var(--danger)]">
          <p className="text-sm">{t('reportDetail.failedToLoad')}: {error}</p>
        </div>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="page-enter flex flex-col gap-5 p-6">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { label: t('reports.title'), href: '/reports' },
          { label: data.model || taskId || '' },
        ]}
      />

      {/* Header Card */}
      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold text-[var(--text)] mb-1">
              {data.model}
            </h1>
            <p className="text-sm text-[var(--text-muted)]">
              {data.model_type === 'txt2img' ? '文生图' : data.model_type === 'txt2video' ? '文生视频' : '图生图'}
              {' · '}
              {data.num_samples} 张样本
              {' · '}
              耗时 {data.generation_time.toFixed(1)}s
            </p>
          </div>

          {/* Metrics badges */}
          {metricsSummary.length > 0 && (
            <div className="flex flex-wrap gap-3">
              {metricsSummary.map((m) => (
                <div
                  key={m.label}
                  className="px-4 py-2 rounded-[var(--radius-sm)] bg-[var(--bg-card2)] border border-[var(--border)]"
                >
                  <div className="text-xs text-[var(--text-muted)]">{m.label}</div>
                  <div className="text-lg font-semibold text-[var(--accent)]">{m.value}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Image Gallery */}
      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-6">
        <h2 className="text-lg font-semibold text-[var(--text)] mb-4">
          生成结果
        </h2>

        {samples.length === 0 ? (
          <p className="text-sm text-[var(--text-muted)] py-8 text-center">
            暂无生成结果
          </p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {samples.map((sample, idx) => (
              <SampleCard
                key={sample.index}
                sample={sample}
                onClick={() => setLightboxIndex(idx)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Lightbox */}
      {lightboxIndex !== null && samples[lightboxIndex] && (
        <Lightbox
          url={samples[lightboxIndex].url}
          alt={samples[lightboxIndex].prompt}
          onClose={() => setLightboxIndex(null)}
          onPrev={() => setLightboxIndex((i) => (i !== null && i > 0 ? i - 1 : i))}
          onNext={() => setLightboxIndex((i) => (i !== null && i < samples.length - 1 ? i + 1 : i))}
          hasPrev={lightboxIndex > 0}
          hasNext={lightboxIndex < samples.length - 1}
        />
      )}
    </div>
  )
}

// ---- Sample Card Component ----
function SampleCard({ sample, onClick }: { sample: AIGCSampleResult; onClick: () => void }) {
  return (
    <div
      className="group cursor-pointer rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] overflow-hidden hover:border-[var(--accent)] hover:shadow-lg transition-all duration-200"
      onClick={onClick}
    >
      {/* Thumbnail */}
      <div className="aspect-square overflow-hidden bg-[var(--bg-card2)]">
        <img
          src={sample.thumbnail_url}
          alt={sample.prompt}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          loading="lazy"
        />
      </div>

      {/* Info */}
      <div className="p-3">
        <p className="text-xs text-[var(--text-muted)] line-clamp-2 mb-1" title={sample.prompt}>
          {sample.prompt}
        </p>
        {sample.clip_score !== undefined && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-[var(--text-dim)]">CLIP:</span>
            <span className="text-xs font-medium text-[var(--accent)]">
              {sample.clip_score.toFixed(3)}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
