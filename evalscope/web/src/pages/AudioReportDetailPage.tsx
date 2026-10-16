import { useCallback, useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { api } from '@/api/client'
import Skeleton from '@/components/ui/Skeleton'
import { useLocale } from '@/contexts/LocaleContext'

interface ASRSample {
  audio_path?: string
  audio_url?: string
  reference?: string
  hypothesis?: string
  wer?: number
  cer?: number
  language?: string
  elapsed_seconds?: number
}

interface TTSSample {
  index: number
  prompt?: string
  audio_path?: string
  audio_url?: string
  elapsed_seconds?: number
  error?: string
}

interface AudioReport {
  tool: string
  model: string
  asr_model?: string
  metrics?: Record<string, number>
  per_sample?: ASRSample | ASRSample[] | TTSSample[]
  elapsed_seconds?: number
  num_samples?: number
  total_elapsed_seconds?: number
}

interface AudioSample {
  index?: number
  reference?: string
  hypothesis?: string
  wer?: number
  cer?: number
  language?: string
  elapsed_seconds?: number
  audio_url?: string
  prompt?: string
  error?: string
  duration_seconds?: number
  asr_elapsed_seconds?: number
}

export default function AudioReportDetailPage() {
  const { t } = useLocale()
  const { taskId } = useParams<{ taskId: string }>()
  const navigate = useNavigate()

  const [report, setReport] = useState<AudioReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchReport = useCallback(async () => {
    if (!taskId) return
    setLoading(true)
    setError(null)
    try {
      const data = await api<AudioReport>(`/api/v1/audio/report/${encodeURIComponent(taskId)}`)
      setReport(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : t('reports.loadFailed'))
    } finally {
      setLoading(false)
    }
  }, [taskId, t])

  useEffect(() => {
    // The callback performs the asynchronous fetch and updates loading/error state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchReport()
  }, [fetchReport])

  if (loading) {
    return (
      <div className="page-enter flex flex-col gap-4 max-w-4xl mx-auto">
        <Skeleton className="h-8 w-40" />
        <Skeleton className="h-64 w-full rounded-lg" />
      </div>
    )
  }

  if (error || !report) {
    return (
      <div className="page-enter max-w-2xl mx-auto">
        <button onClick={() => navigate(-1)} className="flex items-center gap-2 text-sm text-[var(--text-muted)] hover:text-[var(--text)] mb-4">
          <ArrowLeft size={16} /> {t('evalAudioExtra.detailBack')}
        </button>
        <div className="px-4 py-3 rounded-[var(--radius)] bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger)]">
          {error || t('evalAudioExtra.detailNotFound')}
        </div>
      </div>
    )
  }

  const isASR = report.tool === 'asr'
  const samples: AudioSample[] = report.per_sample
    ? (Array.isArray(report.per_sample) ? report.per_sample : [report.per_sample])
    : []

  return (
    <div className="page-enter flex flex-col gap-5 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <button onClick={() => navigate(-1)} className="flex items-center gap-2 text-sm text-[var(--text-muted)] hover:text-[var(--text)]">
          <ArrowLeft size={16} /> {t('evalAudioExtra.detailBack')}
        </button>
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
          isASR ? 'bg-blue-500/10 text-blue-400' : 'bg-green-500/10 text-green-400'
        }`}>
          {isASR ? 'ASR' : 'TTS'}
        </span>
      </div>

      <h1 className="text-2xl font-bold text-[var(--text)]">
        {isASR ? t('evalAudioExtra.detailAsrTitle') : t('evalAudioExtra.detailTtsTitle')}
      </h1>

      {/* Model Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
        <div>
          <div className="text-xs text-[var(--text-muted)]">{t('evalAudioExtra.detailModel')}</div>
          <div className="text-sm font-medium text-[var(--text)]">{report.model}</div>
        </div>
        <div>
          <div className="text-xs text-[var(--text-muted)]">{t('evalAudioExtra.detailTool')}</div>
          <div className="text-sm font-medium text-[var(--text)]">{isASR ? t('evalAudioExtra.asrOption') : t('evalAudioExtra.ttsOption')}</div>
        </div>
        <div>
          <div className="text-xs text-[var(--text-muted)]">{t('evalAudioExtra.detailElapsed')}</div>
          <div className="text-sm font-medium text-[var(--text)]">
            {isASR
              ? `${report.elapsed_seconds?.toFixed(1) ?? '-'} ${t('evalAudioExtra.detailSeconds')}`
              : `${report.total_elapsed_seconds?.toFixed(1) ?? '-'} ${t('evalAudioExtra.detailSeconds')}`}
          </div>
        </div>
      </div>

      {/* TTS→ASR Closed-Loop Metrics */}
      {!isASR && report.metrics && (
        <div className="grid grid-cols-2 gap-4">
          {report.metrics.wer_avg != null && (
            <div className="p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
              <div className="text-xs text-[var(--text-muted)] mb-1">{t('evalAudioExtra.detailMeanWer')}</div>
              <div className={`text-2xl font-bold font-mono ${report.metrics.wer_avg < 0.1 ? 'text-green-400' : report.metrics.wer_avg < 0.3 ? 'text-yellow-400' : 'text-red-400'}`}>
                {(report.metrics.wer_avg * 100).toFixed(2)}%
              </div>
              <div className="text-xs text-[var(--text-dim)] mt-1">{t('evalAudioExtra.detailLowerBetter')} · ASR: {report.asr_model}</div>
            </div>
          )}
          {report.metrics.cer_avg != null && (
            <div className="p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
              <div className="text-xs text-[var(--text-muted)] mb-1">{t('evalAudioExtra.detailMeanCer')}</div>
              <div className={`text-2xl font-bold font-mono ${report.metrics.cer_avg < 0.1 ? 'text-green-400' : report.metrics.cer_avg < 0.3 ? 'text-yellow-400' : 'text-red-400'}`}>
                {(report.metrics.cer_avg * 100).toFixed(2)}%
              </div>
              <div className="text-xs text-[var(--text-dim)] mt-1">{t('evalAudioExtra.detailLowerBetter')} · ASR: {report.asr_model}</div>
            </div>
          )}
        </div>
      )}

      {/* ASR Metrics */}
      {isASR && report.metrics && (
        <div className="grid grid-cols-2 gap-4">
          {report.metrics.wer != null && (
            <div className="p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
              <div className="text-xs text-[var(--text-muted)] mb-1">{t('evalAudioExtra.detailWer')}</div>
              <div className={`text-2xl font-bold font-mono ${report.metrics.wer < 0.2 ? 'text-green-400' : report.metrics.wer < 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>
                {(report.metrics.wer * 100).toFixed(2)}%
              </div>
              <div className="text-xs text-[var(--text-dim)] mt-1">{t('evalAudioExtra.detailLowerBetter')}</div>
            </div>
          )}
          {report.metrics.cer != null && (
            <div className="p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
              <div className="text-xs text-[var(--text-muted)] mb-1">{t('evalAudioExtra.detailCer')}</div>
              <div className={`text-2xl font-bold font-mono ${report.metrics.cer < 0.2 ? 'text-green-400' : report.metrics.cer < 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>
                {(report.metrics.cer * 100).toFixed(2)}%
              </div>
              <div className="text-xs text-[var(--text-dim)] mt-1">{t('evalAudioExtra.detailLowerBetter')}</div>
            </div>
          )}
        </div>
      )}

      {/* TTS Summary */}
      {isASR && report.metrics?.wer == null && report.metrics?.cer == null && !samples[0]?.reference?.trim() && (
        <p className="text-sm text-[var(--text-muted)]">{t('evalAudioExtra.detailNoMetrics')}</p>
      )}
      {!isASR && (
        <div className="p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <div className="text-xs text-[var(--text-muted)] mb-2">{t('evalAudioExtra.detailSummary')}</div>
          <div className="flex gap-6">
            <div>
              <div className="text-2xl font-bold text-[var(--accent)]">{report.num_samples || samples.length}</div>
              <div className="text-xs text-[var(--text-dim)]">{t('evalAudioExtra.detailSamples')}</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-[var(--text)]">{report.total_elapsed_seconds?.toFixed(1) || '-'}s</div>
              <div className="text-xs text-[var(--text-dim)]">{t('evalAudioExtra.detailTotalElapsed')}</div>
            </div>
          </div>
        </div>
      )}

      {/* Sample Details */}
      {samples.length > 0 && (
        <>
          <h2 className="text-lg font-semibold text-[var(--text)] border-b border-[var(--border)] pb-2">
            {isASR ? t('evalAudioExtra.detailRecognition') : t('evalAudioExtra.detailAudioSamples')}
          </h2>
          {samples.map((sample, i) => (
            <div key={i} className="p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
              {isASR ? (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                    <div>
                      <div className="text-xs text-[var(--text-muted)] mb-1">{t('evalAudioExtra.detailReference')}</div>
                      <div className="text-sm text-[var(--text)] bg-[var(--bg-card2)] p-2 rounded">{sample.reference || t('evalAudioExtra.detailNoReference')}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[var(--text-muted)] mb-1">{t('evalAudioExtra.detailRecognition')}</div>
                      <div className="text-sm text-[var(--text)] bg-[var(--bg-card2)] p-2 rounded">{sample.hypothesis || '-'}</div>
                    </div>
                  </div>
                  <div className="flex gap-4 text-xs">
                    {sample.wer != null && <span className="text-[var(--accent)]">WER: {(sample.wer * 100).toFixed(2)}%</span>}
                    {sample.cer != null && <span className="text-[var(--text)]">CER: {(sample.cer * 100).toFixed(2)}%</span>}
                    {sample.language && <span className="text-[var(--text-dim)]">{t('evalAudioExtra.detailLanguage')}: {sample.language}</span>}
                    {sample.elapsed_seconds != null && <span className="text-[var(--text-dim)]">{t('evalAudioExtra.detailElapsed')}: {sample.elapsed_seconds}s</span>}
                  </div>
                  {sample.audio_url && (
                    <div className="mt-3">
                      <audio controls className="w-full h-8">
                        <source src={sample.audio_url} />
                        {t('evalAudioExtra.detailUnsupportedAudio')}
                      </audio>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="flex items-start gap-3">
                    <span className="text-xs text-[var(--text-dim)] font-mono mt-1">#{(sample.index ?? i) + 1}</span>
                    <div className="flex-1 min-w-0">
                      {sample.error ? (
                        <div className="text-sm text-[var(--danger)]">{t('evalAudioExtra.detailError')}: {sample.error}</div>
                      ) : (
                        <>
                          <div className="text-sm text-[var(--text)] mb-2">{sample.prompt || '-'}</div>
                          {sample.audio_url && (
                            <audio controls className="w-full h-8">
                              <source src={sample.audio_url} />
                              {t('evalAudioExtra.detailUnsupportedAudio')}
                            </audio>
                          )}
                          <div className="text-xs text-[var(--text-dim)] mt-1">
                            {t('evalAudioExtra.detailElapsed')}: {sample.elapsed_seconds}s
                            {sample.duration_seconds != null && ` · ${t('evalAudioExtra.detailDuration')}: ${sample.duration_seconds.toFixed(1)}s`}
                          </div>
                          {sample.hypothesis != null && (
                            <div className="mt-2 p-2 rounded bg-[var(--bg-deep)] text-xs">
                              <div className="text-[var(--text-muted)] mb-1">{t('evalAudioExtra.detailAsrRecognition')}:</div>
                              <div className="text-[var(--text)]">{sample.hypothesis}</div>
                              <div className="flex gap-3 mt-1">
                                {sample.wer != null && <span className="text-[var(--accent)]">WER: {(sample.wer * 100).toFixed(2)}%</span>}
                                {sample.cer != null && <span className="text-[var(--text)]">CER: {(sample.cer * 100).toFixed(2)}%</span>}
                                {sample.asr_elapsed_seconds != null && <span className="text-[var(--text-dim)]">{t('evalAudioExtra.detailAsrElapsed')}: {sample.asr_elapsed_seconds}s</span>}
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          ))}
        </>
      )}
    </div>
  )
}
