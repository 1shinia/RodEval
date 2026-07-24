import { useCallback, useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Mic, Volume2 } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import Skeleton from '@/components/ui/Skeleton'

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
  metrics?: Record<string, number>
  per_sample?: ASRSample | ASRSample[] | TTSSample[]
  elapsed_seconds?: number
  num_samples?: number
  total_elapsed_seconds?: number
}

export default function AudioReportDetailPage() {
  const { taskId } = useParams<{ taskId: string }>()
  const navigate = useNavigate()
  const { t } = useLocale()

  const [report, setReport] = useState<AudioReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchReport = useCallback(async () => {
    if (!taskId) return
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/v1/audio/report/${encodeURIComponent(taskId)}`)
      if (!response.ok) {
        throw new Error(`Failed to load report: ${response.statusText}`)
      }
      const data = await response.json()
      setReport(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load report')
    } finally {
      setLoading(false)
    }
  }, [taskId])

  useEffect(() => {
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
          <ArrowLeft size={16} /> 返回
        </button>
        <div className="px-4 py-3 rounded-[var(--radius)] bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger)]">
          {error || '报告未找到'}
        </div>
      </div>
    )
  }

  const isASR = report.tool === 'asr'
  const samples: any[] = report.per_sample
    ? (Array.isArray(report.per_sample) ? report.per_sample : [report.per_sample])
    : []

  return (
    <div className="page-enter flex flex-col gap-5 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <button onClick={() => navigate(-1)} className="flex items-center gap-2 text-sm text-[var(--text-muted)] hover:text-[var(--text)]">
          <ArrowLeft size={16} /> 返回
        </button>
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
          isASR ? 'bg-blue-500/10 text-blue-400' : 'bg-green-500/10 text-green-400'
        }`}>
          {isASR ? 'ASR' : 'TTS'}
        </span>
      </div>

      <h1 className="text-2xl font-bold text-[var(--text)]">
        {isASR ? '语音识别评估报告' : '语音合成评估报告'}
      </h1>

      {/* Model Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
        <div>
          <div className="text-xs text-[var(--text-muted)]">模型</div>
          <div className="text-sm font-medium text-[var(--text)]">{report.model}</div>
        </div>
        <div>
          <div className="text-xs text-[var(--text-muted)]">功能</div>
          <div className="text-sm font-medium text-[var(--text)]">{isASR ? '语音识别 (ASR)' : '语音合成 (TTS)'}</div>
        </div>
        <div>
          <div className="text-xs text-[var(--text-muted)]">耗时</div>
          <div className="text-sm font-medium text-[var(--text)]">
            {isASR
              ? `${report.elapsed_seconds?.toFixed(1) || '-'} 秒`
              : `${report.total_elapsed_seconds?.toFixed(1) || '-'} 秒`}
          </div>
        </div>
      </div>

      {/* ASR Metrics */}
      {isASR && report.metrics && (
        <div className="grid grid-cols-2 gap-4">
          {report.metrics.wer != null && (
            <div className="p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
              <div className="text-xs text-[var(--text-muted)] mb-1">WER (词错误率)</div>
              <div className={`text-2xl font-bold font-mono ${report.metrics.wer < 0.2 ? 'text-green-400' : report.metrics.wer < 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>
                {(report.metrics.wer * 100).toFixed(2)}%
              </div>
              <div className="text-xs text-[var(--text-dim)] mt-1">越低越好</div>
            </div>
          )}
          {report.metrics.cer != null && (
            <div className="p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
              <div className="text-xs text-[var(--text-muted)] mb-1">CER (字错误率)</div>
              <div className={`text-2xl font-bold font-mono ${report.metrics.cer < 0.2 ? 'text-green-400' : report.metrics.cer < 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>
                {(report.metrics.cer * 100).toFixed(2)}%
              </div>
              <div className="text-xs text-[var(--text-dim)] mt-1">越低越好</div>
            </div>
          )}
        </div>
      )}

      {/* TTS Summary */}
      {!isASR && (
        <div className="p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <div className="text-xs text-[var(--text-muted)] mb-2">生成汇总</div>
          <div className="flex gap-6">
            <div>
              <div className="text-2xl font-bold text-[var(--accent)]">{report.num_samples || samples.length}</div>
              <div className="text-xs text-[var(--text-dim)]">样本数</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-[var(--text)]">{report.total_elapsed_seconds?.toFixed(1) || '-'}s</div>
              <div className="text-xs text-[var(--text-dim)]">总耗时</div>
            </div>
          </div>
        </div>
      )}

      {/* Sample Details */}
      {samples.length > 0 && (
        <>
          <h2 className="text-lg font-semibold text-[var(--text)] border-b border-[var(--border)] pb-2">
            {isASR ? '识别结果' : '音频样本'}
          </h2>
          {samples.map((sample, i) => (
            <div key={i} className="p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
              {isASR ? (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                    <div>
                      <div className="text-xs text-[var(--text-muted)] mb-1">参考文本</div>
                      <div className="text-sm text-[var(--text)] bg-[var(--bg-card2)] p-2 rounded">{sample.reference || '(无参考)'}</div>
                    </div>
                    <div>
                      <div className="text-xs text-[var(--text-muted)] mb-1">识别结果</div>
                      <div className="text-sm text-[var(--text)] bg-[var(--bg-card2)] p-2 rounded">{sample.hypothesis || '-'}</div>
                    </div>
                  </div>
                  <div className="flex gap-4 text-xs">
                    {sample.wer != null && <span className="text-[var(--accent)]">WER: {(sample.wer * 100).toFixed(2)}%</span>}
                    {sample.cer != null && <span className="text-[var(--text)]">CER: {(sample.cer * 100).toFixed(2)}%</span>}
                    {sample.language && <span className="text-[var(--text-dim)]">语言: {sample.language}</span>}
                    {sample.elapsed_seconds != null && <span className="text-[var(--text-dim)]">耗时: {sample.elapsed_seconds}s</span>}
                  </div>
                  {sample.audio_url && (
                    <div className="mt-3">
                      <audio controls className="w-full h-8">
                        <source src={sample.audio_url} />
                        浏览器不支持音频播放
                      </audio>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="flex items-start gap-3">
                    <span className="text-xs text-[var(--text-dim)] font-mono mt-1">#{sample.index + 1}</span>
                    <div className="flex-1 min-w-0">
                      {sample.error ? (
                        <div className="text-sm text-[var(--danger)]">错误: {sample.error}</div>
                      ) : (
                        <>
                          <div className="text-sm text-[var(--text)] mb-2">{sample.prompt || '-'}</div>
                          {sample.audio_url && (
                            <audio controls className="w-full h-8">
                              <source src={sample.audio_url} />
                              浏览器不支持音频播放
                            </audio>
                          )}
                          <div className="text-xs text-[var(--text-dim)] mt-1">耗时: {sample.elapsed_seconds}s</div>
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
