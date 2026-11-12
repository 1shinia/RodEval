import { useId, useEffect, useState, type SyntheticEvent } from 'react'
import { api, getAuthHeaders } from '@/api/client'
import { useLocale } from '@/contexts/LocaleContext'

import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS } from '@/components/ui/formStyles'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

type AudioTool = 'asr' | 'tts'
type TTSPrompt = { id: string; category: string; text: string }
type FleursSample = { id: string; text: string; audio_url: string }
type FleursStatus = { ready: boolean; status: string; sample_count: number }
type FleursList = FleursStatus & { samples: FleursSample[]; total: number }

const PROVIDER_OPTIONS: { value: string; label: string; apiPlaceholder: string; modelPlaceholder: string; defaultVoice: string; voices: { value: string; label: string }[] }[] = [
  {
    value: 'openai', label: 'providerOpenai', apiPlaceholder: 'https://api.openai.com/v1', modelPlaceholder: 'whisper-1 / tts-1',
    defaultVoice: 'alloy',
    voices: [
      { value: 'alloy', label: 'voiceAlloy' },
      { value: 'echo', label: 'voiceEcho' },
      { value: 'fable', label: 'voiceFable' },
      { value: 'onyx', label: 'voiceOnyx' },
      { value: 'nova', label: 'voiceNova' },
      { value: 'shimmer', label: 'voiceShimmer' },
    ],
  },
  {
    value: 'dashscope', label: 'providerDashscope', apiPlaceholder: 'https://dashscope.aliyuncs.com/api/v1', modelPlaceholder: 'qwen-audio-asr / cosyvoice-v3.5-flash / sambert-zhichu-v1',
    defaultVoice: 'longxiaochun',
    voices: [
      { value: 'longxiaochun', label: 'voiceLongxiaochun' },
      { value: 'longbella', label: 'voiceLongbella' },
      { value: 'longcheng', label: 'voiceLongcheng' },
      { value: 'longxiaoxia', label: 'voiceLongxiaoxia' },
      { value: 'longxiaoqi', label: 'voiceLongxiaoqi' },
      { value: 'longyue', label: 'voiceLongyue' },
      { value: 'longfeiyu', label: 'voiceLongfeiyu' },
      { value: 'zhichu', label: 'voiceZhichu' },
      { value: 'zhixiaobai', label: 'voiceZhixiaobai' },
      { value: 'zhiluo', label: 'voiceZhiluo' },
      { value: 'zhiyu', label: 'voiceZhiyu' },
    ],
  },
]

const METRICS_OPTIONS = [
  { value: 'wer', label: 'metricWer' },
  { value: 'cer', label: 'metricCer' },
]

export default function AudioEvalForm({ onSubmit, disabled }: Props) {
  const { t } = useLocale()
  const audioFileInputId = useId()
  const [tool, setTool] = useState<AudioTool>('asr')

  const handleToolChange = (newTool: AudioTool) => {
    setTool(newTool)
    if (newTool === 'asr') {
      setMetrics(['wer'])
    } else {
      setMetrics([])
    }
  }


  // API fields
  const [provider, setProvider] = useState('openai')
  const [model, setModel] = useState('')
  const [apiBase, setApiBase] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [language, setLanguage] = useState('zh')


  const currentProvider = PROVIDER_OPTIONS.find(p => p.value === provider)!

  // ASR-specific
  const [audioBase64, setAudioBase64] = useState('')
  const [audioFileName, setAudioFileName] = useState('')
  const [referenceText, setReferenceText] = useState('')
  const [audioSource, setAudioSource] = useState<'upload' | 'fleurs'>('upload')
  const [fleursStatus, setFleursStatus] = useState<FleursStatus | null>(null)
  const [fleursSamples, setFleursSamples] = useState<FleursSample[]>([])
  const [fleursQuery, setFleursQuery] = useState('')
  const [fleursOffset, setFleursOffset] = useState(0)
  const [fleursTotal, setFleursTotal] = useState(0)
  const [sampleId, setSampleId] = useState('')
  const [selectedSample, setSelectedSample] = useState<FleursSample | null>(null)
  const [fleursError, setFleursError] = useState('')
  const [previewUrl, setPreviewUrl] = useState('')
  const [previewError, setPreviewError] = useState('')

  // TTS-specific
  const [voice, setVoice] = useState(currentProvider.defaultVoice)
  const [audioFormat, setAudioFormat] = useState('mp3')
  const [speed, setSpeed] = useState('1.0')
  const [customPrompt, setCustomPrompt] = useState('')
  const [builtinPrompts, setBuiltinPrompts] = useState<TTSPrompt[]>([])
  const [promptCategory, setPromptCategory] = useState('')
  const [promptSearch, setPromptSearch] = useState('')
  const [promptLoadError, setPromptLoadError] = useState('')
  const [promptLimit, setPromptLimit] = useState('1')
  const [promptSeed, setPromptSeed] = useState('42')
  const [selectedPrompt, setSelectedPrompt] = useState<number | null>(null)
  // TTS→ASR closed-loop
  const [asrEnabled, setAsrEnabled] = useState(false)
  const [asrProvider, setAsrProvider] = useState('dashscope')
  const [asrModelName, setAsrModelName] = useState('paraformer-realtime-v2')
  const [asrApiBase, setAsrApiBase] = useState('https://ws-dwy81zmcwutmk6py.cn-beijing.maas.aliyuncs.com/api/v1')
  const [asrApiKey, setAsrApiKey] = useState('')

  // Metrics
  const [metrics, setMetrics] = useState<string[]>(['wer'])

  const [errors, setErrors] = useState<Record<string, string>>({})

  useEffect(() => {
    let active = true
    api<{ dataset: { items: TTSPrompt[] } }>('/api/v1/audio/tts-prompts')
      .then(data => { if (active) setBuiltinPrompts(data.dataset.items) })
      .catch(() => { if (active) setPromptLoadError(t('evalAudioExtra.promptLoadFailed')) })
    return () => { active = false }
  }, [t])

  useEffect(() => {
    if (audioSource !== 'fleurs' || tool !== 'asr') return
    let active = true
    api<FleursStatus>('/api/v1/audio/fleurs/status')
      .then(data => { if (active) { setFleursStatus(data); setFleursError('') } })
      .catch(() => { if (active) { setFleursStatus(null); setFleursError(t('evalAudioExtra.fleursLoadFailed')) } })
    return () => { active = false }
  }, [audioSource, tool, t])

  useEffect(() => {
    if (audioSource !== 'fleurs' || !fleursStatus?.ready) return
    let active = true
    api<FleursList>('/api/v1/audio/fleurs/samples', { q: fleursQuery, offset: fleursOffset, limit: 20 })
      .then(data => { if (active) { setFleursSamples(data.samples); setFleursTotal(data.total); setFleursError('') } })
      .catch(() => { if (active) setFleursError(t('evalAudioExtra.fleursLoadFailed')) })
    return () => { active = false }
  }, [audioSource, fleursStatus?.ready, fleursQuery, fleursOffset, t])

  useEffect(() => {
    if (!selectedSample || audioSource !== 'fleurs') return
    let active = true
    let url = ''
    fetch(selectedSample.audio_url, { headers: getAuthHeaders() })
      .then(res => { if (!res.ok) throw new Error('Audio unavailable'); return res.blob() })
      .then(blob => { if (active) { url = URL.createObjectURL(blob); setPreviewUrl(url) } })
      .catch(() => { if (active) setPreviewError(t('evalAudioExtra.fleursPreviewFailed')) })
    return () => { active = false; if (url) URL.revokeObjectURL(url) }
  }, [selectedSample, audioSource, t])

  const selectFleursSample = async (id: string) => {
    setPreviewUrl('')
    setPreviewError('')
    setSampleId(id)
    setSelectedSample(null)
    setFleursError('')
    if (!id) return
    try {
      const sample = await api<FleursSample>(`/api/v1/audio/fleurs/sample/${encodeURIComponent(id)}`)
      setSelectedSample(sample)
      setErrors(prev => ({ ...prev, audio: '' }))
    } catch {
      setFleursError(t('evalAudioExtra.fleursSampleMissing'))
    }
  }

  const chooseRandomFleurs = async () => {
    setPreviewUrl('')
    setPreviewError('')
    setSelectedSample(null)
    try {
      const sample = await api<FleursSample>('/api/v1/audio/fleurs/random', { nonce: Date.now() })
      setSampleId(sample.id)
      setSelectedSample(sample)
      setFleursError('')
      setErrors(prev => ({ ...prev, audio: '' }))
    } catch {
      setFleursError(t('evalAudioExtra.fleursSampleMissing'))
    }
  }

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) {
      setAudioBase64('')
      setAudioFileName('')
      return
    }
    if (file.size > 25 * 1024 * 1024) {
      setAudioBase64('')
      setAudioFileName('')
      setErrors(prev => ({ ...prev, audio: t('evalAudioExtra.audioTooLarge') }))
      e.target.value = ''
      return
    }
    setErrors(prev => ({ ...prev, audio: '' }))
    setAudioBase64('')
    setAudioFileName(file.name)
    const reader = new FileReader()
    reader.onload = () => {
      setAudioBase64(reader.result as string)
    }
    reader.onerror = () => setErrors(prev => ({ ...prev, audio: t('evalAudioExtra.audioReadFailed') }))
    reader.readAsDataURL(file)
  }

  const categoryPrompts = builtinPrompts.map((item, index) => ({ ...item, index })).filter(item => !promptCategory || item.category === promptCategory)
  const visiblePrompts = categoryPrompts.filter(item => `${item.id} ${item.text}`.toLowerCase().includes(promptSearch.toLowerCase()))

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}

    if (!model.trim()) newErrors.model = t('evalAudioExtra.modelRequired')
    if (!apiBase.trim()) newErrors.apiBase = t('evalAudioExtra.apiUrlRequired')
    if (!apiKey.trim()) newErrors.apiKey = t('evalAudioExtra.apiKeyRequired')
    if (provider === 'volcengine') {
      const expected = tool === 'asr' ? ['volc.bigasr.auc_turbo'] : ['seed-tts-2.0', 'seed-icl-2.0']
      if (model.trim() && !expected.includes(model.trim())) newErrors.model = t('evalAudioExtra.volcModelHint')
      if (tool === 'tts' && !voice.trim()) newErrors.voice = t('evalAudioExtra.volcVoiceHint')
    }
    if (tool === 'tts' && asrEnabled && asrProvider === 'volcengine' && asrModelName.trim() !== 'volc.bigasr.auc_turbo') {
      newErrors.asrModel = t('evalAudioExtra.volcModelHint')
    }

    // ASR requires audio file
    if (tool === 'tts' && asrEnabled && !asrModelName.trim()) newErrors.asrModel = t('evalAudioExtra.requireAsrModel')
    if (tool === 'tts' && asrEnabled && !metrics.length) newErrors.metrics = t('evalAudioExtra.requireMetric')
    if (tool === 'asr' && audioSource === 'upload' && !audioBase64) {
      newErrors.audio = t('evalAudioExtra.audioRequired')
    }
    if (tool === 'asr' && audioSource === 'fleurs' && (!fleursStatus?.ready || !selectedSample || selectedSample.id !== sampleId)) {
      newErrors.audio = t('evalAudioExtra.fleursUnavailable')
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }
    setErrors({})

    const modelConfig: Record<string, unknown> = {
      model_name_or_path: model.trim(),
      model_type: tool,
      provider,
      api_base: apiBase.trim(),
    }
    if (apiKey.trim()) modelConfig.api_key = apiKey.trim()

    const generateConfig: Record<string, unknown> = {}

    if (tool === 'asr') {
      if (audioSource === 'fleurs') {
        generateConfig.audio_source = 'fleurs'
        generateConfig.dataset_sample_id = selectedSample!.id
      } else {
        generateConfig.audio_base64 = audioBase64
        generateConfig.audio_filename = audioFileName
        generateConfig.reference_text = referenceText.trim()
      }
    } else {
      modelConfig.language = language
      generateConfig.voice = voice
      generateConfig.response_format = audioFormat
      generateConfig.speed = Number(speed) || 1.0
    }

    const evalConfig: Record<string, unknown> = {
      metrics: tool === 'asr' ? metrics : [],
      prompt_dataset: 'builtin',
      prompt_limit: Math.min(Number(promptLimit) || 1, categoryPrompts.length || 1),
    }

    if (tool === 'tts') {
      evalConfig.prompt_seed = parseInt(promptSeed) || 0
      if (!customPrompt.trim()) {
        if (promptCategory) evalConfig.prompt_category = promptCategory
        if (selectedPrompt !== null) evalConfig.selected_prompt_index = selectedPrompt
      }
    }
    if (tool === 'tts' && customPrompt.trim()) {
      evalConfig.custom_prompt = customPrompt.trim()
    }
    if (tool === 'tts' && asrEnabled && asrModelName.trim()) {
      evalConfig.asr_model_name = asrModelName.trim()
      evalConfig.asr_provider = asrProvider
      if (asrApiBase.trim()) evalConfig.asr_api_base = asrApiBase.trim()
      if (asrApiKey.trim()) evalConfig.asr_api_key = asrApiKey.trim()
      evalConfig.metrics = metrics
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }

    onSubmit({
      eval_backend: 'AudioEval',
      eval_config: {
        tool,
        model: modelConfig,
        generate: generateConfig,
        eval: evalConfig,
      },
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Tool Selector */}
      <div className="flex items-center gap-3 border-b border-[var(--border-md)] pb-4">
        <span className="text-sm font-medium text-[var(--text)]">{t('evalAudioExtra.toolType')}</span>
        <select value={tool} onChange={e => handleToolChange(e.target.value as AudioTool)} className={FORM_INPUT_CLASS + ' !w-auto'}>
          <option value="asr">{t('evalAudioExtra.asrOption')}</option>
          <option value="tts">{t('evalAudioExtra.ttsOption')}</option>
        </select>
      </div>


      {/* Model Configuration */}
      <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2">
        {t('aigc.modelConfig')}
      </h4>

      <>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <FormField label={t('evalAudioExtra.apiProvider')} className="md:col-span-2">
          <select value={provider} onChange={e => {
            const newProvider = e.target.value
            setProvider(newProvider)
            const p = PROVIDER_OPTIONS.find(o => o.value === newProvider)
            if (p) setVoice(p.defaultVoice)
            setModel('')
            setApiBase('')
            if (newProvider === 'volcengine') {
              setAudioFormat('mp3')
              setSpeed('1.0')
            }
          }} className={FORM_INPUT_CLASS}>
            {PROVIDER_OPTIONS.map(p => (
              <option key={p.value} value={p.value}>{t(`evalAudioExtra.${p.label}`)}</option>
            ))}
          </select>
        </FormField>

        <FormField label={t('evalAudioExtra.model')} required error={errors.model} className="md:col-span-2">
          <input value={model}
            onChange={e => { setModel(e.target.value.trimStart()); if (errors.model) setErrors(p => ({ ...p, model: '' })) }}
            className={FORM_INPUT_CLASS} placeholder={currentProvider.modelPlaceholder} />
        </FormField>

        <FormField label={t('evalAudioExtra.apiUrl')} required error={errors.apiBase} className="md:col-span-2">
          <input value={apiBase}
            onChange={e => { setApiBase(e.target.value); if (errors.apiBase) setErrors(p => ({ ...p, apiBase: '' })) }}
            className={FORM_INPUT_CLASS} placeholder={currentProvider.apiPlaceholder} />
        </FormField>

        <FormField label={t('evalAudioExtra.modelApiKey')} required error={errors.apiKey} className="md:col-span-2">
          <input type="password" value={apiKey}
            onChange={e => { setApiKey(e.target.value); if (errors.apiKey) setErrors(p => ({ ...p, apiKey: '' })) }}
            className={FORM_INPUT_CLASS} placeholder="sk-..." />
        </FormField>
      </div>

      {tool !== 'asr' && (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <FormField label={t('eval.ragLanguage')}>
          <select value={language} onChange={e => setLanguage(e.target.value)} className={FORM_INPUT_CLASS}>
            <option value="zh">{t('evalAudioExtra.languageChinese')}</option>
            <option value="en">{t('evalAudioExtra.languageEnglish')}</option>
            <option value="auto">{t('evalAudioExtra.languageAuto')}</option>
          </select>
        </FormField>
      </div>
      )}
      </>

      {/* ── ASR-specific ────────────────────────────── */}
      {tool === 'asr' && (
        <>
          <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2">
            {t('evalAudioExtra.asrTitle')}
          </h4>
          <div className="flex items-center gap-3 text-sm">
            <span className="font-medium text-[var(--text)]">{t('evalAudioExtra.audioSource')}</span>
            <select className={FORM_INPUT_CLASS + ' !w-auto'} value={audioSource} onChange={e => {
              setAudioSource(e.target.value as 'upload' | 'fleurs'); setSelectedSample(null); setSampleId(''); setErrors({})
            }}>
              <option value="upload">{t('evalAudioExtra.sourceUpload')}</option>
              <option value="fleurs">{t('evalAudioExtra.sourceFleurs')}</option>
            </select>
          </div>
          {audioSource === 'upload' ? <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FormField label={t('evalAudioExtra.audioFile')} required error={errors.audio} hint={provider === 'volcengine' ? t('evalAudioExtra.volcAsrFileHint') : undefined}>
              <div className="flex items-center gap-3">
                <input id={audioFileInputId} type="file" accept={provider === 'volcengine' ? '.wav,.mp3,.ogg' : 'audio/*'}
                  onChange={handleAudioChange} className="peer sr-only" />
                <label htmlFor={audioFileInputId}
                  className="inline-flex cursor-pointer rounded bg-[var(--accent)] px-3 py-1 text-sm text-white peer-focus-visible:ring-2 peer-focus-visible:ring-[var(--accent)]">
                  {t('evalAudioExtra.chooseFile')}
                </label>
                <span className="text-sm text-[var(--text-muted)]" aria-live="polite">
                  {audioFileName || t('evalAudioExtra.noFileChosen')}
                </span>
              </div>
            </FormField>

            <FormField label={t('evalAudioExtra.referenceText')} hint={t('evalAudioExtra.referenceTextHint')}>
              <textarea value={referenceText}
                onChange={e => setReferenceText(e.target.value)}
                className={FORM_INPUT_CLASS} rows={2}
                placeholder={t('evalAudioExtra.referenceTextPlaceholder')} />
            </FormField>
          </div>

          : <div className="space-y-3 rounded-[var(--radius)] border border-[var(--border)] p-3">
            {!fleursStatus?.ready && <p className="text-sm text-[var(--text-muted)]" role="status">
              {fleursError || (fleursStatus ? t('evalAudioExtra.fleursUnavailable') : t('evalAudioExtra.fleursChecking'))}
            </p>}
            {fleursStatus?.ready && <>
              <p className="text-sm text-[var(--text-muted)]">{t('evalAudioExtra.fleursCount', { count: fleursStatus.sample_count })}</p>
              <div className="flex flex-wrap items-center gap-2">
                <input aria-label={t('evalAudioExtra.fleursSearch')} className={FORM_INPUT_CLASS + ' !w-auto flex-1'} value={fleursQuery}
                  onChange={e => { setFleursQuery(e.target.value); setFleursOffset(0) }} placeholder={t('evalAudioExtra.fleursSearch')} />
                <button type="button" onClick={chooseRandomFleurs} className="text-sm text-[var(--accent)]">{t('evalAudioExtra.fleursRandom')}</button>
              </div>
              <select aria-label={t('evalAudioExtra.fleursSelect')} className={FORM_INPUT_CLASS} value={sampleId}
                onChange={e => void selectFleursSample(e.target.value)}>
                <option value="">{t('evalAudioExtra.fleursSelect')}</option>
                {selectedSample && !fleursSamples.some(s => s.id === selectedSample.id) && <option value={selectedSample.id}>{selectedSample.id} · {selectedSample.text}</option>}
                {fleursSamples.map(s => <option key={s.id} value={s.id}>{s.id} · {s.text}</option>)}
              </select>
              <div className="flex items-center gap-3 text-sm text-[var(--text-muted)]">
                <button type="button" disabled={fleursOffset === 0} onClick={() => setFleursOffset(v => Math.max(0, v - 20))}>{t('evalAudioExtra.fleursPrevious')}</button>
                <span>{fleursTotal ? `${fleursOffset + 1}–${Math.min(fleursOffset + 20, fleursTotal)} / ${fleursTotal}` : '0'}</span>
                <button type="button" disabled={fleursOffset + 20 >= fleursTotal} onClick={() => setFleursOffset(v => v + 20)}>{t('evalAudioExtra.fleursNext')}</button>
              </div>
              {selectedSample && <>
                <div className="text-sm text-[var(--text)]">{t('evalAudioExtra.referenceText')}: {selectedSample.text}</div>
                {previewUrl && <audio controls src={previewUrl} className="w-full" />}
                {previewError && <p role="alert" className="text-sm text-red-500">{previewError}</p>}
              </>}
            </>}
            {fleursError && fleursStatus?.ready && <p role="alert" className="text-sm text-red-500">{fleursError}</p>}
            {errors.audio && <p role="alert" className="text-sm text-red-500">{errors.audio}</p>}
          </div>}

          {/* ASR Metrics */}
          <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2">
            {t('aigc.metrics')}
          </h4>
          <div className="flex gap-6">
            {METRICS_OPTIONS.map(({ value, label }) => (
              <label key={value} className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={metrics.includes(value)}
                  onChange={() => setMetrics(prev =>
                    prev.includes(value) ? prev.filter(m => m !== value) : [...prev, value]
                  )}
                  className="accent-[var(--accent)]" />
                <span className="text-sm text-[var(--text)]">{t(`evalAudioExtra.${label}`)}</span>
              </label>
            ))}
          </div>
        </>
      )}

      {/* ── TTS-specific ────────────────────────────── */}
      {tool === 'tts' && (
        <>
          <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2">
            {t('evalAudioExtra.ttsTitle')}
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <FormField label={t('evalAudioExtra.voice')} required={provider === 'volcengine'} error={errors.voice} hint={provider === 'volcengine' ? t('evalAudioExtra.volcVoiceHint') : undefined}>
              <input
                type="text"
                value={voice}
                onChange={e => setVoice(e.target.value)}
                list="voice-suggestions"
                className={FORM_INPUT_CLASS}
                placeholder={currentProvider.defaultVoice}
              />
              <datalist id="voice-suggestions">
                {currentProvider.voices.map(v => (
                  <option key={v.value} value={v.value}>{t(`evalAudioExtra.${v.label}`)}</option>
                ))}
              </datalist>
            </FormField>

            <FormField label={t('evalAudioExtra.outputFormat')}>
              <select value={audioFormat} onChange={e => setAudioFormat(e.target.value)} className={FORM_INPUT_CLASS}>
                <option value="mp3">MP3</option>
                <option value="wav">WAV</option>
                {provider !== 'volcengine' && <option value="ogg">OGG</option>}
                {provider === 'volcengine' && <option value="ogg">OGG Opus</option>}
              </select>
            </FormField>

            <FormField label={t('evalAudioExtra.speed')}>
              <input type="range" min={provider === 'volcengine' ? '0.5' : '0.25'} max={provider === 'volcengine' ? '2.0' : '4.0'} step="0.25"
                value={speed} onChange={e => setSpeed(e.target.value)}
                className="w-full accent-[var(--accent)]" />
              <span className="text-xs text-[var(--text-muted)]">{speed}x</span>
            </FormField>
          </div>

          {/* TTS→ASR Closed-Loop Evaluation */}
          <div className="p-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-deep)]">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={asrEnabled} onChange={e => setAsrEnabled(e.target.checked)}
                className="w-4 h-4 rounded accent-[var(--accent)]" />
              <span className="text-sm font-medium text-[var(--text)]">{t('evalAudioExtra.asrClosedLoop')}</span>
              <span className="text-xs text-[var(--text-dim)]">{t('evalAudioExtra.asrClosedLoopHint')}</span>
            </label>
            {asrEnabled && (
              <div className="mt-2 grid grid-cols-2 gap-3">
                <FormField label={t('evalAudioExtra.apiProvider')}>
                  <select value={asrProvider} onChange={e => {
                    const selected = e.target.value
                    setAsrProvider(selected)
                    setAsrModelName(selected === 'volcengine' ? 'volc.bigasr.auc_turbo' : selected === 'dashscope' ? 'paraformer-realtime-v2' : '')
                    setAsrApiBase(selected === 'volcengine' ? 'https://openspeech.bytedance.com' : selected === 'dashscope' ? 'https://dashscope.aliyuncs.com/api/v1' : '')
                  }} className={FORM_INPUT_CLASS}>
                    {PROVIDER_OPTIONS.map(p => <option key={p.value} value={p.value}>{t(`evalAudioExtra.${p.label}`)}</option>)}
                  </select>
                </FormField>
                <FormField label={t('evalAudioExtra.asrModel')} error={errors.asrModel}>
                  <input type="text" value={asrModelName} onChange={e => setAsrModelName(e.target.value)}
                    className={FORM_INPUT_CLASS}
                    placeholder={asrProvider === 'volcengine' ? 'volc.bigasr.auc_turbo' : asrProvider === 'dashscope' ? 'paraformer-realtime-v2' : 'whisper-1'} />
                </FormField>
                <FormField label={t('evalAudioExtra.asrApiUrl')} hint={t('evalAudioExtra.asrApiUrlHint')}>
                  <input type="text" value={asrApiBase} onChange={e => setAsrApiBase(e.target.value)}
                    className={FORM_INPUT_CLASS}
                    placeholder="https://ws-dwy81zmcwutmk6py.cn-beijing.maas.aliyuncs.com/api/v1" />
                </FormField>
              </div>
            )}
            {asrEnabled && (
              <div className="mt-2">
                <FormField label={t('evalAudioExtra.asrApiKey')}>
                  <input type="password" value={asrApiKey} onChange={e => setAsrApiKey(e.target.value)}
                    className={FORM_INPUT_CLASS}
                    placeholder="sk-..." />
                </FormField>
              </div>
            )}
            {asrEnabled && (
              <div className="mt-2">
                <div className="mb-1.5 text-sm font-medium text-[var(--text)]">{t('evalAudioExtra.metricsLabel')}</div>
                {errors.metrics && <p className="text-sm text-red-500">{errors.metrics}</p>}
                <div className="flex gap-6">
                  {METRICS_OPTIONS.map(({ value, label }) => (
                    <label key={value} className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" checked={metrics.includes(value)}
                        onChange={() => setMetrics(prev =>
                          prev.includes(value) ? prev.filter(m => m !== value) : [...prev, value]
                        )}
                        className="accent-[var(--accent)]" />
                      <span className="text-sm text-[var(--text)]">{t(`evalAudioExtra.${label}`)}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* TTS Prompts */}
          <FormField label={t('evalAudioExtra.customPrompt')} hint={t('evalAudioExtra.customPromptHint')}>
            <textarea value={customPrompt}
              onChange={e => setCustomPrompt(e.target.value)}
              className={FORM_INPUT_CLASS} rows={2}
              placeholder={t('evalAudioExtra.customPromptPlaceholder')} />
          </FormField>
          <div className="text-sm text-[var(--text)]">
            {t('evalAudioExtra.promptDataset')}: {t('evalAudioExtra.builtinPrompts')}
          </div>
          <FormField label={t('evalAudioExtra.promptSeed')}>
            <input type="number" value={promptSeed} disabled={!!customPrompt.trim() || selectedPrompt !== null} onChange={e => setPromptSeed(e.target.value)} className={FORM_INPUT_CLASS} />
          </FormField>
          <FormField label={t('evalAudioExtra.promptCount')} hint={t('evalAudioExtra.promptCountHint')}>
            <input disabled={!!customPrompt.trim() || selectedPrompt !== null} type="number" min="1" max={categoryPrompts.length || 100}
              value={promptLimit} onChange={e => setPromptLimit(e.target.value.replace(/[^0-9]/g, ''))}
              className={FORM_INPUT_CLASS} />
          </FormField>
          <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-deep)] p-3 space-y-2">
            <div className="text-sm font-medium text-[var(--text)]">{t('evalAudioExtra.builtinPreview')}</div>
            {promptLoadError ? <div className="text-sm text-red-500">{promptLoadError}</div> : (
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2">
                  {['', ...new Set(builtinPrompts.map(item => item.category))].map(category => (
                    <button key={category} type="button" aria-pressed={promptCategory === category} disabled={!!customPrompt.trim()} onClick={() => { setPromptCategory(category); setSelectedPrompt(null); setPromptLimit('1') }} className={`px-3 py-1 rounded-full border text-sm ${promptCategory === category ? 'border-[var(--accent)] text-[var(--accent)]' : 'border-[var(--border)] text-[var(--text-muted)]'}`}>
                      {category || t('evalAudioExtra.allScenarios')} · {category ? builtinPrompts.filter(item => item.category === category).length : builtinPrompts.length}
                    </button>
                  ))}
                </div>
                <input className={FORM_INPUT_CLASS} value={promptSearch} onChange={e => setPromptSearch(e.target.value)} placeholder={t('evalAudioExtra.searchPrompts')} aria-label={t('evalAudioExtra.searchPrompts')} />
                <div className="max-h-64 overflow-y-auto space-y-3 text-sm text-[var(--text-muted)]">
                  {visiblePrompts.map(({ id, category, text, index }) => (
                    <label key={id} className="flex items-start gap-2 cursor-pointer">
                      <input type="checkbox" className="mt-1" disabled={!!customPrompt.trim()} checked={selectedPrompt === index} onChange={() => setSelectedPrompt(selectedPrompt === index ? null : index)} />
                      <span><span className="font-medium text-[var(--text)]">{id} · {category}</span><br />{text}</span>
                    </label>
                  ))}
                  {!visiblePrompts.length && <div>{t('evalAudioExtra.noPrompts')}</div>}
                </div>
                {selectedPrompt !== null && <div className="text-sm text-[var(--text)]">{t('evalAudioExtra.selectedText')}: {builtinPrompts[selectedPrompt]?.id} · {builtinPrompts[selectedPrompt]?.text}</div>}
              </div>
            )}
          </div>
        </>
      )}

      {/* Submit Button */}
      <div className="flex justify-end pt-2 border-t border-[var(--border-md)]">
        <Button type="submit" variant="primary" disabled={disabled || (tool === 'asr' && audioSource === 'fleurs' && (!fleursStatus?.ready || !selectedSample || selectedSample.id !== sampleId))}>
          {disabled ? t('evalAudioExtra.running') : t('eval.startEval')}
        </Button>
      </div>
    </form>
  )
}
