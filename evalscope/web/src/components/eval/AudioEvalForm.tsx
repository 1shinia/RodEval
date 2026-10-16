import { useId, useEffect, useState, type SyntheticEvent } from 'react'
import { api } from '@/api/client'
import { useLocale } from '@/contexts/LocaleContext'

import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS } from '@/components/ui/formStyles'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

type AudioTool = 'asr' | 'tts'

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

  // TTS-specific
  const [voice, setVoice] = useState(currentProvider.defaultVoice)
  const [audioFormat, setAudioFormat] = useState('mp3')
  const [speed, setSpeed] = useState('1.0')
  const [customPrompt, setCustomPrompt] = useState('')
  const [builtinPrompts, setBuiltinPrompts] = useState<string[]>([])
  const [promptLoadError, setPromptLoadError] = useState('')
  const [promptLimit, setPromptLimit] = useState('1')
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
    api<{ prompts: string[] }>('/api/v1/audio/tts-prompts')
      .then(data => { if (active) setBuiltinPrompts(data.prompts || []) })
      .catch(() => { if (active) setPromptLoadError(t('evalAudioExtra.promptLoadFailed')) })
    return () => { active = false }
  }, [t])

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
    if (tool === 'asr' && !audioBase64) {
      newErrors.audio = t('evalAudioExtra.audioRequired')
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
      generateConfig.audio_base64 = audioBase64
      generateConfig.audio_filename = audioFileName
      generateConfig.reference_text = referenceText.trim()
    } else {
      generateConfig.voice = voice
      generateConfig.response_format = audioFormat
      generateConfig.speed = Number(speed) || 1.0
    }

    const evalConfig: Record<string, unknown> = {
      metrics: tool === 'asr' ? metrics : [],
      prompt_dataset: 'builtin',
      prompt_limit: Number(promptLimit) || 1,
    }

    if (tool === 'tts' && customPrompt.trim()) {
      evalConfig.custom_prompt = customPrompt.trim()
    }
    if (tool === 'tts' && asrEnabled && asrModelName.trim()) {
      evalConfig.asr_model_name = asrModelName.trim()
      evalConfig.asr_provider = asrProvider
      if (asrApiBase.trim()) evalConfig.asr_api_base = asrApiBase.trim()
      if (asrApiKey.trim()) evalConfig.asr_api_key = asrApiKey.trim()
      evalConfig.metrics = metrics.length > 0 ? metrics : ['wer', 'cer']
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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                <FormField label={t('evalAudioExtra.asrApiKey')} hint={t('evalAudioExtra.asrApiKeyHint')}>
                  <input type="password" value={asrApiKey} onChange={e => setAsrApiKey(e.target.value)}
                    className={FORM_INPUT_CLASS}
                    placeholder="sk-..." />
                </FormField>
              </div>
            )}
            {asrEnabled && (
              <div className="mt-2 flex gap-6">
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
          <FormField label={t('evalAudioExtra.promptCount')} hint={t('evalAudioExtra.promptCountHint')}>
            <input type="number" min="1" max={10}
              value={promptLimit} onChange={e => setPromptLimit(e.target.value.replace(/[^0-9]/g, ''))}
              className={FORM_INPUT_CLASS} />
          </FormField>
          <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-deep)] p-3 space-y-2">
            <div className="text-sm font-medium text-[var(--text)]">{t('evalAudioExtra.builtinPreview')}</div>
            {promptLoadError ? <div className="text-sm text-red-500">{promptLoadError}</div> : (
              <ol className="list-decimal pl-5 space-y-1 text-sm text-[var(--text-muted)]">
                {builtinPrompts.map((prompt, index) => <li key={`${index}-${prompt}`}>{prompt}</li>)}
              </ol>
            )}
          </div>
        </>
      )}

      {/* Submit Button */}
      <div className="flex justify-end pt-2 border-t border-[var(--border-md)]">
        <Button type="submit" variant="primary" disabled={disabled}>
          {disabled ? t('evalAudioExtra.running') : t('eval.startEval')}
        </Button>
      </div>
    </form>
  )
}
