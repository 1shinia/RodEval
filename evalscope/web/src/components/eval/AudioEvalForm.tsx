import { useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, FORM_LABEL_CLASS } from '@/components/ui/formStyles'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

type AudioTool = 'asr' | 'tts'

const PROVIDER_OPTIONS: { value: string; label: string; apiPlaceholder: string; modelPlaceholder: string; defaultVoice: string; voices: { value: string; label: string }[] }[] = [
  {
    value: 'openai', label: 'OpenAI 兼容', apiPlaceholder: 'https://api.openai.com/v1', modelPlaceholder: 'whisper-1 / tts-1',
    defaultVoice: 'alloy',
    voices: [
      { value: 'alloy', label: 'Alloy (中性)' },
      { value: 'echo', label: 'Echo (男声)' },
      { value: 'fable', label: 'Fable (英式)' },
      { value: 'onyx', label: 'Onyx (深沉)' },
      { value: 'nova', label: 'Nova (女声)' },
      { value: 'shimmer', label: 'Shimmer (轻快)' },
    ],
  },
  {
    value: 'dashscope', label: 'DashScope (百炼)', apiPlaceholder: 'https://dashscope.aliyuncs.com/api/v1', modelPlaceholder: 'qwen-audio-asr / cosyvoice-v3.5-flash',
    defaultVoice: 'longxiaochun',
    voices: [
      { value: 'longxiaochun', label: 'longxiaochun (龙小春·女声)' },
      { value: 'longbella', label: 'longbella (龙贝拉·女声)' },
      { value: 'longcheng', label: 'longcheng (龙成·男声)' },
      { value: 'longxiaoxia', label: 'longxiaoxia (龙小夏·女声)' },
      { value: 'longxiaoqi', label: 'longxiaoqi (龙小七·男声)' },
      { value: 'longyue', label: 'longyue (龙悦·女声)' },
      { value: 'longfeiyu', label: 'longfeiyu (龙飞宇·男声)' },
    ],
  },
  {
    value: 'volcengine', label: '火山方舟', apiPlaceholder: 'https://ark.cn-beijing.volces.com/api/v3', modelPlaceholder: 'doubao-xxx',
    defaultVoice: 'zh_female_qingxin',
    voices: [
      { value: 'zh_female_qingxin', label: 'zh_female_qingxin (清新·女声)' },
      { value: 'zh_male_chunhou', label: 'zh_male_chunhou (醇厚·男声)' },
    ],
  },
]

const METRICS_OPTIONS = [
  { value: 'wer', label: 'WER (词错误率)' },
  { value: 'cer', label: 'CER (字错误率)' },
]

export default function AudioEvalForm({ onSubmit, disabled }: Props) {
  const { t } = useLocale()

  const [tool, setTool] = useState<AudioTool>('asr')

  const handleToolChange = (newTool: AudioTool) => {
    setTool(newTool)
    if (newTool === 'asr') {
      setMetrics(['wer'])
    } else {
      setMetrics([])
    }
  }

  // Model source: API or Local
  const [modelSource, setModelSource] = useState<'api' | 'local'>('api')
  const isLocal = modelSource === 'local'

  // API fields
  const [provider, setProvider] = useState('openai')
  const [model, setModel] = useState('')
  const [apiBase, setApiBase] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [language, setLanguage] = useState('zh')

  // Local model fields
  const [modelPath, setModelPath] = useState('')

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
  const [promptDataset, setPromptDataset] = useState('builtin')
  const [promptLimit, setPromptLimit] = useState('1')

  // Metrics
  const [metrics, setMetrics] = useState<string[]>(['wer'])

  const [errors, setErrors] = useState<Record<string, string>>({})

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) {
      setAudioBase64('')
      setAudioFileName('')
      return
    }
    setAudioFileName(file.name)
    const reader = new FileReader()
    reader.onload = () => {
      setAudioBase64(reader.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}

    if (isLocal) {
      if (!modelPath.trim()) newErrors.modelPath = '请输入模型路径'
    } else {
      if (!model.trim()) newErrors.model = '请输入模型名称'
      if (!apiBase.trim()) newErrors.apiBase = '请输入 API URL'
      if (!apiKey.trim()) newErrors.apiKey = '请输入 API Key'
    }

    // ASR requires audio file
    if (tool === 'asr' && !audioBase64) {
      newErrors.audio = '请上传音频文件'
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }
    setErrors({})

    const modelConfig: Record<string, unknown> = {
      model_name_or_path: isLocal ? modelPath.trim() : model.trim(),
      model_type: tool,
      provider: isLocal ? 'local' : provider,
    }
    if (!isLocal) {
      modelConfig.api_base = apiBase.trim()
      if (apiKey.trim()) modelConfig.api_key = apiKey.trim()
    }

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
      prompt_dataset: promptDataset,
      prompt_limit: Number(promptLimit) || 1,
    }

    if (tool === 'tts' && customPrompt.trim()) {
      evalConfig.custom_prompt = customPrompt.trim()
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
        <span className="text-sm font-medium text-[var(--text)]">功能类型</span>
        <select value={tool} onChange={e => handleToolChange(e.target.value as AudioTool)} className={FORM_INPUT_CLASS + ' !w-auto'}>
          <option value="asr">ASR (语音识别)</option>
          <option value="tts">TTS (语音合成)</option>
        </select>
      </div>

      {/* Model Source */}
      <div className="flex items-center gap-6">
        <label className={`${FORM_LABEL_CLASS} !mb-0`}>模型来源</label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="audio_ms" value="api" checked={!isLocal}
            onChange={() => setModelSource('api')} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">API</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="audio_ms" value="local" checked={isLocal}
            onChange={() => setModelSource('local')} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">本地模型</span>
        </label>
      </div>

      {/* Model Configuration */}
      <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2">
        模型配置
      </h4>

      {/* API mode fields */}
      {!isLocal && (<>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <FormField label="API 提供商">
          <select value={provider} onChange={e => {
            const newProvider = e.target.value
            setProvider(newProvider)
            const p = PROVIDER_OPTIONS.find(o => o.value === newProvider)
            if (p) setVoice(p.defaultVoice)
          }} className={FORM_INPUT_CLASS}>
            {PROVIDER_OPTIONS.map(p => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </FormField>

        <FormField label="模型名称" required error={errors.model}>
          <input value={model}
            onChange={e => { setModel(e.target.value.trimStart()); if (errors.model) setErrors(p => ({ ...p, model: '' })) }}
            className={FORM_INPUT_CLASS} placeholder={currentProvider.modelPlaceholder} />
        </FormField>

        <FormField label="API URL" required error={errors.apiBase} className="md:col-span-2">
          <input value={apiBase}
            onChange={e => { setApiBase(e.target.value); if (errors.apiBase) setErrors(p => ({ ...p, apiBase: '' })) }}
            className={FORM_INPUT_CLASS} placeholder={currentProvider.apiPlaceholder} />
        </FormField>

        <FormField label="API Key" required error={errors.apiKey}>
          <input type="password" value={apiKey}
            onChange={e => { setApiKey(e.target.value); if (errors.apiKey) setErrors(p => ({ ...p, apiKey: '' })) }}
            className={FORM_INPUT_CLASS} placeholder="sk-..." />
        </FormField>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <FormField label="语言">
          <select value={language} onChange={e => setLanguage(e.target.value)} className={FORM_INPUT_CLASS}>
            <option value="zh">中文</option>
            <option value="en">英文</option>
            <option value="auto">自动检测</option>
          </select>
        </FormField>
      </div>
      </>)}
      {/* Local model fields */}
      {isLocal && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FormField label="模型路径" required error={errors.modelPath}>
            <input value={modelPath}
              onChange={e => { setModelPath(e.target.value); if (errors.modelPath) setErrors(p => ({ ...p, modelPath: '' })) }}
              className={FORM_INPUT_CLASS} placeholder="openai/whisper-large-v3 或 /data/models/whisper" />
          </FormField>
        </div>
      )}

      {/* ── ASR-specific ────────────────────────────── */}
      {tool === 'asr' && (
        <>
          <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2">
            ASR 语音识别
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FormField label="音频文件" required error={errors.audio}>
              <div className="flex items-center gap-3">
                <input type="file" accept="audio/*"
                  onChange={handleAudioChange}
                  className={`${FORM_INPUT_CLASS} file:mr-3 file:px-3 file:py-1 file:rounded file:border-0 file:bg-[var(--accent)] file:text-white file:text-sm file:cursor-pointer`} />
              </div>
              {audioFileName && (
                <p className="text-xs text-[var(--text-muted)] mt-1">已选择: {audioFileName}</p>
              )}
            </FormField>

            <FormField label="参考文本（用于 WER/CER 对比）" hint="留空则只做识别，不计算指标">
              <textarea value={referenceText}
                onChange={e => setReferenceText(e.target.value)}
                className={FORM_INPUT_CLASS} rows={2}
                placeholder="请输入标准参考文本..." />
            </FormField>
          </div>

          {/* ASR Metrics */}
          <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2">
            评估指标
          </h4>
          <div className="flex gap-6">
            {METRICS_OPTIONS.map(({ value, label }) => (
              <label key={value} className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={metrics.includes(value)}
                  onChange={() => setMetrics(prev =>
                    prev.includes(value) ? prev.filter(m => m !== value) : [...prev, value]
                  )}
                  className="accent-[var(--accent)]" />
                <span className="text-sm text-[var(--text)]">{label}</span>
              </label>
            ))}
          </div>
        </>
      )}

      {/* ── TTS-specific ────────────────────────────── */}
      {tool === 'tts' && (
        <>
          <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2">
            TTS 语音合成
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <FormField label="语音音色">
              <select value={voice} onChange={e => setVoice(e.target.value)} className={FORM_INPUT_CLASS}>
                {currentProvider.voices.map(v => (
                  <option key={v.value} value={v.value}>{v.label}</option>
                ))}
              </select>
            </FormField>

            <FormField label="输出格式">
              <select value={audioFormat} onChange={e => setAudioFormat(e.target.value)} className={FORM_INPUT_CLASS}>
                <option value="mp3">MP3</option>
                <option value="wav">WAV</option>
                <option value="ogg">OGG</option>
              </select>
            </FormField>

            <FormField label="语速">
              <input type="range" min="0.25" max="4.0" step="0.25"
                value={speed} onChange={e => setSpeed(e.target.value)}
                className="w-full accent-[var(--accent)]" />
              <span className="text-xs text-[var(--text-muted)]">{speed}x</span>
            </FormField>
          </div>

          {/* TTS Prompts */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FormField label="提示词数据集">
              <select value={promptDataset} onChange={e => setPromptDataset(e.target.value)} className={FORM_INPUT_CLASS}>
                <option value="builtin">内置测试文本 (10 条)</option>
                <option value="custom">自定义数据集</option>
              </select>
            </FormField>

            <FormField label="提示词数量">
              <input type="number" min="1" max="1000"
                value={promptLimit} onChange={e => { const v = Math.max(1, Number(e.target.value)); setPromptLimit(String(v)) }}
                className={FORM_INPUT_CLASS} />
            </FormField>
          </div>

          <FormField label="自定义提示词（可选）" hint="留空则使用数据集">
            <textarea value={customPrompt}
              onChange={e => setCustomPrompt(e.target.value)}
              className={FORM_INPUT_CLASS} rows={2}
              placeholder="请输入要合成的文本..." />
          </FormField>
        </>
      )}

      {/* Submit Button */}
      <div className="flex justify-end pt-2 border-t border-[var(--border-md)]">
        <Button type="submit" variant="primary" disabled={disabled}>
          {disabled ? '运行中...' : '开始评估'}
        </Button>
      </div>
    </form>
  )
}
