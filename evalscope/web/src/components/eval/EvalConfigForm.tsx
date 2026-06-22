import { useEffect, useRef, useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listBenchmarks } from '@/api/eval'
import { toast } from '@/components/common/Toast'
import Button from '@/components/ui/Button'
import Card from '@/components/ui/Card'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, FORM_LABEL_CLASS, inputClass } from '@/components/ui/formStyles'
import { ChevronDown, ChevronUp } from 'lucide-react'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
  initialDataset?: string
  onApiKeyChange?: (apiKey: string) => void
}

interface ParamDef {
  key: string
  label: string
  type: 'number' | 'select' | 'checkbox'
  options?: string[]
  step?: string
  min?: number
  max?: number
  placeholder?: string
  showWhen?: (b: string) => boolean
}

const BACKEND_PARAMS: Record<string, ParamDef[]> = {
  llama_cpp: [
    { key: 'n_ctx', label: 'eval.params.contextLength', type: 'number', min: 1, placeholder: 'eval.params.defaultVal' },
    { key: 'n_threads', label: 'eval.params.threadCount', type: 'number', min: 1, placeholder: 'eval.params.defaultVal' },
  ],
  transformers: [
    { key: 'precision', label: 'eval.params.precision', type: 'select', options: ['float16', 'bfloat16', 'float32', 'auto'] },
    { key: 'device_map', label: 'eval.params.deviceMap', type: 'select', options: ['auto', 'cuda:0', 'cuda:1', 'cpu'] },
    { key: 'attn_implementation', label: 'eval.params.attnImpl', type: 'select', options: ['sdpa', 'eager', 'flash_attention_2'] },
    { key: 'trust_remote_code', label: 'eval.params.trustRemoteCode', type: 'checkbox' },
  ],
  vllm: [
    { key: 'max_model_len', label: 'eval.params.contextLength', type: 'number', min: 1, placeholder: 'eval.params.defaultModelLen' },
    { key: 'dtype', label: 'eval.params.precision', type: 'select', options: ['auto', 'float16', 'bfloat16', 'float32'] },
    { key: 'quantization', label: 'eval.params.quantization', type: 'select', options: ['eval.params.none', 'fp8', 'awq', 'gptq', 'marlin', 'gguf', 'bitsandbytes'] },
    { key: 'kv_cache_dtype', label: 'eval.params.kvCacheDtype', type: 'select', options: ['auto', 'fp8'] },
    { key: 'tensor_parallel_size', label: 'eval.params.tensorParallel', type: 'number', min: 1, placeholder: 'eval.params.defaultAutoGpu' },
    { key: 'pipeline_parallel_size', label: 'eval.params.pipelineParallel', type: 'number', min: 1, placeholder: 'eval.params.defaultVal' },
    { key: 'data_parallel_size', label: 'eval.params.dataParallel', type: 'number', min: 1, placeholder: 'eval.params.defaultVal' },
    { key: 'expert_parallel_size', label: 'eval.params.expertParallel', type: 'number', min: 1, placeholder: 'eval.params.defaultVal' },
    { key: 'gpu_memory_utilization', label: 'eval.params.gpuMemUtil', type: 'number', min: 0, max: 1, step: '0.05', placeholder: 'eval.params.defaultVal' },
    { key: 'max_num_seqs', label: 'eval.params.maxConcurrent', type: 'number', min: 1, placeholder: 'eval.params.defaultVal' },
    { key: 'trust_remote_code', label: 'eval.params.trustRemoteCode', type: 'checkbox' },
  ],
  sglang: [
    { key: 'context_length', label: 'eval.params.contextLength', type: 'number', min: 1, placeholder: 'eval.params.defaultModelLen' },
    { key: 'dtype', label: 'eval.params.precision', type: 'select', options: ['auto', 'float16', 'bfloat16', 'float32'] },
    { key: 'quantization', label: 'eval.params.quantization', type: 'select', options: ['eval.params.none', 'fp8', 'awq', 'gptq', 'marlin', 'gguf', 'bitsandbytes'] },
    { key: 'kv_cache_dtype', label: 'eval.params.kvCacheDtype', type: 'select', options: ['auto', 'fp8'] },
    { key: 'tp_size', label: 'eval.params.tensorParallel', type: 'number', min: 1, placeholder: 'eval.params.defaultAutoGpu' },
    { key: 'pp_size', label: 'eval.params.pipelineParallel', type: 'number', min: 1, placeholder: 'eval.params.defaultVal' },
    { key: 'dp_size', label: 'eval.params.dataParallel', type: 'number', min: 1, placeholder: 'eval.params.defaultVal' },
    { key: 'ep_size', label: 'eval.params.expertParallel', type: 'number', min: 1, placeholder: 'eval.params.defaultVal' },
    { key: 'mem_fraction_static', label: 'eval.params.gpuMemUtil', type: 'number', min: 0, max: 1, step: '0.05', placeholder: 'eval.params.defaultVal' },
    { key: 'max_running_requests', label: 'eval.params.maxConcurrent', type: 'number', min: 1, placeholder: 'eval.params.defaultVal' },
    { key: 'trust_remote_code', label: 'eval.params.trustRemoteCode', type: 'checkbox' },
  ],
}

const DEFAULT_PARAM_VALUES: Record<string, string> = {
  n_ctx: '4096',
  n_threads: '8',
  precision: 'auto',
  device_map: 'auto',
  attn_implementation: 'sdpa',
  dtype: 'auto',
}

function getParams(backend: string): ParamDef[] {
  if (backend === 'auto') return []
  return BACKEND_PARAMS[backend] || []
}

export default function EvalConfigForm({ onSubmit, disabled, initialDataset, onApiKeyChange }: Props) {
  const { t } = useLocale()

  // Model source
  const [modelSource, setModelSource] = useState<'openai' | 'local'>('openai')
  const isLocal = modelSource === 'local'

  // OpenAI API
  const [model, setModel] = useState('')
  const [apiUrl, setApiUrl] = useState('')
  const [apiKey, setApiKey] = useState('')

  // Notify parent of apiKey changes (for resume functionality)
  useEffect(() => {
    onApiKeyChange?.(apiKey)
  }, [apiKey, onApiKeyChange])

  // Local model
  const [modelPath, setModelPath] = useState('')
  const [backend, setBackend] = useState('auto')
  const [backendParamValues, setBackendParamValues] = useState<Record<string, string>>(DEFAULT_PARAM_VALUES)
  const [showBackendOpts, setShowBackendOpts] = useState(false)

  // Dataset
  const [datasetHub, setDatasetHub] = useState('modelscope')
  const [datasets, setDatasets] = useState(initialDataset ?? '')
  const [datasetPath, setDatasetPath] = useState('')
  const [datasetLocalType, setDatasetLocalType] = useState('general_qa')
  const [datasetDir, setDatasetDir] = useState('')
  const isLocalDataset = datasetHub === 'local'

  // Common
  const [limit, setLimit] = useState('5')
  const [evalBatchSize, setEvalBatchSize] = useState('16')
  const [showMore, setShowMore] = useState(false)
  const [repeats, setRepeats] = useState('1')
  const [timeout, setTimeout_] = useState('60')
  const [stream, setStream] = useState(false)
  const [temperature, setTemperature] = useState('')
  const [topP, setTopP] = useState('')
  const [maxTokens, setMaxTokens] = useState('')
  const [topK, setTopK] = useState('')
  const [seed, setSeed] = useState('42')
  const [judgeStrategy, setJudgeStrategy] = useState('auto')
  const [ignoreErrors, setIgnoreErrors] = useState(false)
  const [datasetArgs, setDatasetArgs] = useState('')

  // Judge model (for analysis report)
  const [judgeModel, setJudgeModel] = useState('')
  const [judgeApiUrl, setJudgeApiUrl] = useState('')
  const [judgeApiKey, setJudgeApiKey] = useState('')

  // Validation
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Dataset autocomplete
  const [benchmarkNames, setBenchmarkNames] = useState<string[]>([])
  const GENERIC_LOCAL_TYPES = ['general_qa', 'general_mcq', 'general_fc']
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])
  const datasetInputRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (initialDataset) setDatasets(initialDataset)
  }, [initialDataset])

  useEffect(() => {
    listBenchmarks()
      .then((res) => {
        const names = [
          ...(res.text ?? []).map((b: { name: string }) => b.name),
          ...(res.multimodal ?? []).map((b: { name: string }) => b.name),
        ]
        setBenchmarkNames(names)
      })
      .catch((e) => { toast.error(e instanceof Error ? e.message : 'Failed to load benchmarks') })
  }, [])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (datasetInputRef.current && !datasetInputRef.current.contains(e.target as Node)) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleDatasetChange = (val: string) => {
    setDatasets(val)
    const parts = val.split(',')
    const current = parts[parts.length - 1].trim().toLowerCase()
    if (current) {
      const matches = benchmarkNames.filter((n) => n.toLowerCase().includes(current))
      setFilteredSuggestions(matches.slice(0, 8))
      setShowSuggestions(matches.length > 0)
    } else {
      setShowSuggestions(false)
    }
    if (errors.datasets) setErrors((prev) => ({ ...prev, datasets: '' }))
  }

  const selectSuggestion = (name: string) => {
    const parts = datasets.split(/[,，]/).map((s) => s.trim())
    parts[parts.length - 1] = name
    setDatasets(parts.join(', '))
    setShowSuggestions(false)
  }

  const setParam = (key: string, value: string) => {
    setBackendParamValues((prev) => ({ ...prev, [key]: value }))
  }

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}
    if (!isLocal && !model.trim()) newErrors.model = 'Required'
    if (isLocal && !modelPath.trim()) newErrors.modelPath = 'Required'
    if (!isLocal && !apiUrl.trim()) newErrors.apiUrl = 'Required'
    if (isLocalDataset) {
      if (!datasetPath.trim()) newErrors.datasetPath = 'Required'
    } else {
      if (!datasets.trim()) newErrors.datasets = 'Required'
    }

    // URL format
    if (!isLocal && apiUrl.trim()) {
      try {
        const u = new URL(apiUrl.trim())
        if (!['http:', 'https:'].includes(u.protocol)) {
          newErrors.apiUrl = 'URL 必须以 http:// 或 https:// 开头'
        }
      } catch {
        newErrors.apiUrl = 'URL 格式不正确'
      }
    }

    // Numeric range checks
    const checkPositiveInt = (val: string, key: string, label: string) => {
      if (val) {
        const n = Number(val)
        if (!Number.isInteger(n) || n < 1) newErrors[key] = `${label} 必须为正整数`
      }
    }
    checkPositiveInt(limit, 'limit', '样本数')
    checkPositiveInt(evalBatchSize, 'evalBatchSize', '批大小')
    checkPositiveInt(repeats, 'repeats', '重复次数')
    checkPositiveInt(timeout, 'timeout', '超时时间')

    if (temperature) {
      const t = Number(temperature)
      if (isNaN(t) || t < 0 || t > 2) newErrors.temperature = '温度范围 0~2'
    }
    if (topP) {
      const p = Number(topP)
      if (isNaN(p) || p < 0 || p > 1) newErrors.topP = 'Top P 范围 0~1'
    }
    if (maxTokens) {
      const m = Number(maxTokens)
      if (!Number.isInteger(m) || m < 1) newErrors.maxTokens = '最大 Token 数必须为正整数'
    }
    if (topK) {
      const k = Number(topK)
      if (!Number.isInteger(k) || k < 1) newErrors.topK = 'Top K 必须为正整数'
    }

    // JSON format for datasetArgs
    if (datasetArgs) {
      try { JSON.parse(datasetArgs) } catch { newErrors.datasetArgs = 'JSON 格式不正确' }
    }

    if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return }
    setErrors({})

    const config: Record<string, unknown> = {
      model_source: modelSource,
      model: isLocal ? (model || modelPath.split('/').pop() || 'local-model') : model,
      limit: limit ? Number(limit) : undefined,
      eval_batch_size: evalBatchSize ? Number(evalBatchSize) : undefined,
    }

    // Model
    if (isLocal) {
      config.model_path = modelPath
      config.backend = backend
      const ba: Record<string, unknown> = {}
      // Build backend_args from param definitions (only for the resolved backend)
      const params = getParams(backend)
      for (const p of params) {
        const val = backendParamValues[p.key]
        if (val === '' || val === undefined) continue
        if (p.type === 'checkbox') {
          if (val === 'true') ba[p.key] = true
        } else if (p.type === 'number') {
          const n = Number(val)
          if (!isNaN(n)) ba[p.key] = n
        } else {
          ba[p.key] = val
        }
      }
      if (Object.keys(ba).length > 0) config.backend_args = ba
    } else {
      if (apiUrl) config.api_url = apiUrl.trim()
      if (apiKey) config.api_key = apiKey
    }

    // Datasets
    if (isLocalDataset) {
      config.datasets = [datasetLocalType]
      config.dataset_hub = 'local'
      if (datasetDir) config.dataset_dir = datasetDir
      config.dataset_args = { [datasetLocalType]: { local_path: datasetPath } }
    } else {
      config.datasets = datasets.split(/[,，]/).map((s) => s.trim()).filter(Boolean)
      config.dataset_hub = datasetHub
      if (datasetDir) config.dataset_dir = datasetDir
    }

    if (repeats && Number(repeats) > 1) config.repeats = Number(repeats)
    if (timeout) config.timeout = Number(timeout)
    if (stream) config.stream = true
    const genConfig: Record<string, unknown> = {}
    if (temperature) genConfig.temperature = Number(temperature)
    if (topP) genConfig.top_p = Number(topP)
    if (maxTokens) genConfig.max_tokens = Number(maxTokens)
    if (topK) genConfig.top_k = Number(topK)
    if (Object.keys(genConfig).length > 0) config.generation_config = genConfig
    if (seed && seed !== '42') config.seed = Number(seed)
    if (judgeStrategy && judgeStrategy !== 'auto') config.judge_strategy = judgeStrategy
    if (ignoreErrors) config.ignore_errors = true
    if (datasetArgs) { try { const extra = JSON.parse(datasetArgs); config.dataset_args = { ...(config.dataset_args as Record<string, unknown> || {}), ...extra } } catch { /* ignore */ } }

    // Judge model args (for analysis report generation)
    if (judgeModel.trim() || judgeApiUrl.trim() || judgeApiKey.trim()) {
      const jma: Record<string, unknown> = {}
      if (judgeModel.trim()) jma.model_id = judgeModel.trim()
      if (judgeApiUrl.trim()) jma.api_url = judgeApiUrl.trim()
      if (judgeApiKey.trim()) jma.api_key = judgeApiKey
      config.judge_model_args = jma
    }

    onSubmit(config)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Model Source */}
      <div className="flex items-center gap-6">
        <label className={`${FORM_LABEL_CLASS} !mb-0`}>{t('eval.modelSource')}</label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="ms" value="openai" checked={!isLocal}
            onChange={() => setModelSource('openai')} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">{t('eval.modelSourceOpenAI')}</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="ms" value="local" checked={isLocal}
            onChange={() => setModelSource('local')} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">{t('eval.modelSourceLocal')}</span>
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* Local model fields */}
        {isLocal && (<>
          <FormField label={t('eval.modelPath')} required error={errors.modelPath}>
            <input value={modelPath}
              onChange={(e) => { setModelPath(e.target.value); if (errors.modelPath) setErrors((p) => ({ ...p, modelPath: '' })) }}
              className={inputClass(errors.modelPath)} placeholder="/data/models/qwen.gguf" />
          </FormField>
          <FormField label={t('eval.modelName')}>
            <input value={model} onChange={(e) => setModel(e.target.value)} className={FORM_INPUT_CLASS}
              placeholder={modelPath ? modelPath.split('/').pop() || '' : t('eval.modelNamePlaceholder')} />
          </FormField>
          <FormField label={t('eval.backend')}>
            <select value={backend} onChange={(e) => setBackend(e.target.value)} className={FORM_INPUT_CLASS}>
              <option value="auto">{t('eval.backendAuto')}</option>
              <option value="vllm">{t('eval.backendVllm')}</option>
              <option value="sglang">{t('eval.backendSglang')}</option>
              <option value="llama_cpp">{t('eval.backendLlamaCpp')}</option>
              <option value="transformers">{t('eval.backendTransformers')}</option>
            </select>
          </FormField>
          <div className="md:col-span-2">
          {backend !== 'auto' && (
            <>
              <button type="button" onClick={() => setShowBackendOpts(!showBackendOpts)}
                className="flex items-center gap-1 text-sm text-[var(--accent)] hover:underline cursor-pointer">
                {t('eval.backendAdvanced')}
                {showBackendOpts ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
              {showBackendOpts && (
              <Card className="!p-0 mt-2">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3">
                  {getParams(backend).map((p) => {
                    const val = backendParamValues[p.key] || ''
                    const label = t(p.label)
                    const ph = p.placeholder ? t(p.placeholder, { v: DEFAULT_PARAM_VALUES[p.key] || '' }) : undefined
                    if (p.type === 'checkbox') {
                      return (
                        <div key={p.key} className="flex items-end pb-0.5">
                          <label className="flex items-center gap-1.5 text-sm text-[var(--text-muted)] cursor-pointer">
                            <input type="checkbox"
                              checked={val === 'true'}
                              onChange={(e) => setParam(p.key, e.target.checked ? 'true' : 'false')}
                              className="accent-[var(--accent)]" />
                            {label}
                          </label>
                        </div>
                      )
                    }
                    if (p.type === 'select' && p.options) {
                      return (
                        <FormField key={p.key} label={label}>
                          <select value={val}
                            onChange={(e) => setParam(p.key, e.target.value)}
                            className={FORM_INPUT_CLASS}>
                            {p.options.map((opt) => (
                              <option key={opt} value={opt === 'eval.params.none' ? '' : opt}>{opt.includes('eval.params.') ? t(opt) : opt}</option>
                            ))}
                          </select>
                        </FormField>
                      )
                    }
                    return (
                      <FormField key={p.key} label={label}>
                        <input type="number" step={p.step || '1'} min={p.min} max={p.max}
                          value={val}
                          onChange={(e) => {
                            let v = e.target.value
                            const isInt = !p.step
                            if (isInt) {
                              v = v.replace(/[^0-9]/g, '')
                            } else {
                              v = v.replace(/[^0-9.]/g, '')
                              const parts = v.split('.')
                              if (parts.length > 2) v = parts[0] + '.' + parts.slice(1).join('')
                            }
                            if (v !== '' && p.min !== undefined && Number(v) < p.min) v = ''
                            if (v !== '' && p.max !== undefined && Number(v) > p.max) v = String(p.max)
                            setParam(p.key, v)
                          }}
                          className={FORM_INPUT_CLASS}
                          placeholder={ph} />
                      </FormField>
                    )
                  })}
                </div>
              </Card>
            )}
            </>
            )}
          </div>
        </>)}

        {/* OpenAI API fields */}
        {!isLocal && (<>
          <FormField label={t('eval.modelName')} required error={errors.model}>
            <input value={model}
              onChange={(e) => { setModel(e.target.value.trimStart()); if (errors.model) setErrors((p) => ({ ...p, model: '' })) }}
              className={inputClass(errors.model)} placeholder="Qwen/Qwen2.5-0.5B-Instruct" />
          </FormField>
          <FormField label={t('eval.apiUrl')} required error={errors.apiUrl}>
            <input value={apiUrl}
              onChange={(e) => { setApiUrl(e.target.value); if (errors.apiUrl) setErrors((p) => ({ ...p, apiUrl: '' })) }}
              className={inputClass(errors.apiUrl)} placeholder="http://localhost:8000/v1" />
          </FormField>
          <FormField label={t('eval.apiKey')}>
            <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className={FORM_INPUT_CLASS} placeholder="sk-..." />
          </FormField>
        </>)}

        {/* Dataset Source */}
        <FormField label={t('eval.datasetHub')}>
          <select value={datasetHub} onChange={(e) => setDatasetHub(e.target.value)} className={FORM_INPUT_CLASS}>
            <option value="modelscope">{t('eval.datasetHubModelScope')}</option>
            <option value="huggingface">{t('eval.datasetHubHuggingFace')}</option>
            <option value="local">{t('eval.datasetHubLocal')}</option>
          </select>
        </FormField>

        {isLocalDataset ? (<>
          <FormField label={t('eval.datasetLocalType')} required>
            <select value={datasetLocalType} onChange={(e) => setDatasetLocalType(e.target.value)} className={FORM_INPUT_CLASS}>
              <option value="general_qa">{t('eval.datasetLocalTypeQA')}</option>
              <option value="general_mcq">{t('eval.datasetLocalTypeMCQ')}</option>
              <option value="general_fc">{t('eval.datasetLocalTypeFC')}</option>
              <option disabled>──</option>
              {benchmarkNames.filter((n) => !GENERIC_LOCAL_TYPES.includes(n)).map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </FormField>
          <FormField label={t('eval.datasetPath')} required error={errors.datasetPath}>
            <input value={datasetPath}
              onChange={(e) => { setDatasetPath(e.target.value); if (errors.datasetPath) setErrors((p) => ({ ...p, datasetPath: '' })) }}
              className={inputClass(errors.datasetPath)} placeholder="/data/datasets/my_benchmark" />
          </FormField>
        </>) : (
          <FormField label={t('eval.datasets')} required error={errors.datasets} className="relative">
            <div ref={datasetInputRef}>
              <input value={datasets}
                onChange={(e) => handleDatasetChange(e.target.value)}
                onFocus={() => { if (filteredSuggestions.length) setShowSuggestions(true) }}
                className={inputClass(errors.datasets)} placeholder="gsm8k, arc" />
              {showSuggestions && (
                <div className="absolute z-50 left-0 right-0 mt-1 rounded-[var(--radius-sm)] border border-[var(--border-md)] bg-[var(--bg-card)] shadow-[var(--shadow)] overflow-hidden max-h-48 overflow-y-auto">
                  {filteredSuggestions.map((name) => (
                    <button key={name} type="button" onClick={() => selectSuggestion(name)}
                      className="w-full text-left px-3 py-2 text-sm text-[var(--text)] hover:bg-[var(--bg-card2)] transition-colors cursor-pointer">
                      {name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </FormField>
        )}

        <FormField label={t('eval.datasetDir')}>
          <input value={datasetDir} onChange={(e) => setDatasetDir(e.target.value)} className={FORM_INPUT_CLASS}
            placeholder="~/.cache/modelscope/hub/datasets" />
        </FormField>

        <FormField label={t('eval.limit')} error={errors.limit}>
          <input type="number" value={limit} onChange={(e) => { setLimit(e.target.value.replace(/[^0-9]/g, '')); if (errors.limit) setErrors((p) => ({ ...p, limit: '' })) }} className={inputClass(errors.limit)} />
        </FormField>

        <FormField label={t('eval.batchSize')} error={errors.evalBatchSize}>
          <input type="number" value={evalBatchSize} onChange={(e) => { setEvalBatchSize(e.target.value.replace(/[^0-9]/g, '')); if (errors.evalBatchSize) setErrors((p) => ({ ...p, evalBatchSize: '' })) }} className={inputClass(errors.evalBatchSize)} />
        </FormField>
      </div>

      {/* More params toggle */}
      <button type="button" onClick={() => setShowMore(!showMore)}
        className="flex items-center gap-1 text-sm text-[var(--accent)] hover:underline cursor-pointer">
        {t('eval.moreParams')}
        {showMore ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {showMore && (
        <Card className="!p-0">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4">
            {/* Row 1 — 采样参数 */}
            <FormField label={t('eval.temperature')} error={errors.temperature}>
              <input type="number" step="0.1" min={0} max={2} value={temperature}
                onChange={(e) => {
                  let v = e.target.value.replace(/[^0-9.]/g, '')
                  if (v !== '' && Number(v) > 2) v = '2'
                  setTemperature(v)
                  if (errors.temperature) setErrors((p) => ({ ...p, temperature: '' }))
                }}
                className={inputClass(errors.temperature)} />
            </FormField>
            <FormField label={t('eval.topP')} error={errors.topP}>
              <input type="number" step="0.05" min={0} max={1} value={topP}
                onChange={(e) => {
                  let v = e.target.value.replace(/[^0-9.]/g, '')
                  if (v !== '' && Number(v) > 1) v = '1'
                  setTopP(v)
                  if (errors.topP) setErrors((p) => ({ ...p, topP: '' }))
                }}
                className={inputClass(errors.topP)} />
            </FormField>
            <FormField label={t('eval.topK')} error={errors.topK}>
              <input type="number" min={1} step="1" value={topK}
                onChange={(e) => {
                  const v = e.target.value.replace(/[^0-9]/g, '')
                  setTopK(v)
                  if (errors.topK) setErrors((p) => ({ ...p, topK: '' }))
                }}
                className={inputClass(errors.topK)} />
            </FormField>
            {/* Row 2 — 长度 + 运行控制 */}
            <FormField label={t('eval.maxTokens')} error={errors.maxTokens}>
              <input type="number" min={1} step="1" value={maxTokens}
                onChange={(e) => {
                  const v = e.target.value.replace(/[^0-9]/g, '')
                  setMaxTokens(v)
                  if (errors.maxTokens) setErrors((p) => ({ ...p, maxTokens: '' }))
                }}
                className={inputClass(errors.maxTokens)} />
            </FormField>
            <FormField label={t('eval.repeats')} error={errors.repeats}>
              <input type="number" min={1} step="1" value={repeats}
                onChange={(e) => {
                  const v = e.target.value.replace(/[^0-9]/g, '')
                  setRepeats(v)
                  if (errors.repeats) setErrors((p) => ({ ...p, repeats: '' }))
                }}
                className={inputClass(errors.repeats)} />
            </FormField>
            <FormField label={t('eval.timeout')} error={errors.timeout}>
              <input type="number" min={1} step="1" value={timeout}
                onChange={(e) => {
                  const v = e.target.value.replace(/[^0-9]/g, '')
                  setTimeout_(v)
                  if (errors.timeout) setErrors((p) => ({ ...p, timeout: '' }))
                }}
                className={inputClass(errors.timeout)} />
            </FormField>
            {/* Row 3 — 种子 + 评判 + 开关 */}
            <FormField label={t('eval.seed')}>
              <input type="number" min={1} step="1" value={seed}
                onChange={(e) => {
                  const v = e.target.value.replace(/[^0-9]/g, '')
                  setSeed(v)
                }}
                className={FORM_INPUT_CLASS} />
            </FormField>
            <FormField label={t('eval.judgeStrategy')}>
              <select value={judgeStrategy} onChange={(e) => setJudgeStrategy(e.target.value)} className={FORM_INPUT_CLASS}>
                <option value="auto">auto</option>
                <option value="rule">rule</option>
                <option value="llm">llm</option>
                <option value="llm_recall">llm_recall</option>
              </select>
            </FormField>
            <div className="flex items-end gap-4 pb-0.5">
              <label className="flex items-center gap-1.5 text-sm text-[var(--text-muted)] cursor-pointer">
                <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} className="accent-[var(--accent)]" />
                {t('eval.stream')}
              </label>
              <label className="flex items-center gap-1.5 text-sm text-[var(--text-muted)] cursor-pointer">
                <input type="checkbox" checked={ignoreErrors} onChange={(e) => setIgnoreErrors(e.target.checked)} className="accent-[var(--accent)]" />
                {t('eval.ignoreErrors')}
              </label>
            </div>
            {/* Row 4 — 数据集参数 */}
            <div className="md:col-span-3">
              <label className={FORM_LABEL_CLASS}>{t('eval.datasetArgs')}</label>
              <textarea value={datasetArgs}
                onChange={(e) => { setDatasetArgs(e.target.value); if (errors.datasetArgs) setErrors((p) => ({ ...p, datasetArgs: '' })) }}
                className={`${inputClass(errors.datasetArgs)} h-20 resize-y`} style={{ fontFamily: 'var(--font-mono)' }}
                placeholder='{"gsm8k": {"few_shot_num": 4}}' />
              {errors.datasetArgs && <p className="mt-1 text-xs text-red-500">{errors.datasetArgs}</p>}
            </div>
            {/* Row 5 — 评判模型（用于生成分析报告） */}
            <div className="md:col-span-3 border-t border-[var(--border-md)] pt-4 mt-2">
              <p className={`${FORM_LABEL_CLASS} mb-2`}>{t('eval.judgeModelTitle')}</p>
              <p className="text-xs text-[var(--text-muted)] mb-3">{t('eval.judgeModelHint')}</p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <FormField label={t('eval.judgeModel')}>
                  <input value={judgeModel} onChange={(e) => setJudgeModel(e.target.value)}
                    className={FORM_INPUT_CLASS} placeholder="Qwen/Qwen3-235B-A22B" />
                </FormField>
                <FormField label={t('eval.judgeApiUrl')}>
                  <input value={judgeApiUrl} onChange={(e) => setJudgeApiUrl(e.target.value)}
                    className={FORM_INPUT_CLASS} placeholder="https://api-inference.modelscope.cn/v1/" />
                </FormField>
                <FormField label={t('eval.judgeApiKey')}>
                  <input type="password" value={judgeApiKey} onChange={(e) => setJudgeApiKey(e.target.value)}
                    className={FORM_INPUT_CLASS} placeholder="sk-..." />
                </FormField>
              </div>
            </div>
          </div>
        </Card>
      )}

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow">
        {t('eval.startEval')}
      </Button>
    </form>
  )
}
