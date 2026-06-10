import { useEffect, useRef, useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listBenchmarks } from '@/api/eval'
import Button from '@/components/ui/Button'
import Card from '@/components/ui/Card'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, FORM_LABEL_CLASS, inputClass } from '@/components/ui/formStyles'
import { ChevronDown, ChevronUp } from 'lucide-react'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
  initialDataset?: string
}

export default function EvalConfigForm({ onSubmit, disabled, initialDataset }: Props) {
  const { t } = useLocale()

  // Model source
  const [modelSource, setModelSource] = useState<'openai' | 'local'>('openai')
  const isLocal = modelSource === 'local'

  // OpenAI API
  const [model, setModel] = useState('')
  const [apiUrl, setApiUrl] = useState('')
  const [apiKey, setApiKey] = useState('')

  // Local model
  const [modelPath, setModelPath] = useState('')
  const [backend, setBackend] = useState('auto')
  const [nCtx, setNCtx] = useState('2048')
  const [nThreads, setNThreads] = useState('4')
  const [nGpuLayers, setNGpuLayers] = useState('0')
  const [precision, setPrecision] = useState('float16')
  const [deviceMap, setDeviceMap] = useState('auto')
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
  const [datasetArgs, setDatasetArgs] = useState('')

  // Validation
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Dataset autocomplete
  const [benchmarkNames, setBenchmarkNames] = useState<string[]>([])
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
      .catch(() => {})
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
    const parts = datasets.split(',').map((s) => s.trim())
    parts[parts.length - 1] = name
    setDatasets(parts.join(', '))
    setShowSuggestions(false)
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
      if (nCtx) ba.n_ctx = Number(nCtx)
      if (nThreads) ba.n_threads = Number(nThreads)
      if (nGpuLayers) ba.n_gpu_layers = Number(nGpuLayers)
      if (backend === 'transformers') {
        if (precision) ba.precision = `torch.${precision}`
        if (deviceMap) ba.device_map = deviceMap
      }
      if (Object.keys(ba).length > 0) config.backend_args = ba
    } else {
      if (apiUrl) config.api_url = apiUrl
      if (apiKey) config.api_key = apiKey
    }

    // Datasets
    if (isLocalDataset) {
      config.datasets = [datasetLocalType]
      config.dataset_hub = 'local'
      if (datasetDir) config.dataset_dir = datasetDir
      config.dataset_args = { [datasetLocalType]: { local_path: datasetPath } }
    } else {
      config.datasets = datasets.split(',').map((s) => s.trim()).filter(Boolean)
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
    if (datasetArgs) { try { const extra = JSON.parse(datasetArgs); config.dataset_args = { ...(config.dataset_args as Record<string, unknown> || {}), ...extra } } catch { /* ignore */ } }
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
            <button type="button" onClick={() => setShowBackendOpts(!showBackendOpts)}
              className="flex items-center gap-1 text-xs text-[var(--accent)] hover:underline cursor-pointer">
              {t('eval.backendAdvanced')}
              {showBackendOpts ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {showBackendOpts && (
              <Card className="!p-0 mt-2">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3">
                  <FormField label={t('eval.nCtx')}><input type="number" value={nCtx} onChange={(e) => setNCtx(e.target.value)} className={FORM_INPUT_CLASS} /></FormField>
                  <FormField label={t('eval.nThreads')}><input type="number" value={nThreads} onChange={(e) => setNThreads(e.target.value)} className={FORM_INPUT_CLASS} /></FormField>
                  <FormField label={t('eval.nGpuLayers')}><input type="number" value={nGpuLayers} onChange={(e) => setNGpuLayers(e.target.value)} className={FORM_INPUT_CLASS} /></FormField>
                  {backend === 'transformers' && (<>
                    <FormField label={t('eval.precision')}>
                      <select value={precision} onChange={(e) => setPrecision(e.target.value)} className={FORM_INPUT_CLASS}>
                        <option value="float16">float16</option><option value="bfloat16">bfloat16</option><option value="float32">float32</option><option value="auto">auto</option>
                      </select>
                    </FormField>
                    <FormField label={t('eval.deviceMap')}>
                      <select value={deviceMap} onChange={(e) => setDeviceMap(e.target.value)} className={FORM_INPUT_CLASS}>
                        <option value="auto">auto</option><option value="cuda:0">cuda:0</option><option value="cuda:1">cuda:1</option><option value="cpu">cpu</option>
                      </select>
                    </FormField>
                  </>)}
                </div>
              </Card>
            )}
          </div>
        </>)}

        {/* OpenAI API fields */}
        {!isLocal && (<>
          <FormField label={t('eval.modelName')} required error={errors.model}>
            <input value={model}
              onChange={(e) => { setModel(e.target.value); if (errors.model) setErrors((p) => ({ ...p, model: '' })) }}
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

        <FormField label={t('eval.limit')}>
          <input type="number" value={limit} onChange={(e) => setLimit(e.target.value)} className={FORM_INPUT_CLASS} />
        </FormField>

        <FormField label={t('eval.batchSize')}>
          <input type="number" value={evalBatchSize} onChange={(e) => setEvalBatchSize(e.target.value)} className={FORM_INPUT_CLASS} />
        </FormField>
      </div>

      {/* More params toggle */}
      <button type="button" onClick={() => setShowMore(!showMore)}
        className="flex items-center gap-1 text-xs text-[var(--accent)] hover:underline cursor-pointer">
        {t('eval.moreParams')}
        {showMore ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {showMore && (
        <Card className="!p-0">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4">
            <FormField label={t('eval.repeats')}>
              <input type="number" value={repeats} onChange={(e) => setRepeats(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <FormField label={t('eval.timeout')}>
              <input type="number" value={timeout} onChange={(e) => setTimeout_(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <div className="flex items-end gap-2 pb-0.5">
              <label className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] cursor-pointer">
                <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} className="accent-[var(--accent)]" />
                {t('eval.stream')}
              </label>
            </div>
            <FormField label={t('eval.temperature')}>
              <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <FormField label={t('eval.topP')}>
              <input type="number" step="0.1" value={topP} onChange={(e) => setTopP(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <FormField label={t('eval.maxTokens')}>
              <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <FormField label={t('eval.topK')}>
              <input type="number" value={topK} onChange={(e) => setTopK(e.target.value)} className={FORM_INPUT_CLASS} />
            </FormField>
            <div className="md:col-span-2">
              <label className={FORM_LABEL_CLASS}>{t('eval.datasetArgs')}</label>
              <textarea value={datasetArgs} onChange={(e) => setDatasetArgs(e.target.value)}
                className={`${FORM_INPUT_CLASS} h-20 resize-y`} style={{ fontFamily: 'var(--font-mono)' }}
                placeholder='{"gsm8k": {"few_shot_num": 4}}' />
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
