import { useState, useEffect, useRef, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import Collapsible from '@/components/ui/Collapsible'
import { FORM_INPUT_CLASS, FORM_LABEL_CLASS, inputClass } from '@/components/ui/formStyles'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

export default function PerfConfigForm({ onSubmit, disabled }: Props) {
  const { t } = useLocale()
  const [modelSource, setModelSource] = useState<'openai' | 'local'>('openai')
  const isLocal = modelSource === 'local'
  const modelManualRef = useRef(false)

  // OpenAI API fields
  const [model, setModel] = useState('')
  const [url, setUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [api, setApi] = useState('openai')

  // Local model fields
  const [modelPath, setModelPath] = useState('')
  const [backend, setBackend] = useState('auto')

  // Common fields
  const [parallel, setParallel] = useState('1')
  const [number, setNumber] = useState('10')
  const [rate, setRate] = useState('')
  const [maxTokens, setMaxTokens] = useState('')
  const [minTokens, setMinTokens] = useState('')
  const [dataset, setDataset] = useState('openqa')
  const [customDataset, setCustomDataset] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [maxPromptLen, setMaxPromptLen] = useState('')
  const [minPromptLen, setMinPromptLen] = useState('')
  const [tokenizerPath, setTokenizerPath] = useState('')
  const [prefixLength, setPrefixLength] = useState('')
  const [extraArgs, setExtraArgs] = useState('')

  const [errors, setErrors] = useState<Record<string, string>>({})

  // Auto-fill model name from path for local models
  useEffect(() => {
    if (!isLocal || !modelPath || modelManualRef.current) return
    const name = modelPath.split('/').pop() || ''
    if (name) setModel(name)
  }, [modelPath, isLocal])

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}
    if (isLocal) {
      if (!modelPath.trim()) newErrors.modelPath = 'Required'
    } else {
      if (!model.trim()) newErrors.model = 'Required'
      if (!url.trim()) newErrors.url = 'Required'
    }
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }
    setErrors({})

    const config: Record<string, unknown> = {
      parallel: parallel.replace(/，/g, ',').split(',').map((s) => Number(s.trim())).filter(Boolean),
      number: number.replace(/，/g, ',').split(',').map((s) => Number(s.trim())).filter(Boolean),
    }

    if (isLocal) {
      config.model = modelPath
      config.model_path = modelPath
      // Map backend to perf api type
      if (backend === 'auto') {
        config.api = 'local'  // auto-detect, default to local/transformers
      } else if (backend === 'vllm') {
        config.api = 'local_vllm'
      } else {
        config.api = 'local'
      }
      if (tokenizerPath) config.tokenizer_path = tokenizerPath
    } else {
      config.model = model
      config.api = api
      config.url = url
      if (apiKey) config.api_key = apiKey
      if (tokenizerPath) config.tokenizer_path = tokenizerPath
    }

    if (rate) config.rate = Number(rate)
    if (maxTokens) config.max_tokens = Number(maxTokens)
    if (minTokens) config.min_tokens = Number(minTokens)
    if (dataset) config.dataset = dataset === 'custom' ? 'local_jsonl' : dataset
    if (dataset === 'custom' && customDataset) config.dataset_label = customDataset
    if (datasetPath) config.dataset_path = datasetPath
    if (maxPromptLen) config.max_prompt_length = Number(maxPromptLen)
    if (minPromptLen) config.min_prompt_length = Number(minPromptLen)
    if (prefixLength) config.prefix_length = Number(prefixLength)
    if (extraArgs.trim()) {
      try { config.extra_args = JSON.parse(extraArgs) }
      catch { newErrors.extra_args = 'JSON 格式无效' }
    }
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
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
          <span className="text-sm text-[var(--text)]">API</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="ms" value="local" checked={isLocal}
            onChange={() => setModelSource('local')} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">{t('eval.modelSourceLocal')}</span>
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* ── OpenAI API fields ── */}
        {!isLocal && (<>
          <FormField label={t('eval.modelName')} required error={errors.model}>
            <input
              value={model}
              onChange={(e) => { setModel(e.target.value); if (errors.model) setErrors((p) => ({ ...p, model: '' })) }}
              className={inputClass(errors.model)}
              placeholder="Qwen/Qwen2.5-0.5B-Instruct"
            />
          </FormField>

          <FormField label={t('perf.apiType')}>
            <select value={api} onChange={(e) => setApi(e.target.value)} className={FORM_INPUT_CLASS}>
              <option value="openai">OpenAI</option>
              <option value="openai_responses">OpenAI Responses</option>
              <option value="openai_embedding">OpenAI Embedding</option>
              <option value="openai_rerank">OpenAI Rerank</option>
              <option value="dashscope">DashScope</option>
              <option value="custom">Custom</option>
            </select>
          </FormField>

          <FormField label={t('eval.apiUrl')} required error={errors.url}>
            <input value={url}
              onChange={(e) => { setUrl(e.target.value); if (errors.url) setErrors((p) => ({ ...p, url: '' })) }}
              className={inputClass(errors.url)} placeholder="http://localhost:8000/v1" />
          </FormField>

          <FormField label={t('eval.apiKey')}>
            <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className={FORM_INPUT_CLASS} placeholder="sk-..." />
          </FormField>
        </>)}

        {/* ── Local model fields ── */}
        {isLocal && (<>
          <FormField label={t('eval.modelPath')} required error={errors.modelPath}>
            <input value={modelPath}
              onChange={(e) => { setModelPath(e.target.value); if (errors.modelPath) setErrors((p) => ({ ...p, modelPath: '' })) }}
              className={inputClass(errors.modelPath)} placeholder="/data/models/qwen.gguf" />
          </FormField>
          <FormField label={t('eval.modelName')}>
            <input value={model} onChange={(e) => { setModel(e.target.value); modelManualRef.current = true }} className={FORM_INPUT_CLASS}
              placeholder={t('eval.modelNamePlaceholder')} />
          </FormField>
          <FormField label={t('eval.backend')}>
            <select value={backend} onChange={(e) => setBackend(e.target.value)} className={FORM_INPUT_CLASS}>
              <option value="auto">Auto Detect</option>
              <option value="transformers">Transformers</option>
              <option value="vllm">vLLM</option>
              <option value="sglang">SGLang</option>
              <option value="llama_cpp">llama.cpp</option>
            </select>
          </FormField>
          <FormField label="Tokenizer 路径">
            <input value={tokenizerPath} onChange={(e) => setTokenizerPath(e.target.value)} className={FORM_INPUT_CLASS} placeholder="/data/models/Qwen3-8B/" />
          </FormField>
        </>)}

        {/* ── 数据集 ── */}
        <FormField label={t('perf.dataset')}>
          <select value={dataset} onChange={(e) => setDataset(e.target.value)} className={FORM_INPUT_CLASS}>
            <option value="openqa">openqa（默认）</option>
            <option value="random">random</option>
            <option value="random_vl">random_vl</option>
            <option value="random_multi_turn">random_multi_turn</option>
            <option value="share_gpt_zh">share_gpt_zh</option>
            <option value="share_gpt_en">share_gpt_en</option>
            <option value="longalpaca">longalpaca</option>
            <option value="line_by_line">line_by_line</option>
            <option value="speed_benchmark">speed_benchmark</option>
            <option value="custom">其他（手动输入）</option>
          </select>
        </FormField>

        <FormField label={t('perf.rate')}>
          <input type="number" value={rate} onChange={(e) => setRate(e.target.value)} className={FORM_INPUT_CLASS} placeholder="请求/秒" />
        </FormField>

        {dataset === 'custom' && (
          <>
            <FormField label="自定义数据集名称">
              <input value={customDataset} onChange={(e) => setCustomDataset(e.target.value)} className={FORM_INPUT_CLASS} placeholder="输入数据集名称" />
            </FormField>
            <FormField label="数据集路径">
              <input value={datasetPath} onChange={(e) => setDatasetPath(e.target.value)} className={FORM_INPUT_CLASS} placeholder="/data/datasets/my_perf_data.jsonl" />
            </FormField>
          </>
        )}

        {/* ── 压测参数 ── */}
        <FormField label={t('perf.parallel')}>
          <input value={parallel} onChange={(e) => setParallel(e.target.value)} className={FORM_INPUT_CLASS} placeholder="1, 4, 8" />
        </FormField>

        <FormField label={t('perf.number')}>
          <input value={number} onChange={(e) => setNumber(e.target.value)} className={FORM_INPUT_CLASS} placeholder="10, 100" />
        </FormField>

        {/* ── Token / Prompt ── */}
        <FormField label={t('perf.maxTokens')}>
          <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} className={FORM_INPUT_CLASS} placeholder="默认 2048" />
        </FormField>

        <FormField label={t('perf.minTokens')}>
          <input type="number" value={minTokens} onChange={(e) => setMinTokens(e.target.value)} className={FORM_INPUT_CLASS} placeholder="不限制" />
        </FormField>

        <FormField label={t('perf.maxPromptLen')}>
          <input type="number" value={maxPromptLen} onChange={(e) => setMaxPromptLen(e.target.value)} className={FORM_INPUT_CLASS} placeholder="默认 131072" />
        </FormField>

        <FormField label={t('perf.minPromptLen')}>
          <input type="number" value={minPromptLen} onChange={(e) => setMinPromptLen(e.target.value)} className={FORM_INPUT_CLASS} placeholder="默认 0" />
        </FormField>

        {/* API mode: Tokenizer path shown here */}
        {!isLocal && (
          <FormField label="Tokenizer 路径">
            <input value={tokenizerPath} onChange={(e) => setTokenizerPath(e.target.value)} className={FORM_INPUT_CLASS} placeholder="/data/models/Qwen3-8B/" />
          </FormField>
        )}
      </div>

      {/* ── 高级选项 ── */}
      <Collapsible header={<span className="text-sm text-[var(--accent)]">更多参数</span>} defaultOpen={false} chevronAfter chevronColor="var(--accent)">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
          <FormField label="Prefix 长度">
            <input type="number" value={prefixLength} onChange={(e) => setPrefixLength(e.target.value.replace(/[^0-9]/g, ''))} className={FORM_INPUT_CLASS} placeholder="0" />
          </FormField>

          <FormField label="Extra Args (JSON)" className="md:col-span-2" error={errors.extra_args}>
            <textarea
              value={extraArgs}
              onChange={(e) => { setExtraArgs(e.target.value); if (errors.extra_args) setErrors((p) => ({ ...p, extra_args: '' })) }}
              className={`${FORM_INPUT_CLASS} font-mono text-xs`}
              rows={3}
              placeholder='{"ignore_eos": true}'
            />
          </FormField>
        </div>
      </Collapsible>

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow">
        {t('perf.startPerf')}
      </Button>
    </form>
  )
}
