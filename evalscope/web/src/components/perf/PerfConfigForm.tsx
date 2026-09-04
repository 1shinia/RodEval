import { useState, useEffect, useRef, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import Collapsible from '@/components/ui/Collapsible'
import { FORM_INPUT_CLASS, FORM_LABEL_CLASS, inputClass } from '@/components/ui/formStyles'
import { getTemplateDownloadUrl, uploadBatchCsv } from '@/api/perf'
import type { BatchUploadResponse } from '@/api/perf'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
  onApiKeyChange?: (key: string) => void
  onBatchSubmit?: (batchId: string, sharedConfig: Record<string, unknown>) => Promise<void>
  onModeChange?: (mode: 'single' | 'batch') => void
}

const EMBEDDING_APIS = ['openai_embedding']
const RERANK_APIS = ['openai_rerank']
const isEmbeddingOrRerank = (api: string) => EMBEDDING_APIS.includes(api) || RERANK_APIS.includes(api)

const EMBEDDING_DATASETS = ['random_embedding', 'embedding', 'random_embedding_batch', 'embedding_batch']
const RERANK_DATASETS = ['random_rerank', 'rerank']
const LLM_DATASETS = ['openqa', 'random', 'random_vl', 'random_multi_turn', 'share_gpt_zh', 'share_gpt_en', 'longalpaca', 'line_by_line', 'speed_benchmark']

// SLA auto-tuning metric options (metric values must match sla_run.get_metric_values)
const SLA_METRICS = [
  { value: 'avg_ttft', label: '平均首字延迟(s)' },
  { value: 'avg_latency', label: '平均延迟(s)' },
  { value: 'avg_tpot', label: '平均每Token延迟(s)' },
  { value: 'p99_latency', label: 'P99 延迟(s)' },
  { value: 'p99_ttft', label: 'P99 首字延迟(s)' },
  { value: 'p99_tpot', label: 'P99 每Token延迟(s)' },
  { value: 'rps', label: '每秒请求数' },
  { value: 'tps', label: '输出 Token/s' },
]

interface SlaRule {
  metric: string
  op: string
  value: string
}

export default function PerfConfigForm({ onSubmit, disabled, onApiKeyChange, onBatchSubmit, onModeChange }: Props) {
  const { t } = useLocale()
  const [testMode, setTestMode] = useState<'single' | 'batch'>('single')
  const isBatch = testMode === 'batch'

  // OpenAI API fields
  const [model, setModel] = useState('')
  const [url, setUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [api, setApi] = useState('openai')

  // Sync API key to parent for resume
  useEffect(() => { onApiKeyChange?.(apiKey) }, [apiKey, onApiKeyChange])

  // Common fields
  const [parallel, setParallel] = useState('1')
  const [number, setNumber] = useState('10')
  const [rate, setRate] = useState('')
  const [warmupRatio, setWarmupRatio] = useState('')
  const [duration, setDuration] = useState('')
  const [maxTokens, setMaxTokens] = useState('')
  const [minTokens, setMinTokens] = useState('')
  const [dataset, setDataset] = useState('openqa')
  const [customDataset, setCustomDataset] = useState('')
  const [datasetPath, setDatasetPath] = useState('')
  const [maxPromptLen, setMaxPromptLen] = useState('')
  const [minPromptLen, setMinPromptLen] = useState('')
  const [prefixLength, setPrefixLength] = useState('')
  const [thinkingMode, setThinkingMode] = useState('auto')
  const [extraArgs, setExtraArgs] = useState('')
  const [readTimeout, setReadTimeout] = useState('')

  // SLA auto-tuning state
  const [slaEnabled, setSlaEnabled] = useState(false)
  const [slaVariable, setSlaVariable] = useState<'parallel' | 'rate'>('parallel')
  const [slaRules, setSlaRules] = useState<SlaRule[]>([{ metric: 'avg_ttft', op: '<=', value: '' }])
  const [slaLower, setSlaLower] = useState('')
  const [slaUpper, setSlaUpper] = useState('')
  const [slaNumRuns, setSlaNumRuns] = useState('')

  const updateSlaRule = (i: number, field: keyof SlaRule, v: string) => {
    setSlaRules((prev) => prev.map((r, idx) => (idx === i ? { ...r, [field]: v } : r)))
  }
  const addSlaRule = () => setSlaRules((prev) => [...prev, { metric: 'avg_latency', op: '<=', value: '' }])
  const removeSlaRule = (i: number) => setSlaRules((prev) => prev.filter((_, idx) => idx !== i))

  // Batch state
  const [batchFile, setBatchFile] = useState<File | null>(null)
  const [batchInfo, setBatchInfo] = useState<BatchUploadResponse | null>(null)
  const [batchUploading, setBatchUploading] = useState(false)
  const [batchError, setBatchError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Reset dataset when switching between LLM and embedding/reranker APIs
  useEffect(() => {
    if (isEmbeddingOrRerank(api)) {
      if (![...EMBEDDING_DATASETS, ...RERANK_DATASETS].includes(dataset)) {
        setDataset(EMBEDDING_APIS.includes(api) ? EMBEDDING_DATASETS[0] : RERANK_DATASETS[0])
      }
    } else {
      if ([...EMBEDDING_DATASETS, ...RERANK_DATASETS].includes(dataset)) {
        setDataset(LLM_DATASETS[0])
      }
    }
  }, [api]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) {
      setBatchFile(f)
      setBatchInfo(null)
      setBatchError('')
    }
    // Reset so same file can be re-selected
    e.target.value = ''
  }

  const handleBatchUpload = async () => {
    if (!batchFile) return
    setBatchUploading(true)
    setBatchError('')
    try {
      const info = await uploadBatchCsv(batchFile)
      setBatchInfo(info)
    } catch (e) {
      setBatchError(String(e))
      setBatchInfo(null)
    } finally {
      setBatchUploading(false)
      // Reset file input so re-upload works for same file
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const [errors, setErrors] = useState<Record<string, string>>({})

  const buildSharedConfig = (): Record<string, unknown> => {
    const config: Record<string, unknown> = {
      parallel: parallel.replace(/，/g, ',').split(',').map((s) => Number(s.trim())).filter(Boolean),
      number: number.replace(/，/g, ',').split(',').map((s) => Number(s.trim())).filter(Boolean),
    }
    if (rate) config.rate = Number(rate)
    if (warmupRatio) config.warmup_num = Number(warmupRatio) / 100
    if (duration) config.duration = Number(duration)
    if (maxTokens) config.max_tokens = Number(maxTokens)
    if (minTokens) config.min_tokens = Number(minTokens)
    if (dataset) config.dataset = dataset === 'custom' ? 'local_jsonl' : dataset
    if (dataset === 'custom' && customDataset) config.dataset_label = customDataset
    if (datasetPath) config.dataset_path = datasetPath
    if (maxPromptLen) config.max_prompt_length = Number(maxPromptLen)
    if (minPromptLen) config.min_prompt_length = Number(minPromptLen)
    if (prefixLength) config.prefix_length = Number(prefixLength)
    if (thinkingMode !== 'auto') {
      config.extra_args = { enable_thinking: thinkingMode === 'on' }
    }
    if (extraArgs.trim()) {
      try { config.extra_args = { ...(config.extra_args as Record<string, unknown> || {}), ...JSON.parse(extraArgs) } }
      catch { /* handled in validation */ }
    }
    if (readTimeout) config.read_timeout = Number(readTimeout)
    if (slaEnabled) {
      config.sla_auto_tune = true
      config.sla_variable = slaVariable
      const groups = slaRules.filter((r) => r.metric && r.op && r.value.trim() !== '')
      if (groups.length > 0) {
        const params: Record<string, string> = {}
        for (const r of groups) params[r.metric] = `${r.op}${r.value.trim()}`
        config.sla_params = [params]
      }
      if (slaLower) config.sla_lower_bound = Number(slaLower)
      else config.sla_lower_bound = 1
      if (slaUpper) config.sla_upper_bound = Number(slaUpper)
      else config.sla_upper_bound = 64
      if (slaNumRuns) config.sla_num_runs = Number(slaNumRuns)
      else config.sla_num_runs = 1
    }
    return config
  }

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (isBatch) {
      // Batch mode: validate CSV uploaded, then submit via onBatchSubmit
      if (!batchInfo?.batch_id) {
        setBatchError('请先上传模型列表文件')
        return
      }
      const sharedConfig = buildSharedConfig()
      onBatchSubmit?.(batchInfo.batch_id, sharedConfig)
      return
    }

    // Single mode validation
    const newErrors: Record<string, string> = {}
    if (!model.trim()) newErrors.model = 'Required'
    if (!url.trim()) newErrors.url = 'Required'
    if (!apiKey.trim()) newErrors.apiKey = 'Required'

    // URL format
    if (url.trim()) {
      try {
        const u = new URL(url.trim())
        if (!['http:', 'https:'].includes(u.protocol)) {
          newErrors.url = 'URL 必须以 http:// 或 https:// 开头'
        }
      } catch {
        newErrors.url = 'URL 格式不正确'
      }
    }

    // Parallel & number: comma-separated positive integers
    const checkCommaSepPosInt = (val: string, key: string, label: string) => {
      if (val) {
        const parts = val.replace(/，/g, ',').split(',').map((s) => s.trim()).filter(Boolean)
        for (const p of parts) {
          const n = Number(p)
          if (!Number.isInteger(n) || n < 1) {
            newErrors[key] = `${label} 必须为正整数（逗号分隔）`
            break
          }
        }
      }
    }
    checkCommaSepPosInt(parallel, 'parallel', '并发数')
    checkCommaSepPosInt(number, 'number', '请求数')

    // Rate: positive number
    if (rate) {
      const r = Number(rate)
      if (isNaN(r) || r <= 0) newErrors.rate = '请求速率必须为正数'
    }

    // Warmup ratio: integer 1-99 (percent of total requests)
    if (warmupRatio) {
      const w = Number(warmupRatio)
      if (!Number.isInteger(w) || w < 1 || w > 99) newErrors.warmupRatio = '预热比例必须为 1-99 的整数'
    }

    // Duration budget: positive integer seconds
    if (duration) {
      const d = Number(duration)
      if (!Number.isInteger(d) || d < 1) newErrors.duration = '运行时长预算必须为正整数'
    }

    // SLA: at least one complete constraint when enabled
    if (slaEnabled) {
      const hasRule = slaRules.some((r) => r.metric && r.op && r.value.trim() !== '')
      if (!hasRule) newErrors.slaRules = t('perf.slaNoRules')
    }

    // Token / prompt length fields: positive integers
    const checkPosInt = (val: string, key: string, label: string) => {
      if (val) {
        const n = Number(val)
        if (!Number.isInteger(n) || n < 1) newErrors[key] = `${label} 必须为正整数`
      }
    }
    checkPosInt(maxTokens, 'maxTokens', '最大 Token 数')
    checkPosInt(minTokens, 'minTokens', '最小 Token 数')
    checkPosInt(maxPromptLen, 'maxPromptLen', '最大 Prompt 长度')
    checkPosInt(minPromptLen, 'minPromptLen', '最小 Prompt 长度')

    // prefixLength: non-negative integer
    if (prefixLength) {
      const n = Number(prefixLength)
      if (!Number.isInteger(n) || n < 0) newErrors.prefixLength = '前缀长度必须为非负整数'
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }
    setErrors({})

    const config = buildSharedConfig()

    config.model = model
    config.api = api
    config.url = url.trim()
    if (apiKey) config.api_key = apiKey

    if (extraArgs.trim()) {
      try { config.extra_args = { ...(config.extra_args as Record<string, unknown> || {}), ...JSON.parse(extraArgs) } }
      catch { newErrors.extra_args = t('perf.invalidJson') }
    }
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }
    onSubmit(config)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">

      {/* Test Mode Toggle */}
      <div className="flex items-center gap-6">
        <label className={`${FORM_LABEL_CLASS} !mb-0`}>测试模式</label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="tm" value="single" checked={!isBatch}
            onChange={() => { setTestMode('single'); onModeChange?.('single') }} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">单模型测试</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="tm" value="batch" checked={isBatch}
            onChange={() => { setTestMode('batch'); onModeChange?.('batch') }} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">批量测试</span>
        </label>
      </div>

      {/* ── Batch mode UI ── */}
      {isBatch && (
        <div className="space-y-3 p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-card2)]">
          <div className="flex items-center gap-3 flex-wrap">
            <a
              href={getTemplateDownloadUrl()}
              download="model_list_template.csv"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border border-[var(--accent-dim)] text-[var(--accent)] hover:bg-[var(--accent-dim)]/10 transition-colors"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              下载模板
            </a>

            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
            />
            <Button
              type="button"
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
            >
              选择文件
            </Button>

            {batchFile && (
              <Button
                type="button"
                variant="primary"
                onClick={handleBatchUpload}
                disabled={disabled || batchUploading}
              >
                {batchUploading ? '上传中...' : '上传文件'}
              </Button>
            )}
          </div>

          {batchFile && !batchInfo && !batchError && (
            <p className="text-xs text-[var(--text-muted)]">
              已选择: {batchFile.name} — 点击"上传文件"解析模型列表
            </p>
          )}

          {batchError && (
            <p className="text-xs text-[var(--danger)]">{batchError}</p>
          )}

          {batchInfo && (
            <div className="space-y-1">
              <p className="text-xs text-[var(--green)]">
                ✓ 上传成功，共 {batchInfo.model_count} 个模型
              </p>
              <div className="flex flex-wrap gap-1">
                {batchInfo.models.map((m) => (
                  <span key={m} className="px-2 py-0.5 text-xs rounded bg-[var(--bg)] border border-[var(--border)] text-[var(--text)]">{m}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Model Source (single mode only) — API only ── */}
      {!isBatch && (
      <div className="flex items-center gap-6">
        <label className={`${FORM_LABEL_CLASS} !mb-0`}>{t('eval.modelSource')}</label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="ms" value="openai" checked readOnly className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">API</span>
        </label>
      </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* ── OpenAI API fields ── */}
        {!isBatch && (<>
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
              <option value="anthropic">Anthropic</option>
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

          <FormField label={t('eval.apiKey')} required error={errors.apiKey}>
            <input type="password" value={apiKey} onChange={(e) => { setApiKey(e.target.value); if (errors.apiKey) setErrors((p) => ({ ...p, apiKey: '' })) }} className={inputClass(errors.apiKey)} placeholder="sk-..." />
          </FormField>
        </>)}

        {/* ── Dataset ── */}
        <FormField label={t('perf.dataset')}>
          <select value={dataset} onChange={(e) => setDataset(e.target.value)} className={FORM_INPUT_CLASS}>
            {(isEmbeddingOrRerank(api)
              ? (EMBEDDING_APIS.includes(api) ? EMBEDDING_DATASETS : RERANK_DATASETS)
              : LLM_DATASETS
            ).map((ds) => (
              <option key={ds} value={ds}>{ds === 'openqa' ? t('perf.datasetDefault', { name: ds }) : ds}</option>
            ))}
            <option value="custom">{t('perf.datasetCustom')}</option>
          </select>
        </FormField>

        <FormField label={t('perf.rate')} error={errors.rate}>
          <input type="number" value={rate}
            onChange={(e) => { setRate(e.target.value); if (errors.rate) setErrors((p) => ({ ...p, rate: '' })) }}
            className={inputClass(errors.rate)} placeholder={t('perf.placeholderReqPerSec')} />
        </FormField>

        {dataset === 'custom' && (
          <>
            <FormField label={t('perf.customDatasetName')}>
              <input value={customDataset} onChange={(e) => setCustomDataset(e.target.value)} className={FORM_INPUT_CLASS} placeholder={t('perf.customDatasetNamePh')} />
            </FormField>
            <FormField label={t('perf.customDatasetPath')}>
              <input value={datasetPath} onChange={(e) => setDatasetPath(e.target.value)} className={FORM_INPUT_CLASS} placeholder="/data/datasets/my_perf_data.jsonl" />
            </FormField>
          </>
        )}

        {/* ── 压测参数 ── */}
        <FormField label={t('perf.parallel')} error={errors.parallel}>
          <input value={parallel}
            onChange={(e) => { setParallel(e.target.value); if (errors.parallel) setErrors((p) => ({ ...p, parallel: '' })) }}
            className={inputClass(errors.parallel)} placeholder="1, 4, 8" />
        </FormField>

        <FormField label={t('perf.number')} error={errors.number}>
          <input value={number}
            onChange={(e) => { setNumber(e.target.value); if (errors.number) setErrors((p) => ({ ...p, number: '' })) }}
            className={inputClass(errors.number)} placeholder="10, 100" />
        </FormField>

        <FormField label={t('perf.warmupRatio')} error={errors.warmupRatio} hint={t('perf.warmupHint')}>
          <input type="number" value={warmupRatio}
            onChange={(e) => { setWarmupRatio(e.target.value.replace(/[^0-9]/g, '')); if (errors.warmupRatio) setErrors((p) => ({ ...p, warmupRatio: '' })) }}
            className={inputClass(errors.warmupRatio)} placeholder="10" />
        </FormField>

        <FormField label={t('perf.durationBudget')} error={errors.duration} hint={t('perf.durationHint')}>
          <input type="number" value={duration}
            onChange={(e) => { setDuration(e.target.value.replace(/[^0-9]/g, '')); if (errors.duration) setErrors((p) => ({ ...p, duration: '' })) }}
            className={inputClass(errors.duration)} placeholder={t('perf.placeholderNoLimit')} />
        </FormField>

        {/* ── Token / Prompt ── */}
        {!isEmbeddingOrRerank(api) && (<>
        <FormField label={t('perf.maxTokens')} error={errors.maxTokens}>
          <input type="number" value={maxTokens}
            onChange={(e) => { setMaxTokens(e.target.value.replace(/[^0-9]/g, '')); if (errors.maxTokens) setErrors((p) => ({ ...p, maxTokens: '' })) }}
            className={inputClass(errors.maxTokens)} placeholder={t('perf.placeholderDefaultVal', { v: '2048' })} />
        </FormField>

        <FormField label={t('perf.minTokens')} error={errors.minTokens}>
          <input type="number" value={minTokens}
            onChange={(e) => { setMinTokens(e.target.value.replace(/[^0-9]/g, '')); if (errors.minTokens) setErrors((p) => ({ ...p, minTokens: '' })) }}
            className={inputClass(errors.minTokens)} placeholder={t('perf.placeholderNoLimit')} />
        </FormField>
        </>)}

        <FormField label={t('perf.maxPromptLen')} error={errors.maxPromptLen}>
          <input type="number" value={maxPromptLen}
            onChange={(e) => { setMaxPromptLen(e.target.value.replace(/[^0-9]/g, '')); if (errors.maxPromptLen) setErrors((p) => ({ ...p, maxPromptLen: '' })) }}
            className={inputClass(errors.maxPromptLen)} placeholder={t('perf.placeholderDefaultVal', { v: '131072' })} />
        </FormField>

        <FormField label={t('perf.minPromptLen')} error={errors.minPromptLen}>
          <input type="number" value={minPromptLen}
            onChange={(e) => { setMinPromptLen(e.target.value.replace(/[^0-9]/g, '')); if (errors.minPromptLen) setErrors((p) => ({ ...p, minPromptLen: '' })) }}
            className={inputClass(errors.minPromptLen)} placeholder={t('perf.placeholderDefaultVal', { v: '0' })} />
        </FormField>

        <FormField label="请求超时（秒）" error={errors.readTimeout}>
          <input type="number" value={readTimeout}
            onChange={(e) => { setReadTimeout(e.target.value.replace(/[^0-9]/g, '')); if (errors.readTimeout) setErrors((p) => ({ ...p, readTimeout: '' })) }}
            className={inputClass(errors.readTimeout)} placeholder="默认 300" />
        </FormField>

      </div>

      {/* ── 高级选项 ── */}
      <Collapsible header={<span className="text-sm text-[var(--accent)]">{t('perf.moreParams')}</span>} defaultOpen={false} chevronAfter chevronColor="var(--accent)">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
          {!isEmbeddingOrRerank(api) && (
          <FormField label={t('perf.thinkingMode')}>
            <select value={thinkingMode} onChange={(e) => setThinkingMode(e.target.value)} className={FORM_INPUT_CLASS}>
              <option value="auto">{t('perf.thinkingModeAuto')}</option>
              <option value="on">{t('perf.thinkingModeOn')}</option>
              <option value="off">{t('perf.thinkingModeOff')}</option>
            </select>
          </FormField>
          )}

          <FormField label={t('perf.prefixLength')} error={errors.prefixLength}>
            <input type="number" value={prefixLength}
              onChange={(e) => { setPrefixLength(e.target.value.replace(/[^0-9]/g, '')); if (errors.prefixLength) setErrors((p) => ({ ...p, prefixLength: '' })) }}
              className={inputClass(errors.prefixLength)} placeholder="0" />
          </FormField>

          {/* ── SLA Auto-tuning ── */}
          <div className="md:col-span-2 rounded-lg border border-[var(--border)] bg-[var(--bg-card2)] p-3 space-y-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={slaEnabled} onChange={(e) => setSlaEnabled(e.target.checked)}
                className="accent-[var(--accent)]" />
              <span className="text-sm text-[var(--text)]">{t('perf.slaAutoTune')}</span>
            </label>
            {slaEnabled && (
              <>
                <div className="flex items-center gap-3">
                  <label className="text-xs text-[var(--text-muted)] shrink-0">{t('perf.slaVariable')}</label>
                  <select value={slaVariable} onChange={(e) => setSlaVariable(e.target.value as 'parallel' | 'rate')} className={FORM_INPUT_CLASS}>
                    <option value="parallel">{t('perf.slaConcurrency')}</option>
                    <option value="rate">{t('perf.slaRate')}</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <div className="text-xs text-[var(--text-muted)]">{t('perf.slaRules')}</div>
                  {slaRules.map((r, i) => (
                    <div key={i} className="flex items-center gap-1.5 flex-wrap">
                      <select value={r.metric} onChange={(e) => updateSlaRule(i, 'metric', e.target.value)} className={`${FORM_INPUT_CLASS} flex-1 min-w-[150px]`}>
                        {SLA_METRICS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
                      </select>
                      <select value={r.op} onChange={(e) => updateSlaRule(i, 'op', e.target.value)} className={`${FORM_INPUT_CLASS} !w-16 shrink-0 px-1 text-center`}>
                        <option value="<=">≤</option>
                        <option value=">=">≥</option>
                        <option value="<">&lt;</option>
                        <option value=">">&gt;</option>
                      </select>
                      <input type="number" step="any" value={r.value}
                        onChange={(e) => updateSlaRule(i, 'value', e.target.value)}
                        className={`${FORM_INPUT_CLASS} w-28 shrink-0`} placeholder="2.0" />
                      <button type="button" onClick={() => removeSlaRule(i)}
                        className="text-xs text-[var(--danger)] shrink-0 cursor-pointer">✕</button>
                    </div>
                  ))}
                  {errors.slaRules && <p className="text-xs text-[var(--danger)]">{errors.slaRules}</p>}
                  <button type="button" onClick={addSlaRule}
                    className="text-xs text-[var(--accent)] cursor-pointer">+ {t('perf.slaAddRule')}</button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <FormField label={t('perf.slaLower')}>
                    <input type="number" value={slaLower}
                      onChange={(e) => setSlaLower(e.target.value.replace(/[^0-9]/g, ''))}
                      className={FORM_INPUT_CLASS} placeholder="默认 1" />
                  </FormField>
                  <FormField label={t('perf.slaUpper')}>
                    <input type="number" value={slaUpper}
                      onChange={(e) => setSlaUpper(e.target.value.replace(/[^0-9]/g, ''))}
                      className={FORM_INPUT_CLASS} placeholder="默认 64" />
                  </FormField>
                  <FormField label={t('perf.slaNumRuns')}>
                    <input type="number" value={slaNumRuns}
                      onChange={(e) => setSlaNumRuns(e.target.value.replace(/[^0-9]/g, ''))}
                      className={FORM_INPUT_CLASS} placeholder="默认 1" />
                  </FormField>
                </div>
              </>
            )}
          </div>

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
        {isBatch ? '开始批量测试' : t('perf.startPerf')}
      </Button>
    </form>
  )
}
