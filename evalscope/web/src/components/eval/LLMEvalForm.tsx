import { useEffect, useRef, useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listBenchmarks, getEvalTemplateDownloadUrl } from '@/api/eval'
import { toast } from '@/components/common/Toast'
import Button from '@/components/ui/Button'
import Card from '@/components/ui/Card'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, FORM_LABEL_CLASS, inputClass } from '@/components/ui/formStyles'
import { ChevronDown, ChevronUp } from 'lucide-react'
import type { EvalTabContext } from '@/pages/EvalLayout'
import { TASK_STATUSES, isActiveTaskStatus, normalizeTaskStatus } from '@/hooks/taskLifecycle'

interface Props {
  context: EvalTabContext
}

const ALL_LOCAL_TYPES = ['general_qa', 'general_mcq', 'general_fc', 'general_vqa', 'general_vmcq', 'general_arena', 'general_t2i', 'data_collection']
const LOCAL_TYPE_LABEL: Record<string, string> = {
  general_qa: 'eval.datasetLocalTypeQA',
  general_mcq: 'eval.datasetLocalTypeMCQ',
  general_fc: 'eval.datasetLocalTypeFC',
  general_vqa: 'eval.datasetLocalTypeVQA',
  general_vmcq: 'eval.datasetLocalTypeVMCQ',
  general_arena: 'eval.datasetLocalTypeArena',
  general_t2i: 'eval.datasetLocalTypeT2I',
  data_collection: 'eval.datasetLocalTypeDataCollection',
}

export default function LLMEvalForm({ context }: Props) {
  const { t } = useLocale()
  const { onSubmit, disabled, initialDataset, onApiKeyChange, isBatch, batchRunning, batchState,
    batchInfo, batchError, batchUploading, selectedTaskId, onSelectTask,
    onBatchSubmit, onBatchResume, onBatchStop, onBatchUpload, setBatchMode } = context

  // Batch file state
  const fileInputRef = useRef<HTMLInputElement>(null)
  const resumeConfigRef = useRef<Record<string, unknown> | null>(null)
  const [batchFile, setBatchFile] = useState<File | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setBatchFile(file)
      if (resumeConfigRef.current) {
        const config = resumeConfigRef.current
        resumeConfigRef.current = null
        void onBatchResume(file, config)
      }
    }
    e.target.value = ''
  }

  const handleBatchUpload = async () => {
    if (!batchFile) return
    try {
      await onBatchUpload(batchFile)
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  // Model source — API only. Local (backend / model-path) evaluation is not
  // offered; the model is always reached through an OpenAI/Anthropic API.

  // OpenAI API
  const [model, setModel] = useState('')
  const [apiUrl, setApiUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [evalType, setEvalType] = useState('openai')

  // Notify parent of apiKey changes
  useEffect(() => {
    onApiKeyChange?.(apiKey)
  }, [apiKey, onApiKeyChange])

  // Dataset
  const [datasetHub, setDatasetHub] = useState('modelscope')
  const [datasets, setDatasets] = useState(initialDataset ?? '')
  const [datasetPath, setDatasetPath] = useState('')
  const [datasetDir, setDatasetDir] = useState('')
  const isLocalDataset = datasetHub === 'local'
  // Anthropic 的思考走 thinking.budget_tokens（后端从 generation_config.reasoning_tokens 映射），
  // 不认 OpenAI/Qwen 系的 extra_body.enable_thinking ⇒ 该协议下不下发这个键，并禁用「思考模式」下拉
  const isAnthropic = evalType === 'anthropic'

  // Common
  const [limit, setLimit] = useState('')
  const [randomSample, setRandomSample] = useState(false)
  const [evalBatchSize, setEvalBatchSize] = useState('1')
  const [showMore, setShowMore] = useState(false)
  const [repeats, setRepeats] = useState('1')
  const [timeout, setTimeout_] = useState('300')
  const [stream, setStream] = useState(false)
  const [useSandbox, setUseSandbox] = useState(false)
  const [sandboxDatasets, setSandboxDatasets] = useState<Set<string>>(new Set())
  const [temperature, setTemperature] = useState('')
  const [topP, setTopP] = useState('')
  const [maxTokens, setMaxTokens] = useState('')
  const [topK, setTopK] = useState('')
  const [thinkingMode, setThinkingMode] = useState('auto')
  const [seed, setSeed] = useState('42')
  const [judgeStrategy, setJudgeStrategy] = useState('auto')
  const [ignoreErrors, setIgnoreErrors] = useState(false)
  const [datasetArgs, setDatasetArgs] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('')

  // Judge model
  const [judgeModel, setJudgeModel] = useState('')
  const [judgeApiUrl, setJudgeApiUrl] = useState('')
  const [judgeApiKey, setJudgeApiKey] = useState('')

  // Validation
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Dataset autocomplete
  const [benchmarkNames, setBenchmarkNames] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [filteredSuggestions, setFilteredSuggestions] = useState<string[]>([])
  const datasetInputRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (initialDataset) {
      // Intentionally mirror a changing route/query parameter into form state.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDatasets(initialDataset)
    }
  }, [initialDataset])

  useEffect(() => {
    // Only names, tags and meta.sandbox_config are consumed here, so skip
    // README bodies entirely — this form must not pay for the ~89% of the
    // benchmark payload that descriptions account for.
    listBenchmarks(undefined, true, 'none')
      .then((res) => {
        const all = [...(res.text ?? []), ...(res.multimodal ?? [])]
        const names: string[] = []
        const needsSandbox = new Set<string>()
        for (const b of all) {
          if (!b.tags?.includes('Custom') && b.name !== 'data_collection') {
            names.push(b.name)
            const sc = b.meta?.sandbox_config
            if (sc && Object.keys(sc).length > 0) needsSandbox.add(b.name)
          }
        }
        setBenchmarkNames(names)
        setSandboxDatasets(needsSandbox)
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
      const allPool = [...benchmarkNames, ...ALL_LOCAL_TYPES]
      const matches = allPool.filter((n) => n.toLowerCase().includes(current))
      setFilteredSuggestions(matches.slice(0, 8))
      setShowSuggestions(matches.length > 0)
    } else {
      setShowSuggestions(false)
    }
    if (errors.datasets) setErrors((prev) => ({ ...prev, datasets: '' }))
  }

  // Auto-toggle sandbox when datasets change (covers both typing and suggestion clicks)
  useEffect(() => {
    const selected = datasets.split(/[,，]/).map(s => s.trim()).filter(Boolean)
    const needsSandbox = selected.some(ds => sandboxDatasets.has(ds))
    if (needsSandbox && !useSandbox) {
      // Intentionally derive the sandbox toggle from selected datasets.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setUseSandbox(true)
    } else if (!needsSandbox && useSandbox) {
      setUseSandbox(false)
    }
  }, [datasets, sandboxDatasets, useSandbox])

  const selectSuggestion = (name: string) => {
    if (ALL_LOCAL_TYPES.includes(name)) {
      setDatasetHub('local')
    }
    const parts = datasets.split(/[,，]/).map((s) => s.trim())
    parts[parts.length - 1] = name
    setDatasets(parts.join(', '))
    setShowSuggestions(false)
  }

  const suggestionLabel = (name: string) => {
    if (LOCAL_TYPE_LABEL[name]) return `${name}（${t(LOCAL_TYPE_LABEL[name])}）`
    return name
  }

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()

    // Warn if sandbox-required dataset selected without sandbox
    const selected = datasets.split(',').map(s => s.trim()).filter(Boolean)
    const needsSandbox = selected.some(ds => sandboxDatasets.has(ds))
    if (needsSandbox && !useSandbox) {
      toast.warning('所选数据集需要 Docker 沙箱环境，测试结果可能不准确')
    }

    // Batch mode: delegate to onBatchSubmit
    if (isBatch) {
      const resumable = batchState?.status === 'stopped' && batchState.resumable
      if (!resumable && !batchInfo?.batch_id) { toast.error('请先上传模型列表文件'); return }
      const dsList = datasets.split(/[,，]/).map((s) => s.trim()).filter(Boolean)
      if (!dsList.length) { toast.error('请先填写测试数据集'); return }
      // 本地数据集：批量路径此前不带 hub / path（shared 与后端合并白名单里都没有这两个键），
      // 会被静默丢弃并按默认 hub 去拉数据 —— 这里与单模型路径对齐。
      if (isLocalDataset && !datasetPath.trim()) { toast.error('本地数据集需要填写数据集路径'); return }
      const dsArgs: Record<string, unknown> = {}
      if (isLocalDataset) {
        for (const ds of dsList) dsArgs[ds] = { local_path: datasetPath.trim() }
      }
      if (datasetArgs.trim()) {
        // 此前这里把 JSON 文本原样当字符串发出，后端 TaskConfig.dataset_args 需要的是对象
        try { Object.assign(dsArgs, JSON.parse(datasetArgs) as Record<string, unknown>) }
        catch { toast.error('数据集参数 JSON 格式不正确'); return }
      }
      // 生成参数 / 裁判模型 / 系统提示词必须用单模型那套「嵌套」形状，否则后端静默忽略：
      //  ・TaskConfig 没有顶层 temperature/top_p/max_tokens/top_k ⇒ 平铺发送无效（批量采样参数此前就是这么失效的）
      //  ・裁判只认 judge_model_args（没有顶层 judge_model/judge_api_url/judge_api_key）
      //  ・system_prompt 也不是 TaskConfig 字段，单模型把它注入到每个数据集的参数里
      const genConfig: Record<string, unknown> = {}
      if (temperature) genConfig.temperature = Number(temperature)
      if (topP) genConfig.top_p = Number(topP)
      if (maxTokens) genConfig.max_tokens = Number(maxTokens)
      if (topK) genConfig.top_k = Number(topK)
      // 与单模型一致：Anthropic 不下发 enable_thinking
      if (thinkingMode !== 'auto' && !isAnthropic) {
        genConfig.extra_body = { ...(genConfig.extra_body as Record<string, unknown> || {}), enable_thinking: thinkingMode === 'on' }
      }
      if (systemPrompt.trim()) {
        for (const ds of dsList) {
          const entry = (dsArgs[ds] as Record<string, unknown>) || {}
          dsArgs[ds] = { system_prompt: systemPrompt.trim(), ...entry }
        }
      }
      const judgeArgs: Record<string, unknown> = {}
      if (judgeModel.trim() || judgeApiUrl.trim() || judgeApiKey.trim()) {
        if (judgeModel.trim()) judgeArgs.model_id = judgeModel.trim()
        if (judgeApiUrl.trim()) judgeArgs.api_url = judgeApiUrl.trim()
        // 与后端 _build_task_config_openai 的防御一致：部分配置时用主 key 兜底，避免裁判 401
        judgeArgs.api_key = judgeApiKey.trim() || apiKey || ''
        judgeArgs.eval_type = isAnthropic ? 'anthropic_api' : 'openai_api'
      }
      const shared: Record<string, unknown> = {
        eval_backend: context.evalMode === 'rag' ? 'RAGEval' : context.evalMode === 'aigc' ? 'AIGCEval' : context.evalMode === 'audio' ? 'AudioEval' : '',
        datasets: dsList,
        dataset_hub: isLocalDataset ? 'local' : datasetHub,
        dataset_dir: datasetDir || undefined,
        random_sample: limit && randomSample ? true : undefined,
        limit: limit ? Number(limit) : undefined,
        eval_batch_size: evalBatchSize ? Number(evalBatchSize) : 1,
        repeats: repeats ? Number(repeats) : 1,
        timeout: timeout ? Number(timeout) : 300,
        stream,
        seed: seed || undefined,
        judge_strategy: judgeStrategy,
        ignore_errors: ignoreErrors,
        use_sandbox: useSandbox,
        dataset_args: Object.keys(dsArgs).length ? dsArgs : undefined,
        generation_config: Object.keys(genConfig).length ? genConfig : undefined,
        judge_model_args: Object.keys(judgeArgs).length ? judgeArgs : undefined,
      }
      if (batchState?.status === 'stopped' && batchState.resumable) {
        if (batchFile) void onBatchResume(batchFile, shared)
        else {
          resumeConfigRef.current = shared
          fileInputRef.current?.click()
        }
      } else {
        if (!batchInfo?.batch_id) { toast.error('请先上传模型列表文件'); return }
        onBatchSubmit(batchInfo.batch_id, shared)
      }
      return
    }

    const newErrors: Record<string, string> = {}

    if (!model.trim()) newErrors.model = 'Required'
    if (!apiUrl.trim()) newErrors.apiUrl = 'Required'
    if (!apiKey.trim()) newErrors.apiKey = 'Required'
    if (isLocalDataset) {
      if (!datasetPath.trim()) newErrors.datasetPath = 'Required'
      if (!datasets.trim()) newErrors.datasets = 'Required'
    } else {
      if (!datasets.trim()) newErrors.datasets = 'Required'
    }

    // URL format
    if (apiUrl.trim()) {
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

    if (datasetArgs) {
      try { JSON.parse(datasetArgs) } catch { newErrors.datasetArgs = 'JSON 格式不正确' }
    }

    if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return }
    setErrors({})

    const config: Record<string, unknown> = {
      model_source: 'openai',
      model,
      limit: limit ? Number(limit) : undefined,
      random_sample: limit && randomSample ? true : undefined,
      eval_batch_size: evalBatchSize ? Number(evalBatchSize) : undefined,
    }

    // Model (API)
    if (apiUrl) config.api_url = apiUrl.trim()
    if (apiKey) config.api_key = apiKey
    config.eval_type = evalType === 'anthropic' ? 'anthropic_api' : 'openai_api'

    // Datasets
    if (isLocalDataset) {
      config.datasets = datasets.split(/[,，]/).map((s) => s.trim()).filter(Boolean)
      config.dataset_hub = 'local'
      if (datasetDir) config.dataset_dir = datasetDir
      const args: Record<string, unknown> = {}
      for (const ds of config.datasets as string[]) {
        args[ds] = { local_path: datasetPath }
      }
      config.dataset_args = args
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
    // Thinking mode：enable_thinking 是 OpenAI/Qwen 系参数，Anthropic 不认（其 thinking 走 reasoning_tokens），
    // 该协议下不发，避免未知字段被塞进请求体
    if (thinkingMode !== 'auto' && !isAnthropic) {
      genConfig.extra_body = { ...(genConfig.extra_body || {}), enable_thinking: thinkingMode === 'on' }
      config.generation_config = genConfig
    }
    if (seed && seed !== '42') config.seed = Number(seed)
    if (judgeStrategy && judgeStrategy !== 'auto') config.judge_strategy = judgeStrategy
    if (ignoreErrors) config.ignore_errors = true
    if (useSandbox) config.use_sandbox = true
    if (datasetArgs) { try { const extra = JSON.parse(datasetArgs); config.dataset_args = { ...(config.dataset_args as Record<string, unknown> || {}), ...extra } } catch { /* ignore */ } }
    if (systemPrompt.trim()) {
      const da = (config.dataset_args || {}) as Record<string, unknown>
      for (const key of Object.keys(da)) {
        const entry = da[key] as Record<string, unknown>
        da[key] = { system_prompt: systemPrompt.trim(), ...entry }
      }
      config.dataset_args = da
    }

    // Judge model args
    if (judgeModel.trim() || judgeApiUrl.trim() || judgeApiKey.trim()) {
      const jma: Record<string, unknown> = {}
      if (judgeModel.trim()) jma.model_id = judgeModel.trim()
      if (judgeApiUrl.trim()) jma.api_url = judgeApiUrl.trim()
      // Fall back to the main API key so a partial judge config never sends a blank key.
      jma.api_key = judgeApiKey.trim() || apiKey || ''
      jma.eval_type = evalType === 'anthropic' ? 'anthropic_api' : 'openai_api'
      config.judge_model_args = jma
    }

    onSubmit(config)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* ── 测试模式切换 ── */}
      <div className="flex items-center gap-6">
        <label className={`${FORM_LABEL_CLASS} !mb-0`}>测试模式</label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="batch" checked={!isBatch}
            onChange={() => setBatchMode(false)} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">单模型测试</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="batch" checked={isBatch}
            onChange={() => setBatchMode(true)} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">批量测试</span>
        </label>
      </div>

      {/* ── Batch CSV upload ── */}
      {isBatch && (
        <div className="space-y-3 p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-card2)]">
          <div className="flex items-center gap-3 flex-wrap">
            <a href={getEvalTemplateDownloadUrl()} download="eval_model_list_template.csv"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border border-[var(--accent-dim)] text-[var(--accent)] hover:bg-[var(--accent-dim)]/10 transition-colors">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              下载模板
            </a>
            <input ref={fileInputRef} type="file" accept=".csv" onChange={handleFileChange} className="hidden" />
            <Button type="button" variant="outline" onClick={() => fileInputRef.current?.click()} disabled={disabled || batchRunning}>
              选择文件
            </Button>
            {batchFile && (
              <Button type="button" variant="primary" onClick={handleBatchUpload} disabled={disabled || batchUploading || batchRunning}>
                {batchUploading ? '上传中...' : '上传文件'}
              </Button>
            )}
          </div>
          {batchError && <p className="text-xs text-[var(--danger)]">{batchError}</p>}
          {batchInfo && (
            <p className="text-xs text-[var(--green)]">✓ 上传成功，共 {batchInfo.model_count} 个模型：{batchInfo.models.join(', ')}</p>
          )}
        </div>
      )}

      {/* Model Source — API only.  Local (backend / model-path) evaluation is not
        offered, so the model is always reached through an OpenAI/Anthropic API. */}
      {!isBatch && (<>
      <div className="flex items-center gap-6">
        <label className={`${FORM_LABEL_CLASS} !mb-0`}>{t('eval.modelSource')}</label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="ms" value="openai" checked readOnly className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">{t('eval.modelSourceOpenAI')}</span>
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* OpenAI API fields */}
        <>
          <FormField label={t('eval.modelName')} required error={errors.model}>
            <input value={model}
              onChange={(e) => { setModel(e.target.value.trimStart()); if (errors.model) setErrors((p) => ({ ...p, model: '' })) }}
              className={inputClass(errors.model)} placeholder="Qwen/Qwen2.5-0.5B-Instruct" />
          </FormField>
          <FormField label={t('eval.evalType')}>
            <select value={evalType} onChange={(e) => setEvalType(e.target.value)} className={FORM_INPUT_CLASS}>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
            </select>
          </FormField>
          <FormField label={t('eval.apiUrl')} required error={errors.apiUrl}>
            <input value={apiUrl}
              onChange={(e) => { setApiUrl(e.target.value); if (errors.apiUrl) setErrors((p) => ({ ...p, apiUrl: '' })) }}
              className={inputClass(errors.apiUrl)} placeholder={evalType === 'anthropic' ? 'https://api.anthropic.com' : 'http://localhost:8000/v1'} />
          </FormField>
          <FormField label={t('eval.apiKey')} required error={errors.apiKey}>
            <input type="password" value={apiKey}
              onChange={(e) => { setApiKey(e.target.value); if (errors.apiKey) setErrors((p) => ({ ...p, apiKey: '' })) }}
              className={inputClass(errors.apiKey)} placeholder="sk-..." />
          </FormField>
        </>

      </div>
      </>)}

      {/* Dataset Source */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FormField label={t('eval.datasetHub')}>
          <select value={datasetHub} onChange={(e) => setDatasetHub(e.target.value)} className={FORM_INPUT_CLASS}>
            <option value="modelscope">{t('eval.datasetHubModelScope')}</option>
            <option value="huggingface">{t('eval.datasetHubHuggingFace')}</option>
            <option value="local">{t('eval.datasetHubLocal')}</option>
          </select>
        </FormField>

        {isLocalDataset ? (<>
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
                      {suggestionLabel(name)}
                    </button>
                  ))}
                </div>
              )}
            </div>
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
                      {suggestionLabel(name)}
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
          <div className="flex items-center gap-3">
            <input type="number" value={limit} onChange={(e) => { setLimit(e.target.value.replace(/[^0-9]/g, '')); if (errors.limit) setErrors((p) => ({ ...p, limit: '' })) }} className={inputClass(errors.limit) + ' flex-1'} placeholder={t('common.placeholderAllData')} />
            <label className="flex items-center gap-1.5 text-sm text-[var(--text)] whitespace-nowrap cursor-pointer">
              <input type="checkbox" checked={randomSample}
                onChange={e => setRandomSample(e.target.checked)}
                className="accent-[var(--accent)]" />
              随机抽取
            </label>
          </div>
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
            {/* Thinking mode */}
            <div className="md:col-span-3 border-t border-[var(--border-md)] pt-3">
              <div className="flex items-center gap-4">
                <FormField label={t('eval.thinkingMode')} hint={isAnthropic ? t('eval.thinkingModeAnthropicHint') : undefined}>
                  <select value={thinkingMode} disabled={isAnthropic} onChange={(e) => setThinkingMode(e.target.value)}
                    className={`${FORM_INPUT_CLASS} ${isAnthropic ? 'opacity-50 cursor-not-allowed' : ''}`}>
                    <option value="auto">{t('eval.thinkingModeAuto')}</option>
                    <option value="on">{t('eval.thinkingModeOn')}</option>
                    <option value="off">{t('eval.thinkingModeOff')}</option>
                  </select>
                </FormField>
              </div>
            </div>
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
              <label className="flex items-center gap-1.5 text-sm text-[var(--text-muted)] cursor-pointer">
                <input type="checkbox" checked={useSandbox} onChange={(e) => setUseSandbox(e.target.checked)} className="accent-[var(--accent)]" />
                Docker 沙箱
              </label>
            </div>
            {/* Row 4 — 系统提示 + 数据集参数 */}
            <div className="md:col-span-3">
              <label className={FORM_LABEL_CLASS}>System Prompt</label>
              <textarea value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                className={`${FORM_INPUT_CLASS} h-16 resize-y`}
                placeholder="只输出一个词作为答案，禁止任何解释。" />
              <p className="mt-1 text-xs text-[var(--text-muted)]">提示词模板，会注入到所有数据集中。留空则使用数据集默认值。</p>
            </div>
            <div className="md:col-span-3">
              <label className={FORM_LABEL_CLASS}>{t('eval.datasetArgs')}</label>
              <textarea value={datasetArgs}
                onChange={(e) => { setDatasetArgs(e.target.value); if (errors.datasetArgs) setErrors((p) => ({ ...p, datasetArgs: '' })) }}
                className={`${inputClass(errors.datasetArgs)} h-20 resize-y`} style={{ fontFamily: 'var(--font-mono)' }}
                placeholder='{"gsm8k": {"few_shot_num": 4}}' />
              {errors.datasetArgs && <p className="mt-1 text-xs text-red-500">{errors.datasetArgs}</p>}
            </div>
            {/* Row 5 — 评判模型 */}
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

      <Button type="submit" variant="primary" disabled={disabled || batchUploading} className="btn-glow">
        {isBatch
          ? (batchState?.status === 'stopped' && batchState.resumable ? '继续批量评估' : '开始批量评估')
          : t('eval.startEval')}
      </Button>

      {/* ── Batch progress (during running) ── */}
      {batchRunning && batchState && batchState.status === 'running' && (
        <div className="p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-card2)]">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium">批量评估中：{batchState.completed}/{batchState.total}</h3>
            <button type="button" onClick={onBatchStop}
              className="px-2 py-1 text-xs rounded border border-[var(--danger)] text-[var(--danger)] hover:bg-[var(--danger)]/10">停止</button>
          </div>
          <div className="w-full bg-[var(--bg)] rounded-full h-2 mb-2">
            <div className="bg-[var(--accent)] h-2 rounded-full transition-all" style={{ width: `${batchState.total > 0 ? (batchState.completed / batchState.total) * 100 : 0}%` }} />
          </div>
          {batchState.current_model && <p className="text-xs text-[var(--text-muted)]">当前: {batchState.current_model}</p>}
          {batchState.errors > 0 && <p className="text-xs text-[var(--danger)] mt-1">{batchState.errors} 个失败</p>}
        </div>
      )}

      {/* ── Batch result (after completion) ── */}
      {batchState && !isActiveTaskStatus(batchState.status) && (
        <div className="p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-card2)]">
          <h3 className="text-sm font-medium mb-2">
            {batchState.status === TASK_STATUSES.COMPLETED
              ? '批量评估完成'
              : batchState.status === TASK_STATUSES.PARTIAL_SUCCESS
                ? '批量评估部分完成'
                : batchState.status === TASK_STATUSES.FAILED
                  ? '批量评估失败'
                  : '批量评估已停止'}：
            {batchState.completed} 成功
            {batchState.errors > 0 && <span className="text-[var(--danger)]">，{batchState.errors} 失败</span>}
          </h3>
          <p className="text-xs text-[var(--text-muted)] mb-2">点击模型查看对应日志</p>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {batchState.results.map((r) => (
              <div key={r.task_id}
                onClick={() => onSelectTask(r.task_id)}
                className={`flex items-center gap-2 text-xs rounded px-1.5 py-0.5 -mx-1.5 cursor-pointer transition-colors ${
                  r.task_id === selectedTaskId ? 'bg-[var(--accent)]/10 ring-1 ring-[var(--accent-dim)]' : 'hover:bg-[var(--bg)]'
                }`}>
                {normalizeTaskStatus(r.status) === TASK_STATUSES.FAILED ? (
                  <span className="text-[var(--danger)]">✗</span>
                ) : (
                  <span className="text-[var(--green)]">✓</span>
                )}
                <span className="text-[var(--text)]">{r.name}</span>
                <span className="text-[var(--text-muted)]">({r.model})</span>
                {normalizeTaskStatus(r.status) === TASK_STATUSES.FAILED && r.error && (
                  <span className="text-[var(--danger)] truncate max-w-48" title={r.error}>{r.error}</span>
                )}
                <span className="text-[var(--text-dim)] ml-auto">{r.task_id}</span>
              </div>
            ))}
            {batchState.error_details.filter((e) => !batchState.results.some((r2) => r2.name === e.name)).map((e, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="text-[var(--danger)]">✗</span>
                <span className="text-[var(--text)]">{e.name}</span>
                <span className="text-[var(--text-muted)]">({e.model})</span>
                <span className="text-[var(--danger)] ml-auto">{e.error}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </form>
  )
}
