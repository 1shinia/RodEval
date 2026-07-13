import { useEffect, useRef, useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, FORM_LABEL_CLASS, inputClass } from '@/components/ui/formStyles'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
  evalMode: 'embedding' | 'reranker'
}

// Text task types for embedding models
const EMBEDDING_TASK_TYPES = [
  'Classification', 'Clustering', 'Retrieval', 'STS',
  'PairClassification', 'BitextMining', 'Summarization',
  'MultilabelClassification', 'InstructionRetrieval', 'ZeroShotClassification',
]

// Task types for reranker
const RERANKER_TASK_TYPES = ['Reranking', 'InstructionReranking']

// Mainstream datasets per task type (popular, well-known ones)
const TYPE_DATASETS: Record<string, string[]> = {
  Classification: [
    'AmazonReviewsClassification', 'AmazonCounterfactualClassification',
    'Banking77Classification', 'EmotionClassification', 'ImdbClassification',
    'MassiveIntentClassification', 'MassiveScenarioClassification',
    'MTOPIntentClassification', 'TweetSentimentClassification',
    'ToxicConversationsClassification',
  ],
  Clustering: [
    'ArxivClusteringP2P', 'ArxivClusteringS2S', 'BiorxivClusteringP2P',
    'BiorxivClusteringS2S', 'MedrxivClusteringP2P', 'MedrxivClusteringS2S',
    'RedditClustering', 'RedditClusteringP2P', 'StackExchangeClusteringP2P',
    'TwentyNewsgroupsClustering',
  ],
  Retrieval: [
    'ArguAna', 'ClimateFEVER', 'CQADupstackRetrieval', 'DBPedia',
    'FEVER', 'FiQA2018', 'HotpotQA', 'MSMARCO', 'NFCorpus', 'NQ',
    'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'TRECCOVID', 'Touche2020',
  ],
  STS: [
    'BIOSSES', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17',
    'STS22', 'STSBenchmark', 'SICK-R',
  ],
  PairClassification: [
    'SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus',
  ],
  BitextMining: [
    'BUCC', 'Tatoeba',
  ],
  Summarization: [
    'SummEval',
  ],
  MultilabelClassification: [
    'MultiHateClassification', 'TweetTopicClassification',
  ],
  InstructionRetrieval: [
    'Core17InstructionRetrieval', 'News21InstructionRetrieval',
  ],
  ZeroShotClassification: [
    'AmazonCounterfactualClassification', 'ToxicConversationsClassification',
  ],
  Reranking: [
    'AskUbuntuDupQuestions', 'MindSmallReranking',
    'SciDocsRR', 'StackOverflowDupQuestions',
    'MMarcoReranking', 'T2Reranking', 'WikipediaRerankingMultilingual',
    'CMedQAv1-reranking', 'CMedQAv2-reranking',
  ],
  InstructionReranking: [
    'Core17InstructionRetrieval', 'News21InstructionRetrieval',
    'Robust04InstructionRetrieval', 'mFollowIR', 'mFollowIRCrossLingual',
  ],
}

export default function RAGEvalForm({ onSubmit, disabled, evalMode }: Props) {
  const { t } = useLocale()

  const [ragModelSource, setRagModelSource] = useState<'api' | 'local'>('api')
  const [ragModelPath, setRagModelPath] = useState('')
  const [ragApiBase, setRagApiBase] = useState('')
  const [ragApiKey, setRagApiKey] = useState('')
  const [ragTaskTypes, setRagTaskTypes] = useState('')
  const [ragTaskNames, setRagTaskNames] = useState('')
  const [ragLanguages, setRagLanguages] = useState('')
  const [ragLimit, setRagLimit] = useState('')
  const [ragDimension, setRagDimension] = useState('')
  const [ragMaxSeqLen, setRagMaxSeqLen] = useState('')
  const [ragBatchSize, setRagBatchSize] = useState('')
  const [ragTopK, setRagTopK] = useState('')
  const [ragPooling, setRagPooling] = useState('')
  const [ragTwoStage, setRagTwoStage] = useState(false)
  const [ragEncoderModel, setRagEncoderModel] = useState('')
  const [ragPrompt, setRagPrompt] = useState('')
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Auto-complete for task names
  const [showNameSuggestions, setShowNameSuggestions] = useState(false)
  const [filteredNameSuggestions, setFilteredNameSuggestions] = useState<string[]>([])
  const nameInputRef = useRef<HTMLDivElement>(null)

  // Compute candidate datasets from checked task types
  const checkedTypes = ragTaskTypes.split(/[,，]/).map(s => s.trim()).filter(Boolean)
  const candidateDatasets = checkedTypes.length > 0
    ? [...new Set(checkedTypes.flatMap(tt => TYPE_DATASETS[tt] || []))]
    : Object.values(TYPE_DATASETS).flat()

  // Click outside to close suggestions
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (nameInputRef.current && !nameInputRef.current.contains(e.target as Node)) {
        setShowNameSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleTaskNameChange = (val: string) => {
    setRagTaskNames(val)
    const parts = val.split(/[,，]/)
    const current = parts[parts.length - 1].trim().toLowerCase()
    if (current) {
      const matches = candidateDatasets.filter(n => n.toLowerCase().includes(current))
      setFilteredNameSuggestions(matches.slice(0, 8))
      setShowNameSuggestions(matches.length > 0)
    } else {
      setShowNameSuggestions(false)
    }
  }

  const selectNameSuggestion = (name: string) => {
    const parts = ragTaskNames.split(/[,，]/).map(s => s.trim())
    parts[parts.length - 1] = name
    setRagTaskNames(parts.join(', '))
    setShowNameSuggestions(false)
  }

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}
    const isApi = ragModelSource === 'api'

    if (isApi) {
      if (!ragModelPath.trim()) newErrors.ragModelPath = 'Required'
      if (!ragApiBase.trim()) newErrors.ragApiBase = 'Required'
      if (!ragApiKey.trim()) newErrors.ragApiKey = 'Required'
    } else {
      if (!ragModelPath.trim()) newErrors.ragModelPath = 'Required'
    }

    if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return }
    setErrors({})

    const modelConfig: Record<string, unknown> = {
      is_cross_encoder: evalMode === 'reranker',
    }
    if (ragPrompt.trim()) modelConfig.prompt = ragPrompt.trim()
    if (isApi) {
      modelConfig.model_name = ragModelPath.trim()
      modelConfig.model_name_or_path = ragModelPath.trim()
      modelConfig.api_base = ragApiBase.trim()
      if (ragApiKey) modelConfig.api_key = ragApiKey
      if (ragDimension) modelConfig.dimensions = Number(ragDimension)
    } else {
      modelConfig.model_name_or_path = ragModelPath.trim()
      if (ragMaxSeqLen) modelConfig.max_seq_length = Number(ragMaxSeqLen)
      if (ragBatchSize) modelConfig.encode_kwargs = { batch_size: Number(ragBatchSize) }
      if (evalMode === 'embedding' && ragPooling) modelConfig.pooling_mode = ragPooling
    }

    const models: Record<string, unknown>[] = [modelConfig]
    if (ragTwoStage && ragEncoderModel.trim()) {
      models.unshift({
        model_name_or_path: ragEncoderModel.trim(),
        is_cross_encoder: false,
      })
    }

    const evalCfg: Record<string, unknown> = {}
    if (ragTaskTypes) evalCfg.task_types = ragTaskTypes.split(/[,，]/).map((s: string) => s.trim()).filter(Boolean)
    if (ragTaskNames) evalCfg.task_names = ragTaskNames.split(/[,，]/).map((s: string) => s.trim()).filter(Boolean)
    if (ragLanguages) evalCfg.languages = ragLanguages.split(/[,，]/).map((s: string) => s.trim()).filter(Boolean)
    if (ragLimit) evalCfg.limits = Number(ragLimit)
    if (ragTopK) evalCfg.top_k = Number(ragTopK)

    onSubmit({
      eval_backend: 'rag_eval',
      eval_config: { tool: 'mteb', models, eval: evalCfg },
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Model Source — API / Local */}
      <div className="flex items-center gap-6">
        <label className={`${FORM_LABEL_CLASS} !mb-0`}>{t('eval.modelSource')}</label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="rms" value="api" checked={ragModelSource === 'api'}
            onChange={() => setRagModelSource('api')} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">API</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="rms" value="local" checked={ragModelSource === 'local'}
            onChange={() => setRagModelSource('local')} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">{t('eval.modelSourceLocal')}</span>
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* ── API fields ── */}
        {ragModelSource === 'api' && (<>
          <FormField label={t('eval.modelName')} required error={errors.ragModelPath}>
            <input value={ragModelPath}
              onChange={(e) => { setRagModelPath(e.target.value); if (errors.ragModelPath) setErrors(p => ({ ...p, ragModelPath: '' })) }}
              className={inputClass(errors.ragModelPath)}
              placeholder="text-embedding-3-small" />
          </FormField>
          <FormField label={t('eval.apiUrl')} required error={errors.ragApiBase}>
            <input value={ragApiBase}
              onChange={(e) => { setRagApiBase(e.target.value); if (errors.ragApiBase) setErrors(p => ({ ...p, ragApiBase: '' })) }}
              className={inputClass(errors.ragApiBase)} placeholder="https://api.openai.com/v1" />
          </FormField>
          <FormField label={t('eval.apiKey')} required error={errors.ragApiKey}>
            <input type="password" value={ragApiKey}
              onChange={(e) => { setRagApiKey(e.target.value); if (errors.ragApiKey) setErrors(p => ({ ...p, ragApiKey: '' })) }}
              className={FORM_INPUT_CLASS} placeholder="sk-..." />
          </FormField>
          {evalMode === 'embedding' && (
            <FormField label={t('eval.ragDimension')}>
              <input type="number" value={ragDimension}
                onChange={(e) => setRagDimension(e.target.value.replace(/[^0-9]/g, ''))}
                className={FORM_INPUT_CLASS} placeholder="1024" />
            </FormField>
          )}
        </>)}

        {/* ── Local fields ── */}
        {ragModelSource === 'local' && (<>
          <FormField label={t('eval.modelPath')} required error={errors.ragModelPath}>
            <input value={ragModelPath}
              onChange={(e) => { setRagModelPath(e.target.value); if (errors.ragModelPath) setErrors(p => ({ ...p, ragModelPath: '' })) }}
              className={inputClass(errors.ragModelPath)}
              placeholder="BAAI/bge-large-zh-v1.5" />
          </FormField>

          <FormField label={t('eval.ragMaxSeqLen')}>
            <input type="number" value={ragMaxSeqLen}
              onChange={(e) => setRagMaxSeqLen(e.target.value.replace(/[^0-9]/g, ''))}
              className={FORM_INPUT_CLASS} placeholder="512" />
          </FormField>

          <FormField label={t('eval.ragBatchSize')}>
            <input type="number" value={ragBatchSize}
              onChange={(e) => setRagBatchSize(e.target.value.replace(/[^0-9]/g, ''))}
              className={FORM_INPUT_CLASS} placeholder="32" />
          </FormField>

          {evalMode === 'embedding' && (
            <FormField label={t('eval.ragPoolingMode')}>
              <select value={ragPooling} onChange={(e) => setRagPooling(e.target.value)} className={FORM_INPUT_CLASS}>
                <option value="">auto</option>
                <option value="mean">mean</option>
                <option value="cls">cls</option>
              </select>
            </FormField>
          )}
        </>)}

        {/* Instruction Prefix — shared for API and Local */}
        <div className="md:col-span-2 border-t border-[var(--border-md)] pt-3"></div>
        <div className="md:col-span-2">
          <label className={FORM_LABEL_CLASS}>{t('eval.ragPrompt')}</label>
          <p className="text-xs text-[var(--text-muted)] mb-1.5">{t('eval.ragPromptHint')}</p>
          <input value={ragPrompt}
            onChange={(e) => setRagPrompt(e.target.value)}
            className={FORM_INPUT_CLASS}
            placeholder={t('eval.ragPromptHint')} />
        </div>

        {/* ── 评估数据集 ── */}
        <div className="md:col-span-2 border-t border-[var(--border-md)] pt-3"></div>

        <div className="md:col-span-2">
          <label className={FORM_LABEL_CLASS}>{t('eval.ragTaskTypesTitle')}</label>
          <p className="text-xs text-[var(--text-muted)] mb-2">{t('eval.ragTaskTypesHint')}</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-1.5">
            {(evalMode === 'reranker' ? RERANKER_TASK_TYPES : EMBEDDING_TASK_TYPES).map((tt) => {
              const selected = ragTaskTypes.split(/[,，]/).map(s => s.trim()).filter(Boolean)
              return (
                <label key={tt} className="flex items-center gap-1.5 text-sm text-[var(--text)] cursor-pointer">
                  <input type="checkbox" checked={selected.includes(tt)}
                    onChange={() => {
                      const next = selected.includes(tt) ? selected.filter(s => s !== tt) : [...selected, tt]
                      setRagTaskTypes(next.join(', '))
                    }}
                    className="accent-[var(--accent)]" />
                  {tt}
                </label>
              )
            })}
          </div>
        </div>

        {/* Task Names with auto-complete */}
        <FormField label={t('eval.ragTaskNames')}>
          <div ref={nameInputRef} className="relative">
            <input value={ragTaskNames}
              onChange={(e) => handleTaskNameChange(e.target.value)}
              onFocus={() => { if (filteredNameSuggestions.length) setShowNameSuggestions(true) }}
              className={FORM_INPUT_CLASS} placeholder="AmazonReviewsClassification, ArxivClusteringP2P" />
            {showNameSuggestions && (
              <div className="absolute z-50 left-0 right-0 mt-1 rounded-[var(--radius-sm)] border border-[var(--border-md)] bg-[var(--bg-card)] shadow-[var(--shadow)] overflow-hidden max-h-48 overflow-y-auto">
                {filteredNameSuggestions.map((name) => (
                  <button key={name} type="button" onClick={() => selectNameSuggestion(name)}
                    className="w-full text-left px-3 py-2 text-sm text-[var(--text)] hover:bg-[var(--bg-card2)] transition-colors cursor-pointer">
                    {name}
                  </button>
                ))}
              </div>
            )}
          </div>
        </FormField>

        <FormField label={t('eval.ragLanguages')}>
          <input value={ragLanguages}
            onChange={(e) => setRagLanguages(e.target.value)}
            className={FORM_INPUT_CLASS} placeholder={t('eval.ragLanguagesHint')} />
        </FormField>

        <FormField label={t('eval.ragLimit')}>
          <input type="number" value={ragLimit}
            onChange={(e) => setRagLimit(e.target.value.replace(/[^0-9]/g, ''))}
            className={FORM_INPUT_CLASS} placeholder="10" />
        </FormField>

        {evalMode === 'reranker' && (
          <FormField label={t('eval.ragTopK')}>
            <input type="number" value={ragTopK}
              onChange={(e) => setRagTopK(e.target.value.replace(/[^0-9]/g, ''))}
              className={FORM_INPUT_CLASS} placeholder="10" />
          </FormField>
        )}
      </div>

      {evalMode === 'reranker' && (
        <div className="border-t border-[var(--border-md)] pt-3">
          <label className="flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer">
            <input type="checkbox" checked={ragTwoStage} onChange={(e) => setRagTwoStage(e.target.checked)}
              className="accent-[var(--accent)]" />
            {t('eval.ragTwoStage')}
          </label>
          {ragTwoStage && (
            <div className="mt-2">
              <FormField label={t('eval.ragEncoderModel')}>
                <input value={ragEncoderModel}
                  onChange={(e) => setRagEncoderModel(e.target.value)}
                  className={FORM_INPUT_CLASS} placeholder="BAAI/bge-large-zh-v1.5" />
              </FormField>
            </div>
          )}
        </div>
      )}

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow !mt-6">
        {t('eval.ragStartEval')}
      </Button>
    </form>
  )
}
