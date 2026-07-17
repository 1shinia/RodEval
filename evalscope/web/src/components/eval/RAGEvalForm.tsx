import { useEffect, useRef, useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, FORM_LABEL_CLASS, inputClass } from '@/components/ui/formStyles'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

type RAGTool = 'embedding' | 'reranker' | 'ragas' | 'clip'

const EMBEDDING_TASK_TYPES = [
  'Classification', 'Clustering', 'Retrieval', 'STS',
  'PairClassification', 'BitextMining', 'Summarization',
  'MultilabelClassification', 'InstructionRetrieval', 'ZeroShotClassification',
]

const RERANKER_TASK_TYPES = ['Reranking', 'InstructionReranking']

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
  BitextMining: ['BUCC', 'Tatoeba'],
  Summarization: ['SummEval'],
  MultilabelClassification: ['MultiHateClassification', 'TweetTopicClassification'],
  InstructionRetrieval: ['Core17InstructionRetrieval', 'News21InstructionRetrieval'],
  ZeroShotClassification: ['AmazonCounterfactualClassification', 'ToxicConversationsClassification'],
  Reranking: [
    'AskUbuntuDupQuestions', 'MindSmallReranking', 'SciDocsRR',
    'StackOverflowDupQuestions', 'MMarcoReranking', 'T2Reranking',
    'WikipediaRerankingMultilingual', 'CMedQAv1-reranking', 'CMedQAv2-reranking',
  ],
  InstructionReranking: [
    'Core17InstructionRetrieval', 'News21InstructionRetrieval',
    'Robust04InstructionRetrieval', 'mFollowIR', 'mFollowIRCrossLingual',
  ],
}

const RAGAS_METRICS = [
  'answer_relevancy', 'faithfulness', 'context_precision',
  'context_recall', 'context_relevancy', 'answer_correctness',
]

const LANG_OPTIONS: [string, string][] = [
  ['eng', '英语'], ['zho', '中文'], ['deu', '德语'],
  ['fra', '法语'], ['spa', '西班牙语'], ['ita', '意大利语'],
  ['jpn', '日语'], ['kor', '韩语'], ['ara', '阿拉伯语'],
  ['rus', '俄语'], ['por', '葡萄牙语'],
]

export default function RAGEvalForm({ onSubmit, disabled }: Props) {
  const { t } = useLocale()

  const [ragTool, setRagTool] = useState<RAGTool>('embedding')
  // ── MTEB fields ──
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
  const [ragHub, setRagHub] = useState('modelscope')
  const [errors, setErrors] = useState<Record<string, string>>({})

  // ── RAGAS fields ──
  const [ragasTestset, setRagasTestset] = useState('')
  const [ragasLlmModel, setRagasLlmModel] = useState('')
  const [ragasLlmBase, setRagasLlmBase] = useState('')
  const [ragasLlmKey, setRagasLlmKey] = useState('')
  const [ragasEmbModel, setRagasEmbModel] = useState('')
  const [ragasEmbProv, setRagasEmbProv] = useState('huggingface')
  const [ragasEmbBase, setRagasEmbBase] = useState('')
  const [ragasEmbKey, setRagasEmbKey] = useState('')
  const [ragasMetrics, setRagasMetrics] = useState<string[]>(['answer_relevancy', 'faithfulness'])
  const [ragasLang, setRagasLang] = useState('english')

  // ── CLIP fields ──
  const [clipModelPath, setClipModelPath] = useState('')
  const [clipApiBase, setClipApiBase] = useState('')
  const [clipApiKey, setClipApiKey] = useState('')
  const [clipDatasets, setClipDatasets] = useState('')
  const [clipBatchSize, setClipBatchSize] = useState('128')
  const [clipLimit, setClipLimit] = useState('')

  // MTEB auto-complete
  const [showNameSuggestions, setShowNameSuggestions] = useState(false)
  const [filteredNameSuggestions, setFilteredNameSuggestions] = useState<string[]>([])
  const nameInputRef = useRef<HTMLDivElement>(null)

  const checkedTypes = ragTaskTypes.split(/[,，]/).map(s => s.trim()).filter(Boolean)
  const candidateDatasets = checkedTypes.length > 0
    ? [...new Set(checkedTypes.flatMap(tt => TYPE_DATASETS[tt] || []))]
    : Object.values(TYPE_DATASETS).flat()

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
    setErrors({})

    if (ragTool === 'ragas') {
      onSubmit({
        eval_backend: 'RAGEval',
        eval_config: {
          tool: 'ragas',
          eval: {
            testset_file: ragasTestset,
            critic_llm: {
              model_name: ragasLlmModel || 'gpt-4o-mini',
              provider: 'openai',
              api_base: ragasLlmBase,
              api_key: ragasLlmKey || undefined,
            },
            embeddings: {
              model_name_or_path: ragasEmbModel || 'BAAI/bge-small-en-v1.5',
              provider: ragasEmbProv,
              api_base: ragasEmbBase || undefined,
              api_key: ragasEmbKey || undefined,
            },
            metrics: ragasMetrics,
            language: ragasLang,
          },
        },
      })
      return
    }

    if (ragTool === 'clip') {
      const modelConfig: Record<string, unknown> = {}
      if (clipApiBase) {
        modelConfig.model_name = clipModelPath
        modelConfig.api_base = clipApiBase
        modelConfig.api_key = clipApiKey
      } else {
        modelConfig.model_name_or_path = clipModelPath
      }
      onSubmit({
        eval_backend: 'RAGEval',
        eval_config: {
          tool: 'clip_benchmark',
          eval: {
            models: [modelConfig],
            dataset_name: clipDatasets.split(/[,，]/).map(s => s.trim()).filter(Boolean),
            batch_size: clipBatchSize ? Number(clipBatchSize) : 128,
            limit: clipLimit ? Number(clipLimit) : undefined,
          },
        },
      })
      return
    }

    // MTEB submit
    const isApi = ragModelSource === 'api'
    const modelConfig: Record<string, unknown> = {
      is_cross_encoder: ragTool === 'reranker',
      hub: ragHub,
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
      if (ragTool === 'embedding' && ragPooling) modelConfig.pooling_mode = ragPooling
    }

    const models: Record<string, unknown>[] = [modelConfig]
    if (ragTwoStage && ragEncoderModel.trim()) {
      models.unshift({
        model_name_or_path: ragEncoderModel.trim(),
        is_cross_encoder: false,
      })
    }

    const evalCfg: Record<string, unknown> = {
      hub: ragHub,
    }
    if (ragTaskTypes) evalCfg.task_types = ragTaskTypes.split(/[,，]/).map((s: string) => s.trim()).filter(Boolean)
    if (ragTaskNames) evalCfg.task_names = ragTaskNames.split(/[,，]/).map((s: string) => s.trim()).filter(Boolean)
    if (ragLanguages) evalCfg.languages = ragLanguages.split(/[,，]/).map((s: string) => s.trim()).filter(Boolean)
    if (ragLimit) evalCfg.limits = Number(ragLimit)
    if (ragTopK) evalCfg.top_k = Number(ragTopK)

    onSubmit({
      eval_backend: 'RAGEval',
      eval_config: { tool: 'mteb', models, eval: evalCfg },
    })
  }

  const isMTEB = ragTool === 'embedding' || ragTool === 'reranker'

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* RAG Tool Selector */}
      <div className="flex items-center gap-3 border-b border-[var(--border-md)] pb-4">
        <span className="text-sm font-medium text-[var(--text)]">RAG Tool</span>
        <select value={ragTool} onChange={e => setRagTool(e.target.value as RAGTool)} className={FORM_INPUT_CLASS + ' !w-auto'}>
          <option value="embedding">MTEB Embedding</option>
          <option value="reranker">MTEB Reranker</option>
          <option value="ragas">RAGAS</option>
          <option value="clip">CLIP Benchmark</option>
        </select>
      </div>

      {/* ── MTEB Embedding / Reranker ── */}
      {isMTEB && (
        <>
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
            {ragModelSource === 'api' && (
              <>
                <FormField label={t('eval.modelName')} required>
                  <input value={ragModelPath}
                    onChange={e => { setRagModelPath(e.target.value); if (errors.ragModelPath) setErrors(p => ({ ...p, ragModelPath: '' })) }}
                    className={inputClass(errors.ragModelPath)} placeholder="text-embedding-3-small" />
                </FormField>
                <FormField label={t('eval.apiUrl')} required>
                  <input value={ragApiBase}
                    onChange={e => { setRagApiBase(e.target.value); if (errors.ragApiBase) setErrors(p => ({ ...p, ragApiBase: '' })) }}
                    className={inputClass(errors.ragApiBase)} placeholder="https://api.openai.com/v1（自动追加 /embeddings，无需手动填写）" />
                </FormField>
                <FormField label={t('eval.apiKey')} required>
                  <input type="password" value={ragApiKey}
                    onChange={e => { setRagApiKey(e.target.value); if (errors.ragApiKey) setErrors(p => ({ ...p, ragApiKey: '' })) }}
                    className={FORM_INPUT_CLASS} placeholder="sk-..." />
                </FormField>
                {ragTool === 'embedding' && (
                  <FormField label={t('eval.ragDimension')}>
                    <input type="number" value={ragDimension}
                      onChange={e => setRagDimension(e.target.value.replace(/[^0-9]/g, ''))}
                      className={FORM_INPUT_CLASS} placeholder="1024" />
                  </FormField>
                )}
              </>
            )}

            {ragModelSource === 'local' && (
              <>
                <FormField label={t('eval.modelPath')} required>
                  <input value={ragModelPath}
                    onChange={e => { setRagModelPath(e.target.value); if (errors.ragModelPath) setErrors(p => ({ ...p, ragModelPath: '' })) }}
                    className={inputClass(errors.ragModelPath)} placeholder="BAAI/bge-small-zh-v1.5" />
                </FormField>
                <FormField label={t('eval.ragMaxSeqLen')}>
                  <input type="number" value={ragMaxSeqLen}
                    onChange={e => setRagMaxSeqLen(e.target.value.replace(/[^0-9]/g, ''))}
                    className={FORM_INPUT_CLASS} placeholder="512" />
                </FormField>
                <FormField label={t('eval.ragBatchSize')}>
                  <input type="number" value={ragBatchSize}
                    onChange={e => setRagBatchSize(e.target.value.replace(/[^0-9]/g, ''))}
                    className={FORM_INPUT_CLASS} placeholder="32" />
                </FormField>
                {ragTool === 'embedding' && (
                  <FormField label={t('eval.ragPoolingMode')}>
                    <select value={ragPooling} onChange={e => setRagPooling(e.target.value)} className={FORM_INPUT_CLASS}>
                      <option value="">auto</option>
                      <option value="mean">mean</option>
                      <option value="cls">cls</option>
                    </select>
                  </FormField>
                )}
              </>
            )}

            <FormField label="数据集来源">
              <select value={ragHub} onChange={e => setRagHub(e.target.value)} className={FORM_INPUT_CLASS}>
                <option value="modelscope">ModelScope</option>
                <option value="huggingface">HuggingFace</option>
              </select>
            </FormField>

            <div className="md:col-span-2 border-t border-[var(--border-md)] pt-3"></div>
            <div className="md:col-span-2">
              <label className={FORM_LABEL_CLASS}>{t('eval.ragPrompt')}</label>
              <p className="text-xs text-[var(--text-muted)] mb-1.5">{t('eval.ragPromptHint')}</p>
              <input value={ragPrompt} onChange={e => setRagPrompt(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder={t('eval.ragPromptHint')} />
            </div>

            <div className="md:col-span-2 border-t border-[var(--border-md)] pt-3"></div>
            <div className="md:col-span-2">
              <FormField label={t('eval.ragTaskTypesTitle')} required>
                <p className="text-xs text-[var(--text-muted)] mb-2">{t('eval.ragTaskTypesHint')}</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-1.5">
                  {(ragTool === 'reranker' ? RERANKER_TASK_TYPES : EMBEDDING_TASK_TYPES).map((tt) => {
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
              </FormField>
            </div>

            <FormField label={t('eval.ragTaskNames')} required>
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
              <div className="flex flex-wrap gap-1.5 mb-2">
                {LANG_OPTIONS.map(([code, label]) => {
                  const langs = ragLanguages.split(/[,，]/).map(s => s.trim()).filter(Boolean)
                  const selected = langs.includes(code)
                  return (
                    <button key={code} type="button"
                      onClick={() => {
                        const next = selected ? langs.filter(s => s !== code) : [...langs, code]
                        setRagLanguages(next.join(', '))
                      }}
                      className={`px-2 py-1 text-xs rounded-full border transition-colors cursor-pointer ${selected ? 'bg-[var(--accent)] text-white border-[var(--accent)]' : 'bg-[var(--bg-card)] text-[var(--text-muted)] border-[var(--border)] hover:border-[var(--accent-dim)]'}`}>
                      {label} ({code})
                    </button>
                  )
                })}
              </div>
              <input value={ragLanguages}
                onChange={e => setRagLanguages(e.target.value)}
                className={`${FORM_INPUT_CLASS} text-xs`} placeholder="或手动输入其他语言代码（逗号分隔）" />
            </FormField>

            <FormField label={t('eval.ragLimit')}>
              <input type="number" value={ragLimit}
                onChange={e => setRagLimit(e.target.value.replace(/[^0-9]/g, ''))}
                className={FORM_INPUT_CLASS} placeholder="全量" />
            </FormField>

            {ragTool === 'reranker' && (
              <FormField label={t('eval.ragTopK')}>
                <input type="number" value={ragTopK}
                  onChange={e => setRagTopK(e.target.value.replace(/[^0-9]/g, ''))}
                  className={FORM_INPUT_CLASS} placeholder="10" />
              </FormField>
            )}
          </div>

          {ragTool === 'reranker' && (
            <div className="md:col-span-2 border-t border-[var(--border-md)] pt-3"></div>
          )}

          {ragTool === 'reranker' && (
            <div className="md:col-span-2">
              <div className="flex items-center gap-3 p-4 rounded-[var(--radius)] border border-[var(--border-md)] bg-[var(--bg-card2)] cursor-pointer"
                onClick={() => setRagTwoStage(!ragTwoStage)}>
                <input type="checkbox" checked={ragTwoStage} readOnly className="accent-[var(--accent)]" />
                <div>
                  <span className="text-sm font-medium text-[var(--text)]">{t('eval.ragTwoStage')}</span>
                  <p className="text-xs text-[var(--text-muted)] mt-0.5">Encoder 先检索 → Reranker 再精排，提升召回精度</p>
                </div>
              </div>
              {ragTwoStage && (
                <div className="mt-3 p-4 rounded-[var(--radius)] border border-[var(--accent-dim)] bg-[var(--bg-card)]">
                  <FormField label="Encoder 模型（Stage 1）">
                    <input value={ragEncoderModel}
                      onChange={e => setRagEncoderModel(e.target.value)}
                      className={FORM_INPUT_CLASS} placeholder="选择或输入 Embedding 模型"
                      list="encoder-models" />
                    <datalist id="encoder-models">
                      <option value="BAAI/bge-large-zh-v1.5" />
                      <option value="BAAI/bge-small-zh-v1.5" />
                      <option value="BAAI/bge-large-en-v1.5" />
                      <option value="BAAI/bge-small-en-v1.5" />
                      <option value="intfloat/e5-large-v2" />
                      <option value="intfloat/e5-base-v2" />
                      <option value="sentence-transformers/all-MiniLM-L6-v2" />
                      <option value="sentence-transformers/all-mpnet-base-v2" />
                    </datalist>
                  </FormField>
                </div>
              )}
            </div>
          )}

          <Button type="submit" variant="primary" disabled={disabled} className="btn-glow !mt-6">
            {t('eval.ragStartEval')}
          </Button>
        </>
      )}

      {/* ── RAGAS ── */}
      {ragTool === 'ragas' && (
        <>
          <FormField label="Testset File" required>
            <input value={ragasTestset} onChange={e => setRagasTestset(e.target.value)}
              className={FORM_INPUT_CLASS} placeholder="/data/testset.json" />
          </FormField>

          <div className="border-t border-[var(--border-md)] pt-3" />
          <h4 className="text-sm font-medium text-[var(--text)]">Critic LLM</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FormField label="Model Name" required>
              <input value={ragasLlmModel} onChange={e => setRagasLlmModel(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="gpt-4o-mini" />
            </FormField>
            <FormField label="API Base URL" required>
              <input value={ragasLlmBase} onChange={e => setRagasLlmBase(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="https://api.openai.com/v1" />
            </FormField>
            <FormField label="API Key">
              <input type="password" value={ragasLlmKey} onChange={e => setRagasLlmKey(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="sk-..." />
            </FormField>
          </div>

          <div className="border-t border-[var(--border-md)] pt-3" />
          <h4 className="text-sm font-medium text-[var(--text)]">Embedding Model</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FormField label="Model Path / Name" required>
              <input value={ragasEmbModel} onChange={e => setRagasEmbModel(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="BAAI/bge-small-en-v1.5" />
            </FormField>
            <FormField label="Provider">
              <select value={ragasEmbProv} onChange={e => setRagasEmbProv(e.target.value)} className={FORM_INPUT_CLASS}>
                <option value="huggingface">huggingface</option>
                <option value="openai">openai</option>
              </select>
            </FormField>
            <FormField label="API Base URL (optional)">
              <input value={ragasEmbBase} onChange={e => setRagasEmbBase(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="https://api.openai.com/v1" />
            </FormField>
            <FormField label="API Key (optional)">
              <input type="password" value={ragasEmbKey} onChange={e => setRagasEmbKey(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="sk-..." />
            </FormField>
          </div>

          <div className="border-t border-[var(--border-md)] pt-3" />
          <div>
            <label className="text-sm font-medium text-[var(--text)]">Metrics</label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-1.5 mt-2">
              {RAGAS_METRICS.map(m => (
                <label key={m} className="flex items-center gap-1.5 text-sm text-[var(--text)] cursor-pointer">
                  <input type="checkbox" checked={ragasMetrics.includes(m)}
                    onChange={() => setRagasMetrics(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m])}
                    className="accent-[var(--accent)]" />
                  {m}
                </label>
              ))}
            </div>
          </div>

          <FormField label="Language">
            <input value={ragasLang} onChange={e => setRagasLang(e.target.value)}
              className={FORM_INPUT_CLASS} placeholder="english" />
          </FormField>

          <Button type="submit" variant="primary" disabled={disabled} className="btn-glow !mt-6">
            Start RAGAS Evaluation
          </Button>
        </>
      )}

      {/* ── CLIP Benchmark ── */}
      {ragTool === 'clip' && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FormField label="Model Path / Name" required>
              <input value={clipModelPath} onChange={e => setClipModelPath(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="openai/clip-vit-base-patch32" />
            </FormField>
            <FormField label="API Base URL (optional)">
              <input value={clipApiBase} onChange={e => setClipApiBase(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="https://api.openai.com/v1" />
            </FormField>
            <FormField label="API Key (optional)">
              <input type="password" value={clipApiKey} onChange={e => setClipApiKey(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="sk-..." />
            </FormField>
            <FormField label="Datasets" required>
              <input value={clipDatasets} onChange={e => setClipDatasets(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="flickr30k, msr-vtt" />
            </FormField>
            <FormField label="Batch Size">
              <input type="number" value={clipBatchSize}
                onChange={e => setClipBatchSize(e.target.value.replace(/[^0-9]/g, ''))}
                className={FORM_INPUT_CLASS} placeholder="128" />
            </FormField>
            <FormField label="Limit">
              <input type="number" value={clipLimit}
                onChange={e => setClipLimit(e.target.value.replace(/[^0-9]/g, ''))}
                className={FORM_INPUT_CLASS} placeholder="全量" />
            </FormField>
          </div>

          <Button type="submit" variant="primary" disabled={disabled} className="btn-glow !mt-6">
            Start CLIP Evaluation
          </Button>
        </>
      )}
    </form>
  )
}
