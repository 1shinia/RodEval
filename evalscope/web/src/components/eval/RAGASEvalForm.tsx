import { useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, inputClass } from '@/components/ui/formStyles'

const RAGAS_METRICS = [
  'answer_relevancy', 'faithfulness', 'context_precision',
  'context_recall', 'context_relevancy', 'answer_correctness',
]

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

export default function RAGASEvalForm({ onSubmit, disabled }: Props) {
  const { t } = useLocale()
  const [testsetFile, setTestsetFile] = useState('')
  const [llmModel, setLlmModel] = useState('')
  const [llmApiBase, setLlmApiBase] = useState('')
  const [llmApiKey, setLlmApiKey] = useState('')
  const [embModel, setEmbModel] = useState('')
  const [embProvider, setEmbProvider] = useState('huggingface')
  const [embApiBase, setEmbApiBase] = useState('')
  const [embApiKey, setEmbApiKey] = useState('')
  const [metrics, setMetrics] = useState<string[]>(['answer_relevancy', 'faithfulness'])
  const [language, setLanguage] = useState('english')
  const [errors, setErrors] = useState<Record<string, string>>({})

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}
    if (!testsetFile.trim()) newErrors.testsetFile = 'Required'
    if (!llmModel.trim()) newErrors.llmModel = 'Required'
    if (!llmApiBase.trim()) newErrors.llmApiBase = 'Required'
    if (!embModel.trim()) newErrors.embModel = 'Required'
    if (metrics.length === 0) newErrors.metrics = 'Select at least one metric'
    if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return }
    setErrors({})

    onSubmit({
      eval_backend: 'RAGEval',
      eval_config: {
        tool: 'ragas',
        eval: {
          testset_file: testsetFile.trim(),
          critic_llm: {
            model_name: llmModel.trim(),
            provider: 'openai',
            api_base: llmApiBase.trim(),
            api_key: llmApiKey,
          },
          embeddings: {
            model_name_or_path: embModel.trim(),
            provider: embProvider,
            api_base: embApiBase || undefined,
            api_key: embApiKey || undefined,
          },
          metrics,
          language,
        },
      },
    })
  }

  const toggleMetric = (m: string) => {
    setMetrics(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m])
    if (errors.metrics) setErrors(p => ({ ...p, metrics: '' }))
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Testset */}
      <FormField label="Testset File" required error={errors.testsetFile}>
        <input value={testsetFile}
          onChange={(e) => { setTestsetFile(e.target.value); if (errors.testsetFile) setErrors(p => ({ ...p, testsetFile: '' })) }}
          className={inputClass(errors.testsetFile)} placeholder="/data/testset.json" />
      </FormField>

      <div className="border-t border-[var(--border-md)] pt-3" />

      {/* Critic LLM */}
      <h4 className="text-sm font-medium text-[var(--text)]">Critic LLM</h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FormField label="Model Name" required error={errors.llmModel}>
          <input value={llmModel}
            onChange={(e) => { setLlmModel(e.target.value); if (errors.llmModel) setErrors(p => ({ ...p, llmModel: '' })) }}
            className={inputClass(errors.llmModel)} placeholder="gpt-4o-mini" />
        </FormField>
        <FormField label="API Base URL" required error={errors.llmApiBase}>
          <input value={llmApiBase}
            onChange={(e) => { setLlmApiBase(e.target.value); if (errors.llmApiBase) setErrors(p => ({ ...p, llmApiBase: '' })) }}
            className={inputClass(errors.llmApiBase)} placeholder="https://api.openai.com/v1" />
        </FormField>
        <FormField label="API Key">
          <input type="password" value={llmApiKey}
            onChange={e => setLlmApiKey(e.target.value)}
            className={FORM_INPUT_CLASS} placeholder="sk-..." />
        </FormField>
      </div>

      <div className="border-t border-[var(--border-md)] pt-3" />

      {/* Embedding Model */}
      <h4 className="text-sm font-medium text-[var(--text)]">Embedding Model</h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FormField label="Model Path / Name" required error={errors.embModel}>
          <input value={embModel}
            onChange={(e) => { setEmbModel(e.target.value); if (errors.embModel) setErrors(p => ({ ...p, embModel: '' })) }}
            className={inputClass(errors.embModel)} placeholder="BAAI/bge-small-en-v1.5" />
        </FormField>
        <FormField label="Provider">
          <select value={embProvider} onChange={e => setEmbProvider(e.target.value)} className={FORM_INPUT_CLASS}>
            <option value="huggingface">huggingface</option>
            <option value="openai">openai</option>
          </select>
        </FormField>
        <FormField label="API Base URL (optional)">
          <input value={embApiBase}
            onChange={e => setEmbApiBase(e.target.value)}
            className={FORM_INPUT_CLASS} placeholder="https://api.openai.com/v1" />
        </FormField>
        <FormField label="API Key (optional)">
          <input type="password" value={embApiKey}
            onChange={e => setEmbApiKey(e.target.value)}
            className={FORM_INPUT_CLASS} placeholder="sk-..." />
        </FormField>
      </div>

      <div className="border-t border-[var(--border-md)] pt-3" />

      {/* Metrics */}
      <div>
        <label className="text-sm font-medium text-[var(--text)]">Metrics</label>
        {errors.metrics && <p className="text-xs text-[var(--danger)] mt-0.5">{errors.metrics}</p>}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-1.5 mt-2">
          {RAGAS_METRICS.map(m => (
            <label key={m} className="flex items-center gap-1.5 text-sm text-[var(--text)] cursor-pointer">
              <input type="checkbox" checked={metrics.includes(m)} onChange={() => toggleMetric(m)}
                className="accent-[var(--accent)]" />
              {m}
            </label>
          ))}
        </div>
      </div>

      <FormField label="Language">
        <input value={language}
          onChange={e => setLanguage(e.target.value)}
          className={FORM_INPUT_CLASS} placeholder="english" />
      </FormField>

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow !mt-6">
        Start RAGAS Evaluation
      </Button>
    </form>
  )
}
