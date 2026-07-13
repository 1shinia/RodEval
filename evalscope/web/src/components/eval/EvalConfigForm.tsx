import { useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS } from '@/components/ui/formStyles'
import LLMEvalForm from './LLMEvalForm'
import RAGEvalForm from './RAGEvalForm'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
  initialDataset?: string
  onApiKeyChange?: (apiKey: string) => void
}

export default function EvalConfigForm({ onSubmit, disabled, initialDataset, onApiKeyChange }: Props) {
  const { t } = useLocale()
  const [evalMode, setEvalMode] = useState('llm')

  return (
    <div className="space-y-4">
      {/* Eval Mode Selector */}
      <div className="flex items-center gap-4 border-b border-[var(--border-md)] pb-3 -mt-1">
        <FormField label={t('eval.evalMode')}>
          <select value={evalMode} onChange={(e) => setEvalMode(e.target.value)} className={FORM_INPUT_CLASS}>
            <option value="llm">{t('eval.evalModeLLM')}</option>
            <option value="embedding">{t('eval.evalModeEmbedding')}</option>
            <option value="reranker">{t('eval.evalModeReranker')}</option>
          </select>
        </FormField>
      </div>

      {/* Delegate to appropriate sub-form */}
      {evalMode !== 'llm' ? (
        <RAGEvalForm
          onSubmit={onSubmit}
          disabled={disabled}
          evalMode={evalMode as 'embedding' | 'reranker'}
        />
      ) : (
        <LLMEvalForm
          onSubmit={onSubmit}
          disabled={disabled}
          initialDataset={initialDataset}
          onApiKeyChange={onApiKeyChange}
        />
      )}
    </div>
  )
}
