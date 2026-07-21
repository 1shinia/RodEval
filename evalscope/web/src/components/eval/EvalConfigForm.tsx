import { useLocale } from '@/contexts/LocaleContext'
import LLMEvalForm from './LLMEvalForm'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
  initialDataset?: string
  onApiKeyChange?: (apiKey: string) => void
}

export default function EvalConfigForm({ onSubmit, disabled, initialDataset, onApiKeyChange }: Props) {
  return (
    <LLMEvalForm
      onSubmit={onSubmit}
      disabled={disabled}
      initialDataset={initialDataset}
      onApiKeyChange={onApiKeyChange}
    />
  )
}
