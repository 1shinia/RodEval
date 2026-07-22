import { useOutletContext } from 'react-router-dom'
import EvalConfigForm from '@/components/eval/EvalConfigForm'
import type { EvalTabContext } from '@/pages/EvalLayout'

export default function EvalLLMTab() {
  const { onSubmit, disabled, onApiKeyChange, initialDataset } = useOutletContext<EvalTabContext>()
  return (
    <EvalConfigForm
      onSubmit={onSubmit}
      disabled={disabled}
      initialDataset={initialDataset ?? undefined}
      onApiKeyChange={onApiKeyChange}
    />
  )
}
