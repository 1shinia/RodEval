import { useOutletContext } from 'react-router-dom'
import AIGCEvalForm from '@/components/eval/AIGCEvalForm'
import type { EvalTabContext } from '@/pages/EvalLayout'

export default function EvalAIGCTab() {
  const { onSubmit, disabled } = useOutletContext<EvalTabContext>()
  return <AIGCEvalForm onSubmit={onSubmit} disabled={disabled} />
}
