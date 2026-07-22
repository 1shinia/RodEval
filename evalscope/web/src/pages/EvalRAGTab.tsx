import { useOutletContext } from 'react-router-dom'
import RAGEvalForm from '@/components/eval/RAGEvalForm'
import type { EvalTabContext } from '@/pages/EvalLayout'

export default function EvalRAGTab() {
  const { onSubmit, disabled } = useOutletContext<EvalTabContext>()
  return <RAGEvalForm onSubmit={onSubmit} disabled={disabled} />
}
