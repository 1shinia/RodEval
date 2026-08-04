import { useOutletContext } from 'react-router-dom'
import EvalConfigForm from '@/components/eval/EvalConfigForm'
import type { EvalTabContext } from '@/pages/EvalLayout'

export default function EvalLLMTab() {
  const context = useOutletContext<EvalTabContext>()
  return <EvalConfigForm context={context} />
}
