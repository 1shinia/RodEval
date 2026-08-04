import { useLocale } from '@/contexts/LocaleContext'
import LLMEvalForm from './LLMEvalForm'
import type { EvalTabContext } from '@/pages/EvalLayout'

interface Props {
  context: EvalTabContext
}

export default function EvalConfigForm({ context }: Props) {
  return (
    <LLMEvalForm context={context} />
  )
}
