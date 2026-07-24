import { useOutletContext } from 'react-router-dom'
import AudioEvalForm from '@/components/eval/AudioEvalForm'
import type { EvalTabContext } from '@/pages/EvalLayout'

export default function EvalAudioTab() {
  const { onSubmit, disabled } = useOutletContext<EvalTabContext>()
  return <AudioEvalForm onSubmit={onSubmit} disabled={disabled} />
}
