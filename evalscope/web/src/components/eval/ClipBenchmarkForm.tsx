import { useState, type SyntheticEvent } from 'react'
import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, inputClass } from '@/components/ui/formStyles'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

export default function ClipBenchmarkForm({ onSubmit, disabled }: Props) {
  const [modelPath, setModelPath] = useState('')
  const [apiBase, setApiBase] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [datasets, setDatasets] = useState('')
  const [batchSize, setBatchSize] = useState('128')
  const [limit, setLimit] = useState('')
  const [errors, setErrors] = useState<Record<string, string>>({})

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}
    if (!modelPath.trim()) newErrors.modelPath = 'Required'
    if (!datasets.trim()) newErrors.datasets = 'Required'
    if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return }
    setErrors({})

    const modelConfig: Record<string, unknown> = {}
    if (apiBase.trim()) {
      modelConfig.model_name = modelPath.trim()
      modelConfig.api_base = apiBase.trim()
      modelConfig.api_key = apiKey
    } else {
      modelConfig.model_name_or_path = modelPath.trim()
    }

    onSubmit({
      eval_backend: 'RAGEval',
      eval_config: {
        tool: 'clip_benchmark',
        eval: {
          models: [modelConfig],
          dataset_name: datasets.split(/[,，]/).map(s => s.trim()).filter(Boolean),
          batch_size: batchSize ? Number(batchSize) : 128,
          limit: limit ? Number(limit) : undefined,
        },
      },
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FormField label="Model Path / Name" required error={errors.modelPath}>
          <input value={modelPath}
            onChange={(e) => { setModelPath(e.target.value); if (errors.modelPath) setErrors(p => ({ ...p, modelPath: '' })) }}
            className={inputClass(errors.modelPath)} placeholder="openai/clip-vit-base-patch32" />
        </FormField>
        <FormField label="API Base URL (optional)">
          <input value={apiBase}
            onChange={e => setApiBase(e.target.value)}
            className={FORM_INPUT_CLASS} placeholder="https://api.openai.com/v1" />
        </FormField>
        <FormField label="API Key (optional)">
          <input type="password" value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            className={FORM_INPUT_CLASS} placeholder="sk-..." />
        </FormField>
        <FormField label="Datasets" required error={errors.datasets}>
          <input value={datasets}
            onChange={(e) => { setDatasets(e.target.value); if (errors.datasets) setErrors(p => ({ ...p, datasets: '' })) }}
            className={inputClass(errors.datasets)} placeholder="flickr30k, msr-vtt" />
        </FormField>
        <FormField label="Batch Size">
          <input type="number" value={batchSize}
            onChange={e => setBatchSize(e.target.value.replace(/[^0-9]/g, ''))}
            className={FORM_INPUT_CLASS} placeholder="128" />
        </FormField>
        <FormField label="Limit">
          <input type="number" value={limit}
            onChange={e => setLimit(e.target.value.replace(/[^0-9]/g, ''))}
            className={FORM_INPUT_CLASS} placeholder="全量" />
        </FormField>
      </div>

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow !mt-6">
        Start CLIP Evaluation
      </Button>
    </form>
  )
}
