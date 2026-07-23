import { useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS, FORM_LABEL_CLASS } from '@/components/ui/formStyles'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

type AIGCTool = 'txt2img' | 'txt2video' | 'img2img'

const IMAGE_DATASETS = [
  { value: 'drawbench', label: 'DrawBench (200 prompts)' },
  { value: 'parti', label: 'PartiPrompts (150 prompts)' },
  { value: 'custom', label: '自定义数据集' },
]

const VIDEO_DATASETS = [
  { value: 'msr_vtt', label: '内置视频提示词 (80 prompts)' },
  { value: 'custom', label: '自定义数据集' },
]

const IMAGE_METRICS = [
  { value: 'clip_score', label: 'CLIP Score' },
  { value: 'lpips', label: 'LPIPS (感知质量)' },
]

const VIDEO_METRICS = [
  { value: 'clip_score', label: 'CLIP Score (文本-视频对齐)' },
  { value: 'fvd', label: 'FVD (视频质量距离)' },
]

const VIDEO_RESOLUTIONS = [
  { value: '480p', label: '480P (标清)' },
  { value: '720p', label: '720P (高清)' },
  { value: '1080p', label: '1080P (全高清)' },
]

const VIDEO_ASPECT_RATIOS = [
  { value: '16:9', label: '16:9 (横屏)' },
  { value: '9:16', label: '9:16 (竖屏)' },
  { value: '4:3', label: '4:3 (传统)' },
  { value: '1:1', label: '1:1 (方形)' },
]

function resolveVideoSize(resolution: string, ratio: string): { width: number; height: number } {
  // Standard resolution mappings
  const presets: Record<string, Record<string, [number, number]>> = {
    '480p':  { '16:9': [854, 480], '9:16': [480, 854], '4:3': [640, 480], '1:1': [480, 480] },
    '720p':  { '16:9': [1280, 720], '9:16': [720, 1280], '4:3': [960, 720], '1:1': [720, 720] },
    '1080p': { '16:9': [1920, 1080], '9:16': [1080, 1920], '4:3': [1440, 1080], '1:1': [1080, 1080] },
  }
  const [w, h] = presets[resolution]?.[ratio] || [1280, 720]
  return { width: w, height: h }
}

export default function AIGCEvalForm({ onSubmit, disabled }: Props) {
  const { t } = useLocale()

  // Model source: API or Local
  const [modelSource, setModelSource] = useState<'api' | 'local'>('api')
  const isLocal = modelSource === 'local'

  const [tool, setTool] = useState<AIGCTool>('txt2img')

  const handleToolChange = (newTool: AIGCTool) => {
    setTool(newTool)
    if (newTool === 'txt2video') {
      setPromptDataset('msr_vtt')
      setMetrics(['clip_score'])
    } else {
      setPromptDataset('drawbench')
      setMetrics(['clip_score'])
    }
  }

  // API fields
  const [model, setModel] = useState('')
  const [apiBase, setApiBase] = useState('')
  const [apiKey, setApiKey] = useState('')

  // Local model fields
  const [modelPath, setModelPath] = useState('')
  const [device, setDevice] = useState('cuda')
  const [dtype, setDtype] = useState('float16')

  // Generation params
  const [width, setWidth] = useState('1024')
  const [height, setHeight] = useState('1024')
  const [steps, setSteps] = useState('50')
  const [guidance, setGuidance] = useState('7.5')
  const [negativePrompt, setNegativePrompt] = useState('')
  const [seed, setSeed] = useState('42')
  const [batchSize, setBatchSize] = useState('1')

  // Video-specific params
  const [numFrames, setNumFrames] = useState('16')
  const [fps, setFps] = useState('8')
  // Video resolution + aspect ratio (replaces width/height)
  const [videoResolution, setVideoResolution] = useState('720p')
  const [videoAspectRatio, setVideoAspectRatio] = useState('16:9')

  // Image-to-image params
  const [strength, setStrength] = useState('0.8')

  // Eval config
  const [promptDataset, setPromptDataset] = useState('drawbench')
  const [promptLimit, setPromptLimit] = useState('1')
  const [metrics, setMetrics] = useState<string[]>(['clip_score'])
  const [referenceVideoDir, setReferenceVideoDir] = useState('')
  const [customDatasetPath, setCustomDatasetPath] = useState('')

  const [errors, setErrors] = useState<Record<string, string>>({})

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}

    if (isLocal) {
      if (!modelPath.trim()) newErrors.modelPath = 'Required'
    } else {
      if (!model.trim()) newErrors.model = 'Required'
      if (!apiBase.trim()) newErrors.apiBase = 'Required'
      if (!apiKey.trim()) newErrors.apiKey = 'Required'
    }

    if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return }
    setErrors({})

    const modelConfig: Record<string, unknown> = {
      model_name_or_path: isLocal ? modelPath.trim() : model.trim(),
      model_type: tool,
      model_source: modelSource,
    }

    if (!isLocal) {
      modelConfig.api_base = apiBase.trim()
      if (apiKey.trim()) modelConfig.api_key = apiKey.trim()
    } else {
      modelConfig.device = device
      modelConfig.dtype = dtype
    }

    const generateConfig: Record<string, unknown> = {
      width: Number(width) || 512,
      height: Number(height) || 512,
      num_inference_steps: Number(steps) || 50,
      guidance_scale: Number(guidance) || 7.5,
      negative_prompt: negativePrompt.trim(),
      seed: Number(seed) || 42,
      batch_size: Number(batchSize) || 1,
    }

    if (tool === 'txt2video') {
      const size = resolveVideoSize(videoResolution, videoAspectRatio)
      generateConfig.width = size.width
      generateConfig.height = size.height
      generateConfig.num_frames = Number(numFrames) || 16
      generateConfig.fps = Number(fps) || 8
    }
    if (tool === 'img2img') {
      generateConfig.strength = Number(strength) || 0.8
    }

    const evalConfig: Record<string, unknown> = {
      metrics,
      prompt_dataset: promptDataset,
      prompt_limit: Number(promptLimit) || 100,
    }

    if (referenceVideoDir.trim()) {
      evalConfig.reference_video_dir = referenceVideoDir.trim()
    }
    if (promptDataset === 'custom' && customDatasetPath.trim()) {
      evalConfig.custom_dataset_path = customDatasetPath.trim()
    }

    onSubmit({
      eval_backend: 'AIGCEval',
      eval_config: {
        tool,
        model: modelConfig,
        generate: generateConfig,
        eval: evalConfig,
      },
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Tool Selector */}
      <div className="flex items-center gap-3 border-b border-[var(--border-md)] pb-4">
        <span className="text-sm font-medium text-[var(--text)]">{t('aigc.toolType')}</span>
        <select value={tool} onChange={e => handleToolChange(e.target.value as AIGCTool)} className={FORM_INPUT_CLASS + ' !w-auto'}>
          <option value="txt2img">{t('aigc.txt2img')}</option>
          <option value="txt2video">{t('aigc.txt2video')}</option>
          <option value="img2img">{t('aigc.img2img')}</option>
        </select>
      </div>

      {/* Model Source */}
      <div className="flex items-center gap-6">
        <label className={`${FORM_LABEL_CLASS} !mb-0`}>{t('eval.modelSource')}</label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="aigc_ms" value="api" checked={!isLocal}
            onChange={() => setModelSource('api')} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">{t('eval.modelSourceOpenAI')}</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="radio" name="aigc_ms" value="local" checked={isLocal}
            onChange={() => setModelSource('local')} className="accent-[var(--accent)]" />
          <span className="text-sm text-[var(--text)]">{t('eval.modelSourceLocal')}</span>
        </label>
      </div>

      {/* Model Configuration */}
      <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2">
        {t('aigc.modelConfig')}
      </h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* API mode fields */}
        {!isLocal && (<>
          <FormField label={t('eval.modelName')} required error={errors.model}>
            <input value={model}
              onChange={e => { setModel(e.target.value.trimStart()); if (errors.model) setErrors(p => ({ ...p, model: '' })) }}
              className={FORM_INPUT_CLASS} placeholder="stable-diffusion-v1-5" />
          </FormField>

          <FormField label={t('eval.apiUrl')} required error={errors.apiBase}>
            <input value={apiBase}
              onChange={e => { setApiBase(e.target.value); if (errors.apiBase) setErrors(p => ({ ...p, apiBase: '' })) }}
              className={FORM_INPUT_CLASS} placeholder="https://api.example.com/v1/images/generations" />
          </FormField>

          <FormField label={t('eval.apiKey')} required error={errors.apiKey}>
            <input type="password" value={apiKey}
              onChange={e => { setApiKey(e.target.value); if (errors.apiKey) setErrors(p => ({ ...p, apiKey: '' })) }}
              className={FORM_INPUT_CLASS} placeholder="sk-..." />
          </FormField>
        </>)}

        {/* Local model fields */}
        {isLocal && (<>
          <FormField label={t('eval.modelPath')} required error={errors.modelPath}>
            <input value={modelPath}
              onChange={e => { setModelPath(e.target.value); if (errors.modelPath) setErrors(p => ({ ...p, modelPath: '' })) }}
              className={FORM_INPUT_CLASS} placeholder="stabilityai/stable-diffusion-2-1 或 /data/models/sd" />
          </FormField>

          <FormField label={t('aigc.device')}>
            <select value={device} onChange={e => setDevice(e.target.value)} className={FORM_INPUT_CLASS}>
              <option value="cuda">CUDA (GPU)</option>
              <option value="cpu">CPU</option>
              <option value="mps">MPS (Apple Silicon)</option>
            </select>
          </FormField>

          <FormField label={t('aigc.dtype')}>
            <select value={dtype} onChange={e => setDtype(e.target.value)} className={FORM_INPUT_CLASS}>
              <option value="float16">float16</option>
              <option value="float32">float32</option>
              <option value="bfloat16">bfloat16</option>
            </select>
          </FormField>
        </>)}
      </div>

      {/* Generation Parameters */}
      <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2 pt-4">
        {t('aigc.generateConfig')}
      </h4>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {tool === 'txt2video' ? (<>
          <FormField label="分辨率">
            <select value={videoResolution} onChange={e => setVideoResolution(e.target.value)} className={FORM_INPUT_CLASS}>
              {VIDEO_RESOLUTIONS.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
            </select>
          </FormField>
          <FormField label="画幅比">
            <select value={videoAspectRatio} onChange={e => setVideoAspectRatio(e.target.value)} className={FORM_INPUT_CLASS}>
              {VIDEO_ASPECT_RATIOS.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
            </select>
          </FormField>
        </>) : (<>
          <FormField label={t('aigc.width')} hint={t('aigc.sizeHint')}>
            <input type="number" value={width}
              onChange={e => setWidth(e.target.value.replace(/[^0-9]/g, ''))}
              className={FORM_INPUT_CLASS} placeholder="1024" />
          </FormField>
          <FormField label={t('aigc.height')} hint={t('aigc.sizeHint')}>
            <input type="number" value={height}
              onChange={e => setHeight(e.target.value.replace(/[^0-9]/g, ''))}
              className={FORM_INPUT_CLASS} placeholder="1024" />
          </FormField>
        </>)}

        <FormField label={t('aigc.steps')}>
          <input type="number" value={steps}
            onChange={e => setSteps(e.target.value.replace(/[^0-9]/g, ''))}
            className={FORM_INPUT_CLASS} placeholder="50" />
        </FormField>

        <FormField label={t('aigc.guidance')}>
          <input type="number" value={guidance}
            onChange={e => setGuidance(e.target.value)}
            className={FORM_INPUT_CLASS} placeholder="7.5" step="0.1" />
        </FormField>

        <FormField label={t('aigc.seed')}>
          <input type="number" value={seed}
            onChange={e => setSeed(e.target.value.replace(/[^0-9]/g, ''))}
            className={FORM_INPUT_CLASS} placeholder="42" />
        </FormField>

        <FormField label={t('aigc.batchSize')}>
          <input type="number" value={batchSize}
            onChange={e => setBatchSize(e.target.value.replace(/[^0-9]/g, ''))}
            className={FORM_INPUT_CLASS} placeholder="1" />
        </FormField>

        {tool === 'txt2video' && (<>
          <FormField label={t('aigc.numFrames')}>
            <input type="number" value={numFrames}
              onChange={e => setNumFrames(e.target.value.replace(/[^0-9]/g, ''))}
              className={FORM_INPUT_CLASS} placeholder="16" />
          </FormField>

          <FormField label={t('aigc.fps')}>
            <input type="number" value={fps}
              onChange={e => setFps(e.target.value.replace(/[^0-9]/g, ''))}
              className={FORM_INPUT_CLASS} placeholder="8" />
          </FormField>
        </>)}

        {tool === 'img2img' && (
          <FormField label={t('aigc.strength')} hint="0-1，越大越偏离原图">
            <input type="number" value={strength}
              onChange={e => setStrength(e.target.value)}
              className={FORM_INPUT_CLASS} placeholder="0.8" step="0.05" min="0" max="1" />
          </FormField>
        )}
      </div>

      <FormField label={t('aigc.negativePrompt')}>
        <textarea value={negativePrompt}
          onChange={e => setNegativePrompt(e.target.value)}
          className={FORM_INPUT_CLASS + ' h-20 resize-none'}
          placeholder="low quality, blurry, distorted..." />
      </FormField>

      {/* Evaluation Configuration */}
      <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2 pt-4">
        {t('aigc.evalConfig')}
      </h4>

      <FormField label={t('aigc.promptDataset')}>
        <select value={promptDataset} onChange={e => setPromptDataset(e.target.value)} className={FORM_INPUT_CLASS}>
          {(tool === 'txt2video' ? VIDEO_DATASETS : IMAGE_DATASETS).map(ds => (
            <option key={ds.value} value={ds.value}>{ds.label}</option>
          ))}
        </select>
      </FormField>

      {promptDataset === 'custom' && (
        <FormField label="自定义数据集路径">
          <input value={customDatasetPath}
            onChange={e => setCustomDatasetPath(e.target.value)}
            className={FORM_INPUT_CLASS}
            placeholder="/path/to/prompts.txt（每行一条 prompt）" />
        </FormField>
      )}

      <FormField label={t('aigc.promptLimit')}>
        <input type="number" value={promptLimit}
          onChange={e => setPromptLimit(e.target.value.replace(/[^0-9]/g, ''))}
          className={FORM_INPUT_CLASS} placeholder="100" />
      </FormField>

      <FormField label={t('aigc.metrics')}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {(tool === 'txt2video' ? VIDEO_METRICS : IMAGE_METRICS).map(m => {
            const selected = metrics.includes(m.value)
            return (
              <label key={m.value} className="flex items-center gap-2 text-sm text-[var(--text)] cursor-pointer">
                <input type="checkbox" checked={selected}
                  onChange={() => {
                    const next = selected ? metrics.filter(x => x !== m.value) : [...metrics, m.value]
                    setMetrics(next)
                  }}
                  className="accent-[var(--accent)]" />
                {m.label}
              </label>
            )
          })}
        </div>
      </FormField>

      {metrics.includes('fvd') && (
        <FormField label="参考视频目录 (FVD)">
          <input value={referenceVideoDir}
            onChange={e => setReferenceVideoDir(e.target.value)}
            className={FORM_INPUT_CLASS} placeholder="/path/to/reference/videos (可选，留空则自参照)" />
        </FormField>
      )}

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow !mt-6">
        {t('aigc.startEval')}
      </Button>
    </form>
  )
}
