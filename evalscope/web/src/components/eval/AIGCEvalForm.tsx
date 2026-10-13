import { useEffect, useMemo, useState, type SyntheticEvent } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import Button from '@/components/ui/Button'
import FormField from '@/components/ui/FormField'
import { FORM_INPUT_CLASS } from '@/components/ui/formStyles'

interface Props {
  onSubmit: (config: Record<string, unknown>) => void
  disabled?: boolean
}

type AIGCTool = 'txt2img' | 'txt2video' | 'img2img'

const IMAGE_DATASETS = [
  { value: 'drawbench', label: 'DrawBench (200 prompts)' },
  { value: 'parti', label: 'PartiPrompts (150 prompts)' },
  { value: 'custom', label: '自定义数据集' },
  { value: '', label: '不使用（仅用自定义提示词）' },
]

const VIDEO_DATASETS = [
  { value: 'msr_vtt', label: '内置视频提示词 (80 prompts)' },
  { value: 'custom', label: '自定义数据集' },
  { value: '', label: '不使用（仅用自定义提示词）' },
]

const TEXT_TO_IMAGE_METRICS = [
  { value: 'clip_score', label: 'CLIP Score' },
]

const IMAGE_TO_IMAGE_METRICS = [
  { value: 'clip_score', label: 'CLIP Score' },
  { value: 'lpips', label: '参考图感知距离' },
]

const VIDEO_METRICS = [
  { value: 'clip_score', label: 'CLIP Score (文本-视频对齐)' },
]

// ── 视频生成能力表 ────────────────────────────────────────────────────
// provider + 模型前缀决定：可选分辨率档位、每档可用画幅比、可选参数、时长提示。
// 与后端 evalscope/backend/aigc_eval/models/adapters.py 的 _DASHSCOPE_VIDEO_FAMILIES 一一对应，
// 依据百炼三套官方协议整理（wan2.7 用 resolution / wan2.1~2.6 用 size / kling 用 mode）。
// 前端只负责「只展示可达选项 + 给提示」，值域校验由后端负责（非法值写进任务日志）。
interface VideoTier {
  value: string
  label: string
  ratios: string[]
}

interface VideoCapability {
  tiers: VideoTier[]
  defaultTier: string
  defaultRatio: string
  seed: boolean
  durationHint: string
  /** 该模型可选的时长（秒，字符串）。空数组 = 无法枚举（如 OpenAI 兼容透传），回退自由输入 */
  durations: string[]
  /** 模型 id 是否可被识别（false 时后端必报错，前端直接拦住） */
  known: boolean
}

/** 生成 low~high 的整秒选项，供时长下拉使用（与后端 range 语义一致） */
const intRange = (low: number, high: number) => Array.from({ length: high - low + 1 }, (_, i) => String(low + i))

const RATIO_LABELS: Record<string, string> = {
  '16:9': '16:9 (横屏)',
  '9:16': '9:16 (竖屏)',
  '1:1': '1:1 (方形)',
  '4:3': '4:3 (传统)',
  '3:4': '3:4 (竖版)',
  '21:9': '21:9 (超宽)',
}

const TIER_LABELS: Record<string, string> = {
  '480p': '480P (标清)',
  '720p': '720P (高清)',
  '1080p': '1080P (全高清)',
}

const KLING_TIER_LABELS: Record<string, string> = {
  '720p': '720P (标准 std)',
  '1080p': '1080P (专业 pro)',
}

/** 火山方舟文生视频画幅比（官方可选值；adaptive 不暴露给用户，仅后端放行） */
const VOLC_RATIOS = ['16:9', '4:3', '1:1', '3:4', '9:16', '21:9']

const WAN_RATIOS: Record<string, string[]> = {
  '480p': ['16:9', '9:16', '1:1'],
  '720p': ['16:9', '9:16', '1:1', '4:3', '3:4'],
  '1080p': ['16:9', '9:16', '1:1', '4:3', '3:4'],
}

function wanCapability(
  tiers: string[], durationHint: string, defaultTier: string, durations: string[],
): VideoCapability {
  return {
    tiers: tiers.map(value => ({ value, label: TIER_LABELS[value] ?? value, ratios: WAN_RATIOS[value] ?? [] })),
    defaultTier,
    defaultRatio: '16:9',
    seed: true,
    durationHint,
    durations,
    known: true,
  }
}

/** 火山方舟 Seedance 能力：分辨率与时长都按模型分档（严格照官方模型列表）。
 *  更具体的前缀必须先判：2-0-fast / 2-0-mini 只到 720P。4k 官方仅 2-0 支持，
 *  但并发仅 1，暂不放入下拉（需要时可加）。 */
function volcengineCapability(model: string): VideoCapability {
  const name = model.trim().toLowerCase()
  const build = (tiers: string[], durationHint: string, durations: string[]): VideoCapability => ({
    tiers: tiers.map(value => ({ value, label: TIER_LABELS[value] ?? value, ratios: VOLC_RATIOS })),
    defaultTier: '720p',
    defaultRatio: '16:9',
    seed: true,
    durationHint,
    durations,
    known: true,
  })
  if (name.startsWith('doubao-seedance-2-5')) {
    return build(['480p', '720p', '1080p'], '限支持4～30 秒整数', intRange(4, 30))
  }
  if (name.startsWith('doubao-seedance-2-0-fast') || name.startsWith('doubao-seedance-2-0-mini')) {
    return build(['480p', '720p'], '限支持4～15 秒整数（该模型分辨率仅 480P/720P）', intRange(4, 15))
  }
  if (name.startsWith('doubao-seedance-2-0')) {
    return build(['480p', '720p', '1080p'], '限支持4～15 秒整数', intRange(4, 15))
  }
  if (name.startsWith('doubao-seedance-1-0')) {
    return build(['480p', '720p', '1080p'], '限支持2～12 秒整数', intRange(2, 12))
  }
  // 未识别的火山模型：不猜参数形态（后端同样报错）
  return { ...build(['480p', '720p'], '未识别的火山视频模型', ['5']), known: false }
}

function videoCapability(provider: string, model: string): VideoCapability {
  const name = model.trim().toLowerCase()
  if (provider === 'dashscope') {
    if (name.startsWith('kling')) {
      return {
        tiers: ['720p', '1080p'].map(value => ({
          value, label: KLING_TIER_LABELS[value], ratios: ['16:9', '9:16', '1:1'],
        })),
        defaultTier: '1080p',
        defaultRatio: '16:9',
        seed: false, // kling 不支持 seed
        durationHint: '限支持3～15 秒整数',
        durations: intRange(3, 15),
        known: true,
      }
    }
    if (name.startsWith('wan2.7')) return wanCapability(['720p', '1080p'], '限支持2～15 秒整数', '1080p', intRange(2, 15))
    if (name.startsWith('wan2.6')) return wanCapability(['720p', '1080p'], '限支持2～15 秒整数', '1080p', intRange(2, 15))
    if (name.startsWith('wan2.5')) return wanCapability(['480p', '720p', '1080p'], '该模型只支持 5 或 10 秒', '1080p', ['5', '10'])
    if (name.startsWith('wan2.2')) return wanCapability(['480p', '1080p'], '该模型时长固定为 5 秒', '1080p', ['5'])
    if (name.includes('t2v-turbo')) return wanCapability(['480p', '720p'], '该模型时长固定为 5 秒', '720p', ['5'])
    if (name.includes('t2v-plus')) return wanCapability(['720p'], '该模型时长固定为 5 秒', '720p', ['5'])
    // 未识别的百炼视频模型：不猜参数形态（后端会明确报错）
    return { ...wanCapability(['720p', '1080p'], '限支持2～15 秒整数', '1080p', intRange(2, 15)), known: false }
  }
  if (provider === 'volcengine') return volcengineCapability(model)
  // OpenAI 兼容：分辨率/画幅比原样透传（时长无法枚举，保留自由输入）
  return {
    tiers: ['480p', '720p', '1080p'].map(value => ({
      value, label: TIER_LABELS[value], ratios: ['16:9', '9:16', '4:3', '1:1'],
    })),
    defaultTier: '720p',
    defaultRatio: '16:9',
    seed: true,
    durationHint: '限支持5～15 秒整数',
    durations: [],
    known: true,
  }
}

/** 百炼（DashScope）地址必须配 DashScope 提供商：否则 OpenAI 兼容协议会请求
 *  {api_base}/video/generations —— 百炼无此端点（实测 404），body 形态也不同。 */
const needsDashscopeProvider = (provider: string, base: string) =>
  provider === 'dashscope' ? false : /aliyuncs\.com/i.test(base.trim())

/** 百炼 compatible-mode 是 OpenAI 兼容入口，没有视频合成端点（实测 404）；
 *  文生视频必须用 {host}/api/v1。 */
const compatibleModeVideoBase = (base: string) =>
  /aliyuncs\.com/i.test(base.trim()) && /compatible-mode/i.test(base)

const PROVIDER_OPTIONS: { value: 'openai' | 'dashscope' | 'volcengine'; label: string; apiPlaceholder: string; modelPlaceholder: string }[] = [
  {
    value: 'openai', label: 'OpenAI 兼容',
    apiPlaceholder: 'https://api.example.com/v1', modelPlaceholder: 'stable-diffusion-v1-5',
  },
  {
    value: 'dashscope', label: 'DashScope (百炼)',
    apiPlaceholder: 'https://dashscope.aliyuncs.com/api/v1', modelPlaceholder: 'kling/kling-v3-omni-video-generation',
  },
  {
    value: 'volcengine', label: '火山引擎',
    apiPlaceholder: 'https://ark.cn-beijing.volces.com/api/v3', modelPlaceholder: 'doubao-seedance-2-0-260128',
  },
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

  // API provider
  const [provider, setProvider] = useState<'openai' | 'dashscope' | 'volcengine'>('openai')
  const currentProvider = PROVIDER_OPTIONS.find(p => p.value === provider) ?? PROVIDER_OPTIONS[0]

  const [tool, setTool] = useState<AIGCTool>('txt2img')

  const handleToolChange = (newTool: AIGCTool) => {
    setTool(newTool)
    if (newTool !== 'txt2video' && provider === 'volcengine') setProvider('openai')
    if (newTool === 'txt2video') {
      setPromptDataset('msr_vtt')
      setMetrics(['clip_score'])
    } else if (newTool === 'img2img') {
      setPromptDataset('')
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


  // Generation params
  const [width, setWidth] = useState('1024')
  const [height, setHeight] = useState('1024')
  const [steps, setSteps] = useState('50')
  const [guidance, setGuidance] = useState('7.5')

  const [seed, setSeed] = useState('42')

  // Video-specific params
  const [videoDuration, setVideoDuration] = useState('5')
  // Video resolution + aspect ratio (replaces width/height)
  const [videoResolution, setVideoResolution] = useState('720p')
  const [videoAspectRatio, setVideoAspectRatio] = useState('16:9')

  // 由 provider + 模型前缀推导的生成能力（可选档位/画幅比/时长/可选参数）
  const videoCap = useMemo(() => videoCapability(provider, model), [provider, model])

  // 时长与分辨率一样按模型给可选项；当前值在新模型下不可用时回落到 5 秒
  // （5 秒是所有受支持模型都合法的值），纯渲染期派生，无需额外 effect。
  const durationOptions = videoCap.durations
  const videoDurationValue = durationOptions.includes(videoDuration)
    ? videoDuration
    : (durationOptions.includes('5') ? '5' : durationOptions[0] ?? videoDuration)

  // 切换供应商或模型后，把已不可用的分辨率回落到该模型默认档位（用户在界面上可见）
  useEffect(() => {
    if (tool !== 'txt2video') return
    if (!videoCap.tiers.some(tier => tier.value === videoResolution)) {
      setVideoResolution(videoCap.defaultTier)
    }
  }, [tool, videoCap, videoResolution])

  // 画幅比同理：当前档位不支持时回落到默认（或该档位第一个）比例
  useEffect(() => {
    if (tool !== 'txt2video') return
    const tier = videoCap.tiers.find(item => item.value === videoResolution) ?? videoCap.tiers[0]
    if (tier && !tier.ratios.includes(videoAspectRatio)) {
      setVideoAspectRatio(tier.ratios.includes(videoCap.defaultRatio) ? videoCap.defaultRatio : tier.ratios[0])
    }
  }, [tool, videoCap, videoResolution, videoAspectRatio])

  const videoRatios = (videoCap.tiers.find(tier => tier.value === videoResolution) ?? videoCap.tiers[0])?.ratios ?? []

  // Image-to-image params
  const [strength, setStrength] = useState('0.8')
  const [customPrompt, setCustomPrompt] = useState('')
  const [referenceImageBase64, setReferenceImageBase64] = useState('')
  const [referenceImageName, setReferenceImageName] = useState('')

  // Eval config
  const [promptDataset, setPromptDataset] = useState('drawbench')
  const [promptLimit, setPromptLimit] = useState('1')
  const [randomPrompt, setRandomPrompt] = useState(false)
  const [metrics, setMetrics] = useState<string[]>(['clip_score'])
  const [customDatasetPath, setCustomDatasetPath] = useState('')

  const [errors, setErrors] = useState<Record<string, string>>({})

  const handleReferenceImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) {
      setReferenceImageBase64('')
      setReferenceImageName('')
      setMetrics(previous => previous.filter(metric => metric !== 'lpips'))
      return
    }
    setReferenceImageName(file.name)
    const reader = new FileReader()
    reader.onload = () => {
      setReferenceImageBase64(reader.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleSubmit = (e: SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault()
    const newErrors: Record<string, string> = {}

    if (!model.trim()) newErrors.model = 'Required'
    if (!apiBase.trim()) newErrors.apiBase = 'Required'
    if (!apiKey.trim()) newErrors.apiKey = 'Required'

    if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return }
    setErrors({})

    const modelConfig: Record<string, unknown> = {
      model_name_or_path: model.trim(),
      model_type: tool,
      model_source: 'api',
      api_base: apiBase.trim(),
      provider,
    }
    if (provider === 'volcengine' && tool !== 'txt2video') {
      newErrors.provider = '火山引擎当前仅支持文生视频'
      setErrors(newErrors)
      return
    }
    // 百炼地址必须选 DashScope 提供商，否则必 404（见 needsDashscopeProvider 注释）
    if (tool === 'txt2video' && needsDashscopeProvider(provider, apiBase)) {
      newErrors.provider = '百炼文生视频请把「API 提供商」选为 DashScope (百炼)：OpenAI 兼容协议会请求 /video/generations，百炼无此端点（必 404）'
      setErrors(newErrors)
      return
    }
    // compatible-mode 是 OpenAI 兼容入口，没有视频合成端点（见 compatibleModeVideoBase 注释）
    if (tool === 'txt2video' && compatibleModeVideoBase(apiBase)) {
      newErrors.apiBase = '文生视频请把 api_base 改成 {你的域名}/api/v1：compatible-mode 是 OpenAI 兼容入口，无视频合成端点（必 404）'
      setErrors(newErrors)
      return
    }
    if (apiKey.trim()) modelConfig.api_key = apiKey.trim()

    const cap = videoCapability(provider, model)
    if (tool === 'txt2video' && !cap.known) {
      newErrors.model = provider === 'dashscope'
        ? '未识别的 DashScope 视频模型（支持 wan2.7/2.6/2.5/2.2/2.1-t2v-*、kling-v3 系列）'
        : '未识别的火山引擎视频模型（支持 doubao-seedance-2-5 / 2-0 系列 / 1-0 系列）'
      setErrors(newErrors)
      return
    }

    const generateConfig: Record<string, unknown> = {
      width: Number(width) || 512,
      height: Number(height) || 512,
      num_inference_steps: Number(steps) || 50,
      guidance_scale: Number(guidance) || 7.5,

      seed: Number(seed) || 42,
    }

    if (tool === 'txt2video') {
      const size = resolveVideoSize(videoResolution, videoAspectRatio)
      generateConfig.width = size.width
      generateConfig.height = size.height
      generateConfig.resolution = videoResolution
      generateConfig.ratio = videoAspectRatio
      generateConfig.duration = videoDurationValue === '' ? 5 : Number(videoDurationValue)
      // 模型不支持 seed 时不发送（避免调用方以为种子生效）
      if (!cap.seed) delete generateConfig.seed
      delete generateConfig.num_inference_steps
      delete generateConfig.guidance_scale
    }
    if (tool === 'img2img') {
      generateConfig.strength = Number(strength) || 0.8
    }

    const evalConfig: Record<string, unknown> = {
      metrics,
      prompt_dataset: promptDataset,
      prompt_limit: Number(promptLimit) || 1,
      random_prompt: randomPrompt,
    }

    if (referenceImageBase64) {
      evalConfig.reference_image = referenceImageBase64
    }
    if (customPrompt.trim()) {
      evalConfig.custom_prompt = customPrompt.trim()
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


      {/* Model Configuration */}
      <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2">
        {t('aigc.modelConfig')}
      </h4>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">

        <>
          <FormField label="API 提供商" className="md:col-span-2">
            <select value={provider} onChange={e => {
              const next = e.target.value as typeof provider
              setProvider(next)
              if (errors.provider) setErrors(p => ({ ...p, provider: '' }))
              if (next === 'volcengine' && tool !== 'txt2video') setTool('txt2video')
            }} className={FORM_INPUT_CLASS}>
              {PROVIDER_OPTIONS.map(p => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
            {errors.provider && <span className="text-xs text-[var(--error)] mt-1">{errors.provider}</span>}
          </FormField>

          <FormField label={t('eval.modelName')} required error={errors.model} className="md:col-span-2">
            <input value={model}
              onChange={e => { setModel(e.target.value.trimStart()); if (errors.model) setErrors(p => ({ ...p, model: '' })) }}
              className={FORM_INPUT_CLASS} placeholder={currentProvider.modelPlaceholder} />
          </FormField>

          <FormField label={t('eval.apiUrl')} required error={errors.apiBase} className="md:col-span-2">
            <input value={apiBase}
              onChange={e => {
                setApiBase(e.target.value)
                if (errors.apiBase || errors.provider) setErrors(p => ({ ...p, apiBase: '', provider: '' }))
              }}
              className={FORM_INPUT_CLASS} placeholder={currentProvider.apiPlaceholder} />
          </FormField>

          <FormField label={t('eval.apiKey')} required error={errors.apiKey} className="md:col-span-2">
            <input type="password" value={apiKey}
              onChange={e => { setApiKey(e.target.value); if (errors.apiKey) setErrors(p => ({ ...p, apiKey: '' })) }}
              className={FORM_INPUT_CLASS} placeholder="sk-..." />
          </FormField>
        </>
      </div>

      {/* Generation Parameters */}
      <h4 className="text-sm font-medium text-[var(--text)] border-b border-[var(--border-md)] pb-2 pt-4">
        {t('aigc.generateConfig')}
      </h4>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {tool === 'txt2video' ? (<>
          <FormField label="分辨率">
            <select value={videoResolution} onChange={e => setVideoResolution(e.target.value)} className={FORM_INPUT_CLASS}>
              {videoCap.tiers.map(tier => <option key={tier.value} value={tier.value}>{tier.label}</option>)}
            </select>
          </FormField>
          <FormField label="画幅比">
            <select value={videoAspectRatio} onChange={e => setVideoAspectRatio(e.target.value)} className={FORM_INPUT_CLASS}>
              {videoRatios.map(value => <option key={value} value={value}>{RATIO_LABELS[value] ?? value}</option>)}
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

        {tool !== 'txt2video' && <FormField label={t('aigc.steps')}>
          <input type="number" value={steps}
            onChange={e => setSteps(e.target.value.replace(/[^0-9]/g, ''))}
            className={FORM_INPUT_CLASS} placeholder="50" />
        </FormField>}

        {tool !== 'txt2video' && <FormField label={t('aigc.guidance')}>
          <input type="number" value={guidance}
            onChange={e => setGuidance(e.target.value)}
            className={FORM_INPUT_CLASS} placeholder="7.5" step="0.1" />
        </FormField>}

        {tool === 'txt2video' && <FormField label="视频时长" hint={videoCap.durationHint}>
          {durationOptions.length > 0 ? (
            <select value={videoDurationValue} onChange={e => setVideoDuration(e.target.value)}
              className={FORM_INPUT_CLASS}>
              {durationOptions.map(value => <option key={value} value={value}>{value} 秒</option>)}
            </select>
          ) : (
            <input type="number" value={videoDuration}
              onChange={e => setVideoDuration(e.target.value.replace(/[^0-9]/g, ''))}
              className={FORM_INPUT_CLASS} step="1" placeholder="5" />
          )}
        </FormField>}

        {(tool !== 'txt2video' || videoCap.seed) ? <FormField label={t('aigc.seed')}>
          <input type="number" value={seed}
            onChange={e => setSeed(e.target.value.replace(/[^0-9]/g, ''))}
            className={FORM_INPUT_CLASS} placeholder="42" />
        </FormField> : null}

        {tool === 'img2img' && (
          <>
            <FormField label={t('aigc.strength')} hint="0-1，越大越偏离原图">
              <input type="number" value={strength}
                onChange={e => setStrength(e.target.value)}
                className={FORM_INPUT_CLASS} placeholder="0.8" step="0.05" min="0" max="1" />
            </FormField>
            <FormField label="参考图" hint="选一张图片作为 img2img 的输入">
              <input
                type="file"
                accept="image/*"
                onChange={handleReferenceImageChange}
                className={FORM_INPUT_CLASS + ' file:mr-3 file:border-0 file:bg-[var(--surface)] file:text-[var(--text)] file:px-3 file:py-1 file:rounded file:cursor-pointer text-sm'}
              />
              {referenceImageName && (
                <span className="text-xs text-[var(--text-secondary)] mt-1 block">
                  已选择: {referenceImageName}
                </span>
              )}
            </FormField>
          </>
        )}
      </div>

      {tool === 'img2img' ? (
        <FormField label="自定义提示词" hint="描述你希望对参考图做的修改（必填）">
          <textarea value={customPrompt}
            onChange={e => setCustomPrompt(e.target.value)}
            className={FORM_INPUT_CLASS + ' h-20 resize-none'}
            placeholder="例如：把背景换成海边日落，保持人物不变..." />
        </FormField>
      ) : (
        <FormField label="自定义提示词（可选）" hint="留空则使用下方选择的数据集">
          <textarea value={customPrompt}
            onChange={e => setCustomPrompt(e.target.value)}
            className={FORM_INPUT_CLASS + ' h-20 resize-none'}
            placeholder={tool === 'txt2video' ? '例如：一架无人机飞过雪山湖泊上空，航拍视角...' : '例如：一只小狗坐在窗台上看着夕阳，温暖的色调...'} />
        </FormField>
      )}


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
        <div className="flex items-center gap-3">
          <input type="number" value={promptLimit}
            onChange={e => setPromptLimit(e.target.value.replace(/[^0-9]/g, ''))}
            className={FORM_INPUT_CLASS + ' flex-1'} placeholder="100" />
          <label className="flex items-center gap-1.5 text-sm text-[var(--text)] whitespace-nowrap cursor-pointer">
            <input type="checkbox" checked={randomPrompt}
              onChange={e => setRandomPrompt(e.target.checked)}
              className="accent-[var(--accent)]" />
            随机抽取
          </label>
        </div>
      </FormField>

      <FormField label={t('aigc.metrics')}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {(tool === 'txt2video'
            ? VIDEO_METRICS
            : tool === 'img2img' && referenceImageBase64
              ? IMAGE_TO_IMAGE_METRICS
              : TEXT_TO_IMAGE_METRICS).map(m => {
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

      <Button type="submit" variant="primary" disabled={disabled} className="btn-glow !mt-6">
        {t('aigc.startEval')}
      </Button>
    </form>
  )
}
