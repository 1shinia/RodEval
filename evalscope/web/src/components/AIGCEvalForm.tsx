/**
 * AIGC evaluation configuration form
 */
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { invokeAIGCEvaluation, type AIGCInvokeRequest } from '../api/aigc';
import type { AIGCTaskResult } from '../api/types';

interface AIGCEvalFormProps {
  onResult: (result: AIGCTaskResult) => void;
}

export function AIGCEvalForm({ onResult }: AIGCEvalFormProps) {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Model config
  const [modelPath, setModelPath] = useState('');
  const [modelType, setModelType] = useState<'txt2img' | 'txt2video' | 'img2img'>('txt2img');
  const [apiBase, setApiBase] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [device, setDevice] = useState('cuda');
  const [dtype, setDtype] = useState('float16');

  // Generate config
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [numInferenceSteps, setNumInferenceSteps] = useState(50);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [negativePrompt, setNegativePrompt] = useState('');
  const [seed, setSeed] = useState(42);
  const [batchSize, setBatchSize] = useState(1);

  // Eval config
  const [metrics, setMetrics] = useState<string[]>(['clip_score']);
  const [promptDataset, setPromptDataset] = useState('drawbench');
  const [promptLimit, setPromptLimit] = useState(100);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const taskId = `aigc_${Date.now()}`;
      const request: AIGCInvokeRequest = {
        tool: modelType,
        model: {
          model_name_or_path: modelPath,
          model_type: modelType,
          api_base: apiBase || undefined,
          api_key: apiKey || undefined,
          device,
          dtype,
        },
        generate: {
          width,
          height,
          num_inference_steps: numInferenceSteps,
          guidance_scale: guidanceScale,
          negative_prompt: negativePrompt || undefined,
          seed,
          batch_size: batchSize,
        },
        eval: {
          metrics,
          prompt_dataset: promptDataset,
          prompt_limit: promptLimit,
        },
      };

      const result = await invokeAIGCEvaluation(taskId, request);
      onResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : '评估失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t('aigc.config', '评估配置')}</CardTitle>
        <CardDescription>{t('aigc.configDesc', '配置 AIGC 模型和评估参数')}</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Model Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">{t('aigc.modelConfig', '模型配置')}</h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="modelPath">{t('aigc.modelPath', '模型路径')}</Label>
                <Input
                  id="modelPath"
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                  placeholder="stabilityai/stable-diffusion-2-1"
                  required
                />
              </div>

              <div>
                <Label htmlFor="modelType">{t('aigc.modelType', '模型类型')}</Label>
                <Select value={modelType} onValueChange={(v) => setModelType(v as any)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="txt2img">文生图</SelectItem>
                    <SelectItem value="txt2video">文生视频</SelectItem>
                    <SelectItem value="img2img">图生图</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="apiBase">{t('aigc.apiBase', 'API Base (可选)')}</Label>
                <Input
                  id="apiBase"
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                  placeholder="https://api.example.com"
                />
              </div>

              <div>
                <Label htmlFor="apiKey">{t('aigc.apiKey', 'API Key (可选)')}</Label>
                <Input
                  id="apiKey"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                />
              </div>

              <div>
                <Label htmlFor="device">{t('aigc.device', '设备')}</Label>
                <Select value={device} onValueChange={setDevice}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cuda">CUDA</SelectItem>
                    <SelectItem value="cpu">CPU</SelectItem>
                    <SelectItem value="mps">MPS</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="dtype">{t('aigc.dtype', '数据类型')}</Label>
                <Select value={dtype} onValueChange={setDtype}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="float16">Float16</SelectItem>
                    <SelectItem value="float32">Float32</SelectItem>
                    <SelectItem value="bfloat16">BFloat16</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          {/* Generate Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">{t('aigc.generateConfig', '生成配置')}</h3>

            <div className="grid grid-cols-3 gap-4">
              <div>
                <Label htmlFor="width">{t('aigc.width', '宽度')}</Label>
                <Input
                  id="width"
                  type="number"
                  value={width}
                  onChange={(e) => setWidth(parseInt(e.target.value.replace(/[^0-9]/g, '')) || 512)}
                  min={64}
                  max={2048}
                />
              </div>

              <div>
                <Label htmlFor="height">{t('aigc.height', '高度')}</Label>
                <Input
                  id="height"
                  type="number"
                  value={height}
                  onChange={(e) => setHeight(parseInt(e.target.value.replace(/[^0-9]/g, '')) || 512)}
                  min={64}
                  max={2048}
                />
              </div>

              <div>
                <Label htmlFor="numInferenceSteps">{t('aigc.steps', '推理步数')}</Label>
                <Input
                  id="numInferenceSteps"
                  type="number"
                  value={numInferenceSteps}
                  onChange={(e) => setNumInferenceSteps(parseInt(e.target.value.replace(/[^0-9]/g, '')) || 50)}
                  min={1}
                  max={200}
                />
              </div>

              <div>
                <Label htmlFor="guidanceScale">{t('aigc.guidanceScale', '引导系数')}</Label>
                <Input
                  id="guidanceScale"
                  type="number"
                  value={guidanceScale}
                  onChange={(e) => setGuidanceScale(parseFloat(e.target.value) || 7.5)}
                  min={0}
                  max={20}
                  step={0.1}
                />
              </div>

              <div>
                <Label htmlFor="seed">{t('aigc.seed', '随机种子')}</Label>
                <Input
                  id="seed"
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value.replace(/[^0-9]/g, '')) || 42)}
                />
              </div>

              <div>
                <Label htmlFor="batchSize">{t('aigc.batchSize', '批次大小')}</Label>
                <Input
                  id="batchSize"
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value.replace(/[^0-9]/g, '')) || 1)}
                  min={1}
                  max={16}
                />
              </div>
            </div>

            <div>
              <Label htmlFor="negativePrompt">{t('aigc.negativePrompt', '负向提示词')}</Label>
              <Input
                id="negativePrompt"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                placeholder="low quality, blurry, distorted"
              />
            </div>
          </div>

          {/* Eval Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">{t('aigc.evalConfig', '评估配置')}</h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="promptDataset">{t('aigc.promptDataset', '提示词数据集')}</Label>
                <Select value={promptDataset} onValueChange={setPromptDataset}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="drawbench">DrawBench (200 prompts)</SelectItem>
                    <SelectItem value="coco_captions">COCO Captions</SelectItem>
                    <SelectItem value="parti">PartiPrompts</SelectItem>
                    <SelectItem value="custom">自定义</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="promptLimit">{t('aigc.promptLimit', '提示词数量限制')}</Label>
                <Input
                  id="promptLimit"
                  type="number"
                  value={promptLimit}
                  onChange={(e) => setPromptLimit(parseInt(e.target.value.replace(/[^0-9]/g, '')) || 100)}
                  min={1}
                  max={1000}
                />
              </div>
            </div>

            <div>
              <Label>{t('aigc.metrics', '评估指标')}</Label>
              <div className="flex gap-2 mt-2">
                {['clip_score', 'fid', 'inception_score'].map((metric) => (
                  <label key={metric} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={metrics.includes(metric)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setMetrics([...metrics, metric]);
                        } else {
                          setMetrics(metrics.filter((m) => m !== metric));
                        }
                      }}
                    />
                    <span className="text-sm">{metric}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-800">
              {error}
            </div>
          )}

          <Button type="submit" disabled={loading || !modelPath} className="w-full">
            {loading ? t('aigc.running', '评估中...') : t('aigc.start', '开始评估')}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
