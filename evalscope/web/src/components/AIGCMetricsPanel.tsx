/**
 * AIGC metrics panel component
 */
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import type { AIGCMetricsResult } from '../api/types';

interface AIGCMetricsPanelProps {
  metrics: AIGCMetricsResult;
}

export function AIGCMetricsPanel({ metrics }: AIGCMetricsPanelProps) {
  const { t } = useTranslation();

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t('aigc.metricsResult', '评估结果')}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="space-y-1">
            <div className="text-sm text-muted-foreground">{t('aigc.totalImages', '总图片数')}</div>
            <div className="text-2xl font-bold">{metrics.total_images}</div>
          </div>

          {metrics.clip_score_mean !== undefined && (
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">{t('aigc.clipScoreMean', 'CLIP Score 均值')}</div>
              <div className="text-2xl font-bold">{metrics.clip_score_mean.toFixed(4)}</div>
              {metrics.clip_score_std !== undefined && (
                <div className="text-xs text-muted-foreground">
                  ±{metrics.clip_score_std.toFixed(4)}
                </div>
              )}
            </div>
          )}

          {metrics.fid !== undefined && (
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">{t('aigc.fid', 'FID')}</div>
              <div className="text-2xl font-bold">{metrics.fid.toFixed(2)}</div>
            </div>
          )}

          {metrics.inception_score !== undefined && (
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">{t('aigc.inceptionScore', 'Inception Score')}</div>
              <div className="text-2xl font-bold">{metrics.inception_score.toFixed(2)}</div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
