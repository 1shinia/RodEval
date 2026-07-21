/**
 * AIGC evaluation task page
 */
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { TaskPageLayout } from '../components/TaskPageLayout';
import { AIGCEvalForm } from '../components/AIGCEvalForm';
import { MediaGallery } from '../components/MediaGallery';
import { AIGCMetricsPanel } from '../components/AIGCMetricsPanel';
import type { AIGCTaskResult } from '../api/types';

export function AIGCTaskPage() {
  const { t } = useTranslation();
  const [result, setResult] = useState<AIGCTaskResult | null>(null);

  return (
    <TaskPageLayout
      title={t('aigc.title', 'AIGC 评估')}
      description={t('aigc.description', '评估文生图、文生视频等 AIGC 模型')}
    >
      <div className="space-y-6">
        <AIGCEvalForm onResult={setResult} />

        {result && (
          <>
            <AIGCMetricsPanel metrics={result.metrics} />
            <MediaGallery images={result.images} taskId={result.task_id} />
          </>
        )}
      </div>
    </TaskPageLayout>
  );
}
