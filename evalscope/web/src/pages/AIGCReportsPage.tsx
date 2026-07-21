/**
 * AIGC reports list page
 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table';

interface AIGCReportSummary {
  task_id: string;
  model_name: string;
  model_type: string;
  total_images: number;
  clip_score_mean?: number;
  fid?: number;
  inception_score?: number;
  created_at: string;
}

export default function AIGCReportsPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [reports, setReports] = useState<AIGCReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadReports = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/v1/aigc/reports');
        if (!response.ok) {
          throw new Error(`Failed to load reports: ${response.statusText}`);
        }
        const data = await response.json();
        setReports(data.reports || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load reports');
      } finally {
        setLoading(false);
      }
    };

    loadReports();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">{t('common.loading', 'Loading...')}</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="text-center text-red-600">
            <p className="text-lg font-semibold">{t('common.error', 'Error')}</p>
            <p className="mt-2">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{t('aigc.reportsList', 'AIGC 评估报告列表')}</h1>
        <p className="text-muted-foreground mt-1">
          {t('aigc.reportsListDesc', '查看所有 AIGC 评估任务的历史报告')}
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>{t('aigc.allReports', '所有报告')}</CardTitle>
        </CardHeader>
        <CardContent>
          {reports.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <p>{t('aigc.noReports', '暂无 AIGC 评估报告')}</p>
              <p className="text-sm mt-2">{t('aigc.noReportsHint', '运行 AIGC 评估任务后，报告将显示在这里')}</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t('aigc.taskId', '任务 ID')}</TableHead>
                  <TableHead>{t('aigc.modelName', '模型名称')}</TableHead>
                  <TableHead>{t('aigc.modelType', '模型类型')}</TableHead>
                  <TableHead className="text-right">{t('aigc.totalImages', '总图片数')}</TableHead>
                  <TableHead className="text-right">{t('aigc.clipScoreMean', 'CLIP Score 均值')}</TableHead>
                  <TableHead className="text-right">{t('aigc.fid', 'FID')}</TableHead>
                  <TableHead>{t('aigc.createdAt', '创建时间')}</TableHead>
                  <TableHead className="text-right">{t('common.actions', '操作')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {reports.map((report) => (
                  <TableRow key={report.task_id}>
                    <TableCell className="font-mono text-sm">{report.task_id}</TableCell>
                    <TableCell>{report.model_name}</TableCell>
                    <TableCell>{report.model_type}</TableCell>
                    <TableCell className="text-right">{report.total_images}</TableCell>
                    <TableCell className="text-right">
                      {report.clip_score_mean !== undefined ? (
                        <span className="font-mono">{report.clip_score_mean.toFixed(4)}</span>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {report.fid !== undefined ? (
                        <span className="font-mono">{report.fid.toFixed(2)}</span>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {new Date(report.created_at).toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => navigate(`/aigc-reports/${report.task_id}`)}
                      >
                        {t('common.view', '查看')}
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
