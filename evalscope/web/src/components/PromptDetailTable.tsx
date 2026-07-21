/**
 * Prompt detail table component
 */
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import type { AIGCGeneratedImage } from '../api/types';

interface PromptDetailTableProps {
  images: AIGCGeneratedImage[];
}

export function PromptDetailTable({ images }: PromptDetailTableProps) {
  const { t } = useTranslation();

  if (images.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t('aigc.promptDetails', '提示词详情')}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="max-h-96 overflow-y-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-16">#</TableHead>
                <TableHead>{t('aigc.prompt', '提示词')}</TableHead>
                <TableHead className="w-32">{t('aigc.clipScore', 'CLIP Score')}</TableHead>
                <TableHead className="w-48">{t('aigc.filename', '文件名')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {images.map((image, index) => (
                <TableRow key={index}>
                  <TableCell className="font-medium">{index + 1}</TableCell>
                  <TableCell className="max-w-md truncate">{image.prompt}</TableCell>
                  <TableCell>
                    {image.clip_score !== undefined ? (
                      <span className="font-mono">{image.clip_score.toFixed(4)}</span>
                    ) : (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground font-mono">
                    {image.filename}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
