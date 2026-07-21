/**
 * Media gallery component for displaying generated images
 */
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import type { AIGCGeneratedImage } from '../api/types';

interface MediaGalleryProps {
  images: AIGCGeneratedImage[];
  taskId: string;
}

export function MediaGallery({ images, taskId }: MediaGalleryProps) {
  const { t } = useTranslation();
  const [selectedImage, setSelectedImage] = useState<AIGCGeneratedImage | null>(null);

  if (images.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t('aigc.gallery', '生成图片画廊')}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {images.map((image, index) => (
            <Dialog key={index} open={selectedImage?.filename === image.filename} onOpenChange={(open) => {
              if (!open) setSelectedImage(null);
            }}>
              <DialogTrigger asChild>
                <div
                  className="cursor-pointer group relative aspect-square overflow-hidden rounded-lg border hover:border-primary transition-colors"
                  onClick={() => setSelectedImage(image)}
                >
                  <img
                    src={image.thumbnail_url}
                    alt={image.prompt}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                  />
                  {image.clip_score !== undefined && (
                    <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white text-xs p-2">
                      CLIP: {image.clip_score.toFixed(2)}
                    </div>
                  )}
                </div>
              </DialogTrigger>
              <DialogContent className="max-w-4xl">
                <DialogHeader>
                  <DialogTitle>{t('aigc.imageDetail', '图片详情')}</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <img
                    src={image.url}
                    alt={image.prompt}
                    className="w-full rounded-lg"
                  />
                  <div className="space-y-2">
                    <div>
                      <span className="font-semibold">{t('aigc.prompt', '提示词')}:</span>
                      <p className="text-sm text-muted-foreground mt-1">{image.prompt}</p>
                    </div>
                    {image.clip_score !== undefined && (
                      <div>
                        <span className="font-semibold">{t('aigc.clipScore', 'CLIP Score')}:</span>
                        <span className="ml-2">{image.clip_score.toFixed(4)}</span>
                      </div>
                    )}
                    <div>
                      <span className="font-semibold">{t('aigc.filename', '文件名')}:</span>
                      <span className="ml-2 text-sm text-muted-foreground">{image.filename}</span>
                    </div>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
