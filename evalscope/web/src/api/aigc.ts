/**
 * AIGC evaluation API client
 */
import type { AIGCEvalConfig, AIGCGenerateConfig, AIGCModelConfig, AIGCTaskResult } from './types';

const API_BASE = '/api/v1/aigc';

export interface AIGCInvokeRequest {
  tool: 'txt2img' | 'txt2video' | 'img2img';
  model: AIGCModelConfig;
  generate: AIGCGenerateConfig;
  eval: AIGCEvalConfig;
}

export interface AIGCProgress {
  current: number;
  total: number;
  percent: number;
  status: 'running' | 'completed' | 'failed' | 'stopped';
  error?: string;
}

export interface AIGCBenchmark {
  name: string;
  description: string;
  prompts: number;
}

/**
 * Submit AIGC evaluation task
 */
export async function invokeAIGCEvaluation(
  taskId: string,
  request: AIGCInvokeRequest
): Promise<AIGCTaskResult> {
  const response = await fetch(`${API_BASE}/invoke`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'EvalScope-Task-Id': taskId,
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'AIGC evaluation failed');
  }

  return response.json();
}

/**
 * Get AIGC evaluation progress
 */
export async function getAIGCProgress(taskId: string): Promise<AIGCProgress> {
  const response = await fetch(`${API_BASE}/progress?task_id=${taskId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get progress');
  }

  return response.json();
}

/**
 * Stop AIGC evaluation task
 */
export async function stopAIGCEvaluation(taskId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/stop`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ task_id: taskId }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to stop task');
  }
}

/**
 * Get list of AIGC benchmarks
 */
export async function getAIGCBenchmarks(): Promise<AIGCBenchmark[]> {
  const response = await fetch(`${API_BASE}/benchmarks`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get benchmarks');
  }

  return response.json();
}

/**
 * Get media file URL
 */
export function getMediaUrl(taskId: string, filename: string): string {
  return `${API_BASE}/media/${taskId}/${filename}`;
}

/**
 * Get thumbnail URL
 */
export function getThumbnailUrl(taskId: string, filename: string): string {
  return `${API_BASE}/thumbnails/${taskId}/${filename}`;
}
