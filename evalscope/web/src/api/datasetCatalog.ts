import { getAuthHeaders } from './client'

export interface CatalogEntry {
  id: number
  fingerprint: string
  benchmark: string
  category: 'llm' | 'vlm' | 'rag' | 'agent'
  source: string
  dataset_id: string
  revision?: string
  subset?: string
  split?: string
  cache_path?: string
  status: string
  published: boolean
  details?: { manual_review?: { passed: boolean; user_id: number; at: string }; automatic_validation?: { reason?: string; validator?: string }; reason?: string; rows?: number; sample_count?: number; [key: string]: unknown }
}

const root = '/api/v1/dataset-catalog'

export interface CatalogDefinition {
  benchmark: string
  category: CatalogEntry['category']
  dataset_id: string
  task_type?: string
  validation_supported: boolean
  manual_review?: { passed: boolean; user_id: number; at: string } | null
  historical_test?: { samples: number; timestamp: number; report: string } | null
  status: string
  published: boolean
  versions: CatalogEntry[]
  dependencies?: CatalogEntry[]
}

export type PublishedDataset = Pick<CatalogDefinition, 'benchmark' | 'category' | 'dataset_id' | 'task_type' | 'manual_review' | 'published'>
export const listPublishedDatasets = () => catalogRequest<{ items: PublishedDataset[] }>(`${root}/published-datasets`, undefined, 90_000)
export const publishDataset = (category: CatalogEntry['category'], benchmark: string, published: boolean) => catalogRequest<{ category: CatalogEntry['category']; benchmark: string; published: boolean; manual_review: NonNullable<CatalogDefinition['manual_review']> }>(`${root}/inventory/publish`, { category, benchmark, published })
export const listInventory = () => catalogRequest<{ items: CatalogDefinition[]; unmapped: CatalogEntry[]; total: number }>(`${root}/inventory`, undefined, 90_000)

async function catalogRequest<T>(path: string, body?: unknown, timeout = 30_000): Promise<T> {
  const controller = new AbortController()
  const timer = timeout > 0 ? window.setTimeout(() => controller.abort(), timeout) : undefined
  try {
    const response = await fetch(path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: { ...getAuthHeaders(), ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: 'no-store',
      signal: controller.signal,
    })
    if (response.status === 401) {
      localStorage.removeItem('evalscope_token')
      localStorage.removeItem('evalscope_user')
      if (!window.location.pathname.startsWith('/login')) window.location.href = '/login'
    }
    const text = await response.text()
    if (response.headers.get('content-type')?.includes('text/html') || /^\s*</.test(text)) {
      throw new Error(`数据集目录接口未就绪 / Dataset catalog API is not ready (HTTP ${response.status}): ${path}`)
    }
    let data: T & { error?: string }
    try { data = JSON.parse(text) } catch {
      throw new Error(`数据集目录接口返回无效 JSON / Invalid catalog response (HTTP ${response.status}): ${path}`)
    }
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}: ${path}`)
    return data
  } catch (error) {
    if (controller.signal.aborted) throw new Error(`数据集目录请求超时 / Catalog request timed out: ${path}`)
    throw error
  } finally {
    if (timer !== undefined) window.clearTimeout(timer)
  }
}

export const listCatalog = (admin = false) => catalogRequest<{ items: CatalogEntry[] }>(`${root}${admin ? '/admin' : ''}`)
export const scanCatalog = () => catalogRequest<{ items: CatalogEntry[] }>(`${root}/scan`, {}, 0)
export const validateCatalog = (id: number) => catalogRequest(`${root}/${id}/validate`, {}, 0)
export const reviewDataset = (category: CatalogEntry['category'], benchmark: string, passed: boolean) => catalogRequest<{ category: CatalogEntry['category']; benchmark: string; manual_review: NonNullable<CatalogDefinition['manual_review']> }>(`${root}/inventory/manual-review`, { category, benchmark, passed })
export const reviewCatalog = (id: number, passed: boolean) => catalogRequest(`${root}/${id}/manual-review`, { passed }, 0)
export const publishCatalog = (id: number, published: boolean) => catalogRequest(`${root}/${id}/publish`, { published })
