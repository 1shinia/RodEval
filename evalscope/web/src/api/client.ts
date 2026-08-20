const DEFAULT_TIMEOUT = 30_000 // 30 seconds

export function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem('evalscope_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    // 401 = 会话失效（token 过期/被吊销/无效）。登录请求走 AuthContext 的
    // 原生 fetch，不经过这里，所以此处 401 一律视为会话死亡：
    // 清掉本地会话并跳回登录页。
    if (res.status === 401) {
      localStorage.removeItem('evalscope_token')
      localStorage.removeItem('evalscope_user')
      if (!window.location.pathname.startsWith('/login')) {
        window.location.href = '/login'
      }
    }
    const body = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(body.error || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

function createAbortSignal(timeoutMs: number = DEFAULT_TIMEOUT): AbortSignal {
  const controller = new AbortController()
  if (timeoutMs > 0) {
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    // Allow the timer to not block process exit (browser: no-op, but safe)
    if (typeof (timer as any).unref === 'function') (timer as any).unref()
  }
  return controller.signal
}

export async function api<T = unknown>(
  path: string,
  params?: Record<string, unknown>,
  timeoutMs?: number,
): Promise<T> {
  const url = new URL(path, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== '') url.searchParams.set(k, String(v))
    }
  }
  const res = await fetch(url.toString(), { signal: createAbortSignal(timeoutMs), cache: 'no-store', headers: getAuthHeaders() })
  return handleResponse<T>(res)
}

export async function apiPost<T = unknown>(
  path: string,
  body: unknown,
  headers?: Record<string, string>,
  timeoutMs?: number,
): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders(), ...headers },
    body: JSON.stringify(body),
    signal: createAbortSignal(timeoutMs),
  })
  return handleResponse<T>(res)
}

export async function apiDelete<T = unknown>(
  path: string,
  body?: unknown,
  timeoutMs?: number,
): Promise<T> {
  const res = await fetch(path, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: body ? JSON.stringify(body) : undefined,
    signal: createAbortSignal(timeoutMs),
  })
  return handleResponse<T>(res)
}
