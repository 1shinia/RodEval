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

/**
 * 创建到点自动 abort 的 AbortSignal，并附带 didTimeout 探针。
 *
 * didTimeout 用来区分「我方超时掐断」与「调用方主动取消」：否则超时会让
 * 浏览器抛出原生 AbortError，其文案（Chrome 为 "The user aborted a
 * request."）既误导用户（他并未取消任何东西），也说不清是哪个接口慢。
 */
function createAbortSignal(timeoutMs: number = DEFAULT_TIMEOUT): {
  signal: AbortSignal
  didTimeout: () => boolean
} {
  const controller = new AbortController()
  let timedOut = false
  if (timeoutMs > 0) {
    const timer = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, timeoutMs)
    // Allow the timer to not block process exit (browser: no-op, but safe)
    if (typeof (timer as any).unref === 'function') (timer as any).unref()
  }
  return { signal: controller.signal, didTimeout: () => timedOut }
}

/**
 * fetch 包装：超时未响应时抛出可定位的中文错误，而非浏览器原生 AbortError。
 *
 * timeoutMs 语义与原实现一致：0 = 不超时（评估/压测等长任务提交走这条），
 * undefined = 默认 30 秒。
 */
async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number | undefined,
  label: string,
): Promise<Response> {
  const effective = timeoutMs === undefined ? DEFAULT_TIMEOUT : timeoutMs
  const { signal, didTimeout } = createAbortSignal(effective)
  try {
    return await fetch(url, { ...init, signal })
  } catch (e) {
    // 只接管我们自己超时的那一种；调用方主动取消仍原样抛出。
    if (didTimeout()) {
      // 不足 1 秒时用毫秒，避免 Math.round 把 500ms 显示成「0 秒」。
      const waited = effective >= 1000 ? `${Math.round(effective / 1000)} 秒` : `${effective} 毫秒`
      throw new Error(`请求超时（${waited}未响应）：${label}`, { cause: e })
    }
    throw e
  }
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
  const res = await fetchWithTimeout(
    url.toString(),
    { cache: 'no-store', headers: getAuthHeaders() },
    timeoutMs,
    path,
  )
  return handleResponse<T>(res)
}

export async function apiPost<T = unknown>(
  path: string,
  body: unknown,
  headers?: Record<string, string>,
  timeoutMs?: number,
): Promise<T> {
  const res = await fetchWithTimeout(
    path,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeaders(), ...headers },
      body: JSON.stringify(body),
    },
    timeoutMs,
    path,
  )
  return handleResponse<T>(res)
}

export async function apiDelete<T = unknown>(
  path: string,
  body?: unknown,
  timeoutMs?: number,
): Promise<T> {
  const res = await fetchWithTimeout(
    path,
    {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
      body: body ? JSON.stringify(body) : undefined,
    },
    timeoutMs,
    path,
  )
  return handleResponse<T>(res)
}
