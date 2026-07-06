const DEFAULT_TIMEOUT = 30_000 // 30 seconds

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
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
  const res = await fetch(url.toString(), { signal: createAbortSignal(timeoutMs) })
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
    headers: { 'Content-Type': 'application/json', ...headers },
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
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
    signal: createAbortSignal(timeoutMs),
  })
  return handleResponse<T>(res)
}
