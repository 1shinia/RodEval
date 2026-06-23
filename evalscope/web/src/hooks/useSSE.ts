import { useEffect, useRef, useState } from 'react'

interface UseSSEOptions<T> {
  url: string | null
  enabled?: boolean
  onData?: (data: T) => void
  onError?: (error: string) => void
  /** Initial reconnect delay in ms (default 1000) */
  reconnectDelay?: number
  /** Maximum reconnect delay in ms (default 30000) */
  maxReconnectDelay?: number
}

/**
 * Hook for consuming Server-Sent Events (SSE) streams.
 * Replaces polling with real-time push updates.
 * Implements exponential backoff reconnection on connection loss.
 */
export function useSSE<T>({
  url,
  enabled = false,
  onData,
  onError,
  reconnectDelay = 1000,
  maxReconnectDelay = 30_000,
}: UseSSEOptions<T>) {
  const [error, setError] = useState<string | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const mountedRef = useRef(true)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const currentDelayRef = useRef(reconnectDelay)

  // Use refs so that changes to callbacks do NOT restart the connection
  const onDataRef = useRef(onData)
  onDataRef.current = onData
  const onErrorRef = useRef(onError)
  onErrorRef.current = onError

  useEffect(() => {
    mountedRef.current = true

    const cleanup = () => {
      mountedRef.current = false
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
    }

    if (!enabled || !url) {
      cleanup()
      return
    }

    const connect = () => {
      if (!mountedRef.current) return

      const es = new EventSource(url)
      eventSourceRef.current = es

      es.onopen = () => {
        if (!mountedRef.current) return
        // Connection successful — reset backoff delay
        currentDelayRef.current = reconnectDelay
        setError(null)
      }

      es.onmessage = (event) => {
        if (!mountedRef.current) return
        try {
          const data = JSON.parse(event.data) as T
          setError(null)
          onDataRef.current?.(data)
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e)
          setError(msg)
          onErrorRef.current?.(msg)
        }
      }

      es.onerror = () => {
        if (!mountedRef.current) return

        // Close the broken connection
        es.close()
        eventSourceRef.current = null

        const delay = currentDelayRef.current
        setError(`连接断开，${Math.round(delay / 1000)}秒后重连...`)
        onErrorRef.current?.('Connection lost')

        // Schedule reconnect with exponential backoff
        reconnectTimerRef.current = setTimeout(() => {
          reconnectTimerRef.current = null
          // Increase delay for next attempt (exponential backoff, capped)
          currentDelayRef.current = Math.min(delay * 2, maxReconnectDelay)
          connect()
        }, delay)
      }
    }

    connect()

    return cleanup
  }, [url, enabled, reconnectDelay, maxReconnectDelay])

  return { error }
}
