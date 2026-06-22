import { useEffect, useRef, useState } from 'react'

interface UseSSEOptions<T> {
  url: string | null
  enabled?: boolean
  onData?: (data: T) => void
  onError?: (error: string) => void
  reconnectDelay?: number
}

/**
 * Hook for consuming Server-Sent Events (SSE) streams.
 * Replaces polling with real-time push updates.
 */
export function useSSE<T>({ url, enabled = false, onData, onError, reconnectDelay = 3000 }: UseSSEOptions<T>) {
  const [error, setError] = useState<string | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const mountedRef = useRef(true)

  // Use refs so that changes to callbacks do NOT restart the connection
  const onDataRef = useRef(onData)
  onDataRef.current = onData
  const onErrorRef = useRef(onError)
  onErrorRef.current = onError

  useEffect(() => {
    mountedRef.current = true

    if (!enabled || !url) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
      return
    }

    const connect = () => {
      if (!mountedRef.current) return

      const es = new EventSource(url)
      eventSourceRef.current = es

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
        // EventSource will auto-reconnect, but we can add custom logic here
        setError('Connection lost, reconnecting...')
        onErrorRef.current?.('Connection lost')
      }
    }

    connect()

    return () => {
      mountedRef.current = false
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
    }
  }, [url, enabled, reconnectDelay])

  return { error }
}
