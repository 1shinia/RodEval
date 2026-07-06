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

interface ConnectionState {
  status: 'disconnected' | 'connecting' | 'connected' | 'reconnecting'
  message: string
}

/**
 * Hook for consuming Server-Sent Events (SSE) streams.
 * Replaces polling with real-time push updates.
 * Implements exponential backoff reconnection on connection loss
 * with last_pos-based resume to avoid log duplication.
 */
export function useSSE<T>({
  url,
  enabled = false,
  onData,
  onError,
  reconnectDelay = 1000,
  maxReconnectDelay = 30_000,
}: UseSSEOptions<T>) {
  const [connectionState, setConnectionState] = useState<ConnectionState>({
    status: 'disconnected',
    message: '',
  })
  const eventSourceRef = useRef<EventSource | null>(null)
  const mountedRef = useRef(true)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const currentDelayRef = useRef(reconnectDelay)
  const lastPosRef = useRef<number>(0)        // Track log position for resume
  const isReconnectRef = useRef(false)         // True when this is a reconnection

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
      setConnectionState({ status: 'disconnected', message: '' })
      return
    }

    const connect = () => {
      if (!mountedRef.current) return

      const isReconnect = isReconnectRef.current
      isReconnectRef.current = false

      // Build URL with last_pos for resume on reconnect
      let connectUrl = url
      if (isReconnect && lastPosRef.current > 0) {
        const sep = connectUrl.includes('?') ? '&' : '?'
        connectUrl = `${connectUrl}${sep}last_pos=${lastPosRef.current}`
      }

      if (!isReconnect) {
        lastPosRef.current = 0
        setConnectionState({ status: 'connecting', message: '正在连接日志...' })
      } else {
        setConnectionState({ status: 'reconnecting', message: `连接断开，${Math.round(currentDelayRef.current / 1000)}秒后重连...` })
      }

      const es = new EventSource(connectUrl)
      eventSourceRef.current = es

      es.onopen = () => {
        if (!mountedRef.current) return
        currentDelayRef.current = reconnectDelay
        setConnectionState({ status: 'connected', message: '' })
      }

      es.onmessage = (event) => {
        if (!mountedRef.current) return
        try {
          const data = JSON.parse(event.data) as T & { pos?: number; event?: string }
          // Timeout sent by backend
          if (data.event === 'timeout') {
            setConnectionState({ status: 'connected', message: '日志流空闲超时' })
            return
          }
          // Track position for resume
          if (typeof data.pos === 'number') {
            lastPosRef.current = data.pos
          }
          setConnectionState({ status: 'connected', message: '' })
          onDataRef.current?.(data)
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e)
          onErrorRef.current?.(msg)
        }
      }

      es.onerror = () => {
        if (!mountedRef.current) return

        // Close the broken connection
        es.close()
        eventSourceRef.current = null

        const delay = currentDelayRef.current
        setConnectionState({
          status: 'reconnecting',
          message: `日志连接中断，${Math.round(delay / 1000)}秒后自动重连...`,
        })
        onErrorRef.current?.('SSE disconnected, reconnecting...')

        // Schedule reconnect with exponential backoff
        isReconnectRef.current = true
        reconnectTimerRef.current = setTimeout(() => {
          reconnectTimerRef.current = null
          currentDelayRef.current = Math.min(delay * 2, maxReconnectDelay)
          connect()
        }, delay)
      }
    }

    connect()

    return cleanup
  }, [url, enabled, reconnectDelay, maxReconnectDelay])

  return { connectionState }
}
