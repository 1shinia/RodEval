export function createInFlightDeduper<K, V>(request: (key: K) => Promise<V>): (key: K) => Promise<V> {
  const inFlight = new Map<K, Promise<V>>()

  return (key) => {
    const existing = inFlight.get(key)
    if (existing) return existing

    const pending = request(key).finally(() => inFlight.delete(key))
    inFlight.set(key, pending)
    return pending
  }
}
