/**
 * Generate a client task id with a millisecond timestamp plus random suffix.
 *
 * Task ids are opaque to the backend/report parser; keeping the timestamp
 * preserves operator readability while the suffix removes the Date.now()-only
 * collision window for automated/concurrent submissions.
 */
export function createTaskId(prefix: string): string {
  const bytes = new Uint8Array(3)
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(bytes)
  } else {
    // Compatibility fallback for environments without Web Crypto.
    for (let i = 0; i < bytes.length; i += 1) {
      bytes[i] = Math.floor(Math.random() * 256)
    }
  }
  const suffix = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
  return `${prefix}_${Date.now()}_${suffix}`
}
