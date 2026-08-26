/**
 * Timestamp formatting for persisted metadata times.
 *
 * Since schema v15 the backend stores timezone-aware UTC ISO-8601 strings.
 * Legacy pre-v15 values were naive server-local (Asia/Shanghai) wall time.
 * Parsing both through `new Date()` yields the correct instant in either
 * case on a CST browser; formatting with local getters then displays them
 * in the viewer's timezone instead of leaking raw UTC strings.
 *
 * Unparseable/opaque values are returned unchanged so historical junk never
 * turns into "Invalid Date".
 */

function parse(ts: string): Date | null {
  const d = new Date(ts)
  return isNaN(d.getTime()) ? null : d
}

const pad = (n: number) => String(n).padStart(2, '0')

/** `2026-08-26T09:18:18+00:00` -> `2026-08-26 17:18:18` (browser-local). */
export function formatTimestamp(ts: string | null | undefined): string {
  if (!ts) return ''
  const d = parse(ts)
  if (!d) return ts.replace('T', ' ').slice(0, 19)
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

/** Short form `MM-DD HH:MM` (browser-local), for compact cards/timelines. */
export function formatTimestampShort(ts: string | null | undefined): string {
  if (!ts) return ''
  const d = parse(ts)
  if (!d) return ts.replace('T', ' ').slice(5, 16)
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
