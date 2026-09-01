import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import Button from '@/components/ui/Button'
import PasswordInput from '@/components/ui/PasswordInput'

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token') || ''
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [done, setDone] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!token) { setError('链接无效或已过期'); return }
    if (!password.trim()) { setError('请输入新密码'); return }
    if (password.length < 6) { setError('密码至少 6 个字符'); return }
    if (password !== confirm) { setError('两次密码不一致'); return }
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/api/v1/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, password }),
      })
      const data = await res.json().catch(() => ({ error: '重置失败' }))
      if (res.ok) setDone(true)
      else setError(data.error || '重置失败')
    } catch { setError('重置失败') }
    finally { setLoading(false) }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--bg)]">
      <div className="w-full max-w-sm p-8 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] shadow-lg">
        <div className="text-center mb-6">
          <img src="/logo.svg" alt="EvalPerf" className="h-12 mx-auto mb-2" />
          <h1 className="text-xl font-semibold text-[var(--text)]">重置密码</h1>
        </div>
        {done ? (
          <div className="text-center">
            <p className="text-sm text-[var(--text)] mb-4">密码已重置，请用新密码登录。</p>
            <Link to="/login">
              <Button variant="primary" className="w-full">去登录</Button>
            </Link>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <PasswordInput value={password} onChange={setPassword} placeholder="新密码 (至少6位)" />
            <PasswordInput value={confirm} onChange={setConfirm} placeholder="确认新密码" />
            {error && <p className="text-xs text-[var(--danger)]">{error}</p>}
            <Button type="submit" variant="primary" disabled={loading} className="w-full">
              {loading ? '提交中...' : '重置密码'}
            </Button>
          </form>
        )}
        {!done && (
          <p className="mt-4 text-center text-xs text-[var(--text-muted)]">
            <Link to="/login" className="text-[var(--accent)] hover:underline">返回登录</Link>
          </p>
        )}
      </div>
    </div>
  )
}
