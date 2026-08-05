import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'
import Button from '@/components/ui/Button'

export default function RegisterPage() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password.trim()) {
      setError('请输入用户名和密码')
      return
    }
    if (username.length < 2 || username.length > 32) {
      setError('用户名需要 2-32 个字符')
      return
    }
    if (password.length < 6) {
      setError('密码至少 6 个字符')
      return
    }
    if (password !== confirm) {
      setError('两次密码不一致')
      return
    }
    setLoading(true)
    setError('')
    try {
      await register(username, password)
      navigate('/dashboard')
    } catch (err) {
      setError(err instanceof Error ? err.message : '注册失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--bg)]">
      <div className="w-full max-w-sm p-8 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] shadow-lg">
        <div className="text-center mb-6">
          <img src="/logo.svg" alt="EvalPerf" className="h-12 mx-auto mb-2" />
          <h1 className="text-xl font-semibold text-[var(--text)]">注册</h1>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input type="text" value={username} onChange={(e) => setUsername(e.target.value)}
            placeholder="用户名 (2-32 字符)" autoFocus
            className="w-full px-3 py-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-[var(--text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]" />
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
            placeholder="密码 (至少6位)"
            className="w-full px-3 py-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-[var(--text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]" />
          <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)}
            placeholder="确认密码"
            className="w-full px-3 py-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-[var(--text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]" />
          {error && <p className="text-xs text-[var(--danger)]">{error}</p>}
          <Button type="submit" variant="primary" disabled={loading} className="w-full">
            {loading ? '注册中...' : '注册'}
          </Button>
        </form>
        <p className="mt-4 text-center text-xs text-[var(--text-muted)]">
          已有账号？{' '}
          <Link to="/login" className="text-[var(--accent)] hover:underline">登录</Link>
        </p>
      </div>
    </div>
  )
}
