import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'
import Button from '@/components/ui/Button'

export default function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password.trim()) {
      setError('请输入用户名和密码')
      return
    }
    setLoading(true)
    setError('')
    try {
      await login(username, password)
      navigate('/dashboard')
    } catch (err) {
      setError(err instanceof Error ? err.message : '登录失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--bg)]">
      <div className="w-full max-w-sm p-8 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] shadow-lg">
        <div className="text-center mb-6">
          <img src="/logo.svg" alt="EvalPerf" className="h-12 mx-auto mb-2" />
          <h1 className="text-xl font-semibold text-[var(--text)]">登录</h1>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="用户名"
              className="w-full px-3 py-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-[var(--text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
              autoFocus
            />
          </div>
          <div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="密码"
              className="w-full px-3 py-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-[var(--text)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
            />
          </div>
          {error && <p className="text-xs text-[var(--danger)]">{error}</p>}
          <Button type="submit" variant="primary" disabled={loading} className="w-full">
            {loading ? '登录中...' : '登录'}
          </Button>
        </form>
        <p className="mt-4 text-center text-xs text-[var(--text-muted)]">
          还没有账号？{' '}
          <Link to="/register" className="text-[var(--accent)] hover:underline">注册</Link>
        </p>
      </div>
    </div>
  )
}
