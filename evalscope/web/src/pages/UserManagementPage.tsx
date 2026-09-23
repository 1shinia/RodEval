import { useCallback, useEffect, useState } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { toast } from '@/components/common/Toast'
import Button from '@/components/ui/Button'
import { Trash2, Key, Plus, X, Link2, Copy, Check } from 'lucide-react'

interface UserInfo {
  id: number
  username: string
  role: string
  created_at: string
}

export default function UserManagementPage() {
  const {
    token, registrationMode, registrationModeLocked, updateRegistrationMode,
  } = useAuth()
  const [selectedRegistrationMode, setSelectedRegistrationMode] = useState<typeof registrationMode | null>(null)
  const [savingRegistrationMode, setSavingRegistrationMode] = useState(false)
  const [users, setUsers] = useState<UserInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [newUsername, setNewUsername] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [newRole, setNewRole] = useState('user')
  const [resetId, setResetId] = useState<number | null>(null)
  const [resetPassword, setResetPassword] = useState('')
  const [resetLink, setResetLink] = useState<string | null>(null)
  const [resetLinkUser, setResetLinkUser] = useState('')
  const [copied, setCopied] = useState(false)
  const [inviteCode, setInviteCode] = useState<string | null>(null)
  const [inviteMaxUses, setInviteMaxUses] = useState(1)
  const [inviteCopied, setInviteCopied] = useState(false)
  const effectiveSelectedRegistrationMode = selectedRegistrationMode ?? registrationMode

  const authHeaders = { Authorization: 'Bearer ' + token }

  const copyText = async (text: string) => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
        return
      }
    } catch {
      // Clipboard API may exist but be blocked in an insecure browser context.
    }
    const el = document.createElement('textarea')
    el.value = text
    el.setAttribute('readonly', '')
    el.style.position = 'fixed'
    el.style.opacity = '0'
    document.body.appendChild(el)
    el.select()
    const copied = document.execCommand('copy')
    document.body.removeChild(el)
    if (!copied) throw new Error('copy command failed')
  }

  const loadUsers = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/auth/users', {
        headers: { Authorization: 'Bearer ' + token },
      })
      const data = await res.json()
      if (res.ok) setUsers(data.users || [])
      else toast.error(data.error || '加载失败')
    } catch { toast.error('加载失败') }
    finally { setLoading(false) }
  }, [token])

  useEffect(() => {
    const timer = window.setTimeout(() => { void loadUsers() }, 0)
    return () => window.clearTimeout(timer)
  }, [loadUsers])

  const handleRegistrationModeSave = async () => {
    setSavingRegistrationMode(true)
    try {
      await updateRegistrationMode(effectiveSelectedRegistrationMode)
      setSelectedRegistrationMode(null)
      setInviteCode(null)
      toast.success('注册策略已更新，立即生效')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '保存失败')
    } finally {
      setSavingRegistrationMode(false)
    }
  }

  const handleCreate = async () => {
    if (!newUsername.trim() || !newPassword.trim()) return
    try {
      const res = await fetch('/api/v1/auth/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({ username: newUsername, password: newPassword, role: newRole }),
      })
      const data = await res.json()
      if (res.ok) {
        toast.success('用户已创建')
        setShowForm(false)
        setNewUsername('')
        setNewPassword('')
        loadUsers()
      } else toast.error(data.error || '创建失败')
    } catch { toast.error('创建失败') }
  }

  const handleDelete = async (id: number, username: string) => {
    if (!window.confirm(`确定删除用户「${username}」吗？`)) return
    try {
      const res = await fetch(`/api/v1/auth/users/${id}`, { method: 'DELETE', headers: authHeaders })
      const data = await res.json()
      if (res.ok) { toast.success('已删除'); loadUsers() }
      else toast.error(data.error || '删除失败')
    } catch { toast.error('删除失败') }
  }

  const handleResetPassword = async (id: number) => {
    if (!resetPassword.trim() || resetPassword.length < 6) {
      toast.error('密码至少6位')
      return
    }
    try {
      const res = await fetch(`/api/v1/auth/users/${id}/password`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({ password: resetPassword }),
      })
      const data = await res.json()
      if (res.ok) { toast.success('密码已重置'); setResetId(null); setResetPassword('') }
      else toast.error(data.error || '重置失败')
    } catch { toast.error('重置失败') }
  }

  const handleGenerateResetLink = async (id: number, username: string) => {
    try {
      const res = await fetch(`/api/v1/auth/users/${id}/reset-token`, { method: 'POST', headers: authHeaders })
      const data = await res.json()
      if (res.ok) {
        setResetLink(`${window.location.origin}/reset-password?token=${data.token}`)
        setResetLinkUser(username)
        setCopied(false)
      } else toast.error(data.error || '生成失败')
    } catch { toast.error('生成失败') }
  }

  const handleCopyResetLink = async () => {
    if (!resetLink) return
    try {
      await copyText(resetLink)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch { toast.error('复制失败，请手动复制') }
  }

  const handleGenerateInvite = async () => {
    try {
      const res = await fetch('/api/v1/auth/invites', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({ max_uses: inviteMaxUses, expires_in_hours: 24 }),
      })
      const data = await res.json()
      if (res.ok) {
        setInviteCode(data.code)
        setInviteCopied(false)
      } else toast.error(data.error || '生成失败')
    } catch { toast.error('生成失败') }
  }

  const handleCopyInvite = async () => {
    if (!inviteCode) return
    try {
      await copyText(inviteCode)
      setInviteCopied(true)
      setTimeout(() => setInviteCopied(false), 2000)
    } catch { toast.error('复制失败，请手动复制') }
  }

  return (
    <div className="page-enter flex flex-col gap-6 max-w-3xl">
      <div className="flex items-center justify-between">
        <h1 className="type-heading-lg text-[var(--text)]">用户管理</h1>
        <div className="flex items-center gap-2">
          {registrationMode === 'invite' && (
            <>
              <input type="number" min={1} max={10000} value={inviteMaxUses}
                onChange={e => setInviteMaxUses(Math.max(1, Math.min(10000, Number(e.target.value) || 1)))}
                className="w-24 px-3 py-2 rounded-lg border border-[var(--border)] bg-white text-sm"
                aria-label="邀请码可用次数" title="邀请码可用次数" />
              <Button variant="outline" size="sm" onClick={handleGenerateInvite}>
                <Link2 size={14} /> 生成邀请码
              </Button>
            </>
          )}
          <Button variant="primary" size="sm" onClick={() => setShowForm(!showForm)}>
            <Plus size={14} /> 创建用户
          </Button>
        </div>
      </div>

      <section className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div className="flex-1">
            <h2 className="text-base font-semibold text-[var(--text)]">注册策略</h2>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              控制新用户能否自行注册，保存后立即生效，无需重启服务。
            </p>
            {registrationModeLocked && (
              <p className="mt-2 text-sm text-[var(--warning-text)]">
                当前策略由服务器配置锁定，只能由运维人员修改。
              </p>
            )}
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1 text-sm text-[var(--text-muted)]">
              注册方式
              <select
                value={effectiveSelectedRegistrationMode}
                onChange={e => setSelectedRegistrationMode(e.target.value as typeof registrationMode)}
                disabled={registrationModeLocked || savingRegistrationMode}
                className="min-w-44 rounded-lg border border-[var(--border)] bg-white px-3 py-2 text-sm text-[var(--text)] disabled:cursor-not-allowed disabled:opacity-60"
              >
                <option value="admin_only">关闭自行注册</option>
                <option value="public">允许公开注册</option>
                <option value="invite">仅邀请码注册</option>
              </select>
            </label>
            <Button
              variant="primary"
              size="sm"
              onClick={handleRegistrationModeSave}
              disabled={registrationModeLocked || savingRegistrationMode || effectiveSelectedRegistrationMode === registrationMode}
            >
              {savingRegistrationMode ? '保存中...' : '保存策略'}
            </Button>
          </div>
        </div>
        <p className="mt-3 text-sm text-[var(--text-muted)]">
          {effectiveSelectedRegistrationMode === 'admin_only' && '新用户不能自行注册，只能由管理员创建账号。'}
          {effectiveSelectedRegistrationMode === 'public' && '登录页将显示注册入口，任何访问者都可以创建普通用户账号。'}
          {effectiveSelectedRegistrationMode === 'invite' && '登录页将显示注册入口，但新用户必须填写有效邀请码。'}
        </p>
      </section>

      {inviteCode && (
        <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-card)] p-4">
          <p className="text-sm text-[var(--text)] mb-1">邀请码（24 小时内有效，可使用 {inviteMaxUses} 次）</p>
          <p className="text-sm text-[var(--text-muted)] mb-3">邀请码只显示一次，请立即复制并安全发送给用户。</p>
          <div className="flex items-center gap-2">
            <input readOnly value={inviteCode} onFocus={e => e.target.select()}
              className="flex-1 px-3 py-2 rounded-lg border border-[var(--border)] bg-white text-sm text-[var(--text)]" />
            <Button variant="primary" size="sm" onClick={handleCopyInvite}>
              {inviteCopied ? <Check size={14} /> : <Copy size={14} />} {inviteCopied ? '已复制' : '复制'}
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setInviteCode(null)}>关闭</Button>
          </div>
        </div>
      )}

      {showForm && (
        <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-card)] p-4 flex items-end gap-3 flex-wrap">
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">用户名</label>
            <input value={newUsername} onChange={e => setNewUsername(e.target.value)}
              className="w-32 px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-sm"
              placeholder="2-32字符" />
          </div>
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">密码</label>
            <input type="password" value={newPassword} onChange={e => setNewPassword(e.target.value)}
              className="w-32 px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-sm"
              placeholder="至少6位" />
          </div>
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">角色</label>
            <select value={newRole} onChange={e => setNewRole(e.target.value)}
              className="w-24 px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-sm">
              <option value="user">user</option>
              <option value="admin">admin</option>
            </select>
          </div>
          <Button variant="primary" size="sm" onClick={handleCreate}
            disabled={!newUsername.trim() || !newPassword.trim()}>创建</Button>
          <Button variant="ghost" size="sm" onClick={() => setShowForm(false)}><X size={14} /></Button>
        </div>
      )}

      {loading ? (
        <div className="text-sm text-[var(--text-muted)]">加载中...</div>
      ) : (
        <div className="rounded-lg border border-[var(--border)] overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-[var(--bg-deep)] border-b border-[var(--border)]">
                <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">ID</th>
                <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">用户名</th>
                <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">角色</th>
                <th className="text-left px-4 py-3 font-medium text-[var(--text-muted)]">创建时间</th>
                <th className="text-right px-4 py-3 font-medium text-[var(--text-muted)]">操作</th>
              </tr>
            </thead>
            <tbody>
              {users.map(u => (
                <tr key={u.id} className="border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--bg-card2)]">
                  <td className="px-4 py-3 text-[var(--text-muted)]">{u.id}</td>
                  <td className="px-4 py-3 text-[var(--text)]">{u.username}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      u.role === 'admin' ? 'bg-[var(--accent)]/10 text-[var(--accent)]' : 'bg-[var(--bg-card2)] text-[var(--text-muted)]'
                    }`}>{u.role}</span>
                  </td>
                  <td className="px-4 py-3 text-[var(--text-muted)] text-xs">{u.created_at?.slice(0, 10)}</td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-1">
                      {resetId === u.id ? (
                        <div className="flex items-center gap-1">
                          <input type="password" value={resetPassword} onChange={e => setResetPassword(e.target.value)}
                            className="w-24 px-2 py-1 rounded border border-[var(--border)] bg-[var(--bg)] text-xs"
                            placeholder="新密码" autoFocus />
                          <Button variant="primary" size="sm" onClick={() => handleResetPassword(u.id)}>确认</Button>
                          <button onClick={() => { setResetId(null); setResetPassword('') }}
                            className="p-1 rounded hover:bg-[var(--bg-card2)]"><X size={12} /></button>
                        </div>
                      ) : (
                        <>
                          <button onClick={() => setResetId(u.id)}
                            className="p-1.5 rounded cursor-pointer opacity-50 hover:opacity-100 hover:bg-[var(--accent-dim)] transition-all" title="重置密码">
                            <Key size={14} />
                          </button>
                          <button onClick={() => handleGenerateResetLink(u.id, u.username)}
                            className="p-1.5 rounded cursor-pointer opacity-50 hover:opacity-100 hover:bg-[var(--accent-dim)] transition-all" title="生成重置链接">
                            <Link2 size={14} />
                          </button>
                          <button onClick={() => handleDelete(u.id, u.username)}
                            className="p-1.5 rounded cursor-pointer opacity-50 hover:opacity-100 hover:bg-[var(--danger-bg)] hover:text-[var(--danger)] transition-all" title="删除">
                            <Trash2 size={14} />
                          </button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {resetLink && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={() => setResetLink(null)}>
          <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-6 max-w-md w-full shadow-xl" onClick={e => e.stopPropagation()}>
            <h2 className="text-base font-semibold text-[var(--text)] mb-1">重置链接（{resetLinkUser}）</h2>
            <p className="text-xs text-[var(--text-muted)] mb-3">复制下面的链接发给用户，24 小时内有效，使用一次后失效。</p>
            <div className="flex items-center gap-2">
              <input readOnly value={resetLink} onFocus={e => e.target.select()}
                className="flex-1 px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-xs text-[var(--text)]" />
              <Button variant="primary" size="sm" onClick={handleCopyResetLink}>
                {copied ? <Check size={14} /> : <Copy size={14} />} {copied ? '已复制' : '复制'}
              </Button>
            </div>
            <div className="mt-4 flex justify-end">
              <Button variant="ghost" size="sm" onClick={() => setResetLink(null)}>关闭</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
