import { useCallback, useEffect, useState } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { toast } from '@/components/common/Toast'
import Button from '@/components/ui/Button'
import { Trash2, Key, Plus, X } from 'lucide-react'

interface UserInfo {
  id: number
  username: string
  role: string
  created_at: string
}

export default function UserManagementPage() {
  const { token } = useAuth()
  const [users, setUsers] = useState<UserInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [newUsername, setNewUsername] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [newRole, setNewRole] = useState('user')
  const [resetId, setResetId] = useState<number | null>(null)
  const [resetPassword, setResetPassword] = useState('')

  const authHeaders = { Authorization: `Bearer ${token}` }

  const loadUsers = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/auth/users', { headers: authHeaders })
      const data = await res.json()
      if (res.ok) setUsers(data.users || [])
      else toast.error(data.error || '加载失败')
    } catch { toast.error('加载失败') }
    finally { setLoading(false) }
  }, [token])

  useEffect(() => { loadUsers() }, [loadUsers])

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

  return (
    <div className="page-enter flex flex-col gap-6 max-w-3xl">
      <div className="flex items-center justify-between">
        <h1 className="type-heading-lg text-[var(--text)]">用户管理</h1>
        <Button variant="primary" size="sm" onClick={() => setShowForm(!showForm)}>
          <Plus size={14} /> 创建用户
        </Button>
      </div>

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
    </div>
  )
}
