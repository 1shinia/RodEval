import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { useAuth } from '@/contexts/AuthContext'
import { toast } from '@/components/common/Toast'
import LocaleToggle from './LocaleToggle'
import ThemeToggle from './ThemeToggle'
import RunningTasksIndicator from './RunningTasksIndicator'
import { LayoutDashboard, Sparkles, ClipboardCheck, GitCompareArrows, Activity, BarChart4, Medal, Menu, X, User, Users, LogOut, KeyRound } from 'lucide-react'

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2 px-3.5 py-2 rounded-lg text-base font-medium transition-all duration-200 ${
    isActive
      ? 'bg-[var(--accent)] text-[var(--text-on-filled)] shadow-[var(--shadow-glow-soft)]'
      : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)]'
  }`

const iconLinkClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center justify-center w-9 h-9 rounded-lg transition-all duration-200 ${
    isActive
      ? 'bg-[var(--accent)] text-[var(--text-on-filled)] shadow-[var(--shadow-glow-soft)]'
      : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)]'
  }`

const mobileLinkClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2.5 px-4 py-3 rounded-lg text-base font-medium transition-all duration-200 ${
    isActive
      ? 'bg-[var(--accent)] text-white'
      : 'text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)]'
  }`

export default function TopNav() {
  const { t } = useLocale()
  const { user, token, logout } = useAuth()
  const navigate = useNavigate()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const [pwDialogOpen, setPwDialogOpen] = useState(false)
  const [oldPw, setOldPw] = useState('')
  const [newPw, setNewPw] = useState('')
  const [pwError, setPwError] = useState('')
  const [pwLoading, setPwLoading] = useState(false)

  const handleChangePassword = async () => {
    if (!oldPw.trim() || !newPw.trim()) return
    if (newPw.length < 6) { setPwError('新密码至少6个字符'); return }
    setPwLoading(true); setPwError('')
    try {
      const res = await fetch('/api/v1/auth/password', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ old_password: oldPw, new_password: newPw }),
      })
      const data = await res.json()
      if (res.ok) {
        toast.success('密码已修改'); setPwDialogOpen(false); setOldPw(''); setNewPw('')
      } else {
        setPwError(data.error || '修改失败')
      }
    } catch { setPwError('网络错误') }
    finally { setPwLoading(false) }
  }

  const navItems = [
    { to: '/dashboard', icon: <LayoutDashboard size={18} />, label: t('nav.dashboard') },
    { to: '/eval', icon: <Sparkles size={18} />, label: t('nav.eval') },
    { to: '/reports', icon: <ClipboardCheck size={18} />, label: t('nav.evalReports') },
    { to: '/perf', icon: <Activity size={18} />, label: t('nav.perf') },
    { to: '/perf-reports', icon: <BarChart4 size={18} />, label: t('nav.perfReports') },
    { to: '/compare', icon: <GitCompareArrows size={18} />, label: '模型对比' },
    { to: '/benchmarks', icon: <Medal size={18} />, label: t('nav.benchmarks') },
  ]

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[var(--surface-glass)] backdrop-blur-xl">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent opacity-40" />
      <div className="flex items-center justify-between px-4 max-w-[1600px] mx-auto" style={{ height: '56px' }}>
        <div className="flex items-center gap-3 lg:gap-5 min-w-0">
          <div className="flex items-center gap-2 shrink-0">
            <img src="/logo.svg" alt="EvalPerf" className="h-10 object-contain" />
          </div>
          <nav className="hidden lg:flex items-center gap-0.5">
            {navItems.map((item) => (
              <NavLink key={item.to} to={item.to} className={linkClass}>
                {item.icon} {item.label}
              </NavLink>
            ))}
          </nav>
          <nav className="hidden md:flex lg:hidden items-center gap-0.5">
            {navItems.map((item) => (
              <NavLink key={item.to} to={item.to} className={iconLinkClass} title={item.label}>
                {item.icon}
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
          <RunningTasksIndicator />

          {/* User menu */}
          <div className="relative">
            <button onClick={() => setUserMenuOpen(!userMenuOpen)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-sm text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)] transition-all duration-200">
              <User size={16} />
              <span className="hidden sm:inline max-w-[100px] truncate">{user?.username ?? ''}</span>
            </button>
            {userMenuOpen && (
              <div className="absolute right-0 top-full mt-1 w-36 rounded-lg border border-[var(--border)] bg-[var(--bg-card)] shadow-lg py-1 z-50">
                {user?.role === 'admin' && (
                  <button onClick={() => { window.open('/admin/users', '_blank'); setUserMenuOpen(false) }}
                    className="flex items-center gap-2 w-full px-3 py-2 text-sm text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg)] transition-colors">
                    <Users size={14} />
                    用户管理
                  </button>
                )}
                <button onClick={() => { setPwDialogOpen(true); setUserMenuOpen(false) }}
                  className="flex items-center gap-2 w-full px-3 py-2 text-sm text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg)] transition-colors">
                  <KeyRound size={14} />
                  修改密码
                </button>
                <button onClick={() => { logout(); navigate('/login'); }}
                  className="flex items-center gap-2 w-full px-3 py-2 text-sm text-[var(--text-muted)] hover:text-[var(--danger)] hover:bg-[var(--bg)] transition-colors">
                  <LogOut size={14} />
                  退出登录
                </button>
              </div>
            )}
          </div>

          <LocaleToggle />
          <ThemeToggle />
          <button onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden w-8 h-8 flex items-center justify-center rounded-lg text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)] transition-all duration-200" aria-label={t('common.toggleMenu')}>
            {mobileOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>
      <div className={`md:hidden overflow-hidden transition-[max-height,opacity] duration-300 ease-in-out ${mobileOpen ? 'max-h-80 opacity-100' : 'max-h-0 opacity-0'}`}>
        <nav className="border-t border-[var(--border)] bg-[var(--bg-card)] px-3 py-2 flex flex-col gap-0.5">
          {navItems.map((item) => (
            <NavLink key={item.to} to={item.to} className={mobileLinkClass} onClick={() => setMobileOpen(false)}>
              {item.icon} {item.label}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* Password change dialog */}
      {pwDialogOpen && (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-[25vh]" onClick={() => setPwDialogOpen(false)}>
          <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] shadow-xl p-6 w-80" onClick={e => e.stopPropagation()}>
            <h3 className="text-sm font-medium mb-4">修改密码</h3>
            <div className="space-y-3">
              <input type="password" value={oldPw} onChange={e => setOldPw(e.target.value)}
                placeholder="当前密码" autoFocus
                className="w-full px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-sm" />
              <input type="password" value={newPw} onChange={e => setNewPw(e.target.value)}
                placeholder="新密码（至少6位）"
                className="w-full px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg)] text-sm" />
              {pwError && <p className="text-xs text-[var(--danger)]">{pwError}</p>}
              <div className="flex gap-2 justify-end">
                <button onClick={() => setPwDialogOpen(false)} className="px-3 py-1.5 text-sm rounded-lg text-[var(--text-muted)] hover:bg-[var(--bg-card2)]">取消</button>
                <button onClick={handleChangePassword} disabled={pwLoading}
                  className="px-3 py-1.5 text-sm rounded-lg bg-[var(--accent)] text-white hover:opacity-90 disabled:opacity-50">
                  {pwLoading ? '修改中...' : '确认'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </header>
  )
}
