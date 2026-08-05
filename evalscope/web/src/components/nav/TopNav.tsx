import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { useAuth } from '@/contexts/AuthContext'
import LocaleToggle from './LocaleToggle'
import ThemeToggle from './ThemeToggle'
import RunningTasksIndicator from './RunningTasksIndicator'
import { LayoutDashboard, Sparkles, ClipboardCheck, GitCompareArrows, Activity, BarChart4, Medal, Menu, X, User, LogOut } from 'lucide-react'

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
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [userMenuOpen, setUserMenuOpen] = useState(false)

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
    </header>
  )
}
