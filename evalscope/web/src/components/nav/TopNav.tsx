import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import LocaleToggle from './LocaleToggle'
import ThemeToggle from './ThemeToggle'
import { BarChart3, Gauge, FlaskConical, BookOpen, FileText, Menu, X } from 'lucide-react'

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
  const [mobileOpen, setMobileOpen] = useState(false)

  const navItems = [
    { to: '/dashboard', icon: <BarChart3 size={18} />, label: t('nav.dashboard') },
    { to: '/reports', icon: <FileText size={18} />, label: t('nav.reports') },
    { to: '/eval', icon: <FlaskConical size={18} />, label: t('nav.eval') },
    { to: '/perf', icon: <Gauge size={18} />, label: t('nav.perf') },
    { to: '/benchmarks', icon: <BookOpen size={18} />, label: t('nav.benchmarks') },
  ]

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[var(--surface-glass)] backdrop-blur-xl">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent opacity-40" />
      <div className="flex items-center justify-between px-4 max-w-[1600px] mx-auto" style={{ height: '56px' }}>
        <div className="flex items-center gap-3 lg:gap-5 min-w-0">
          <div className="flex items-center gap-2 shrink-0">
            <img src="/logo.png" alt="RodCount" className="w-8 h-8 object-contain" />
            <span className="font-bold text-lg tracking-tight text-[var(--text)]">
              Rod<span className="text-[var(--accent)]">Count</span>
            </span>
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
          <a href="https://github.com/modelscope/evalscope" target="_blank" rel="noreferrer" title={t('common.github')}
            className="hidden sm:flex w-8 h-8 items-center justify-center rounded-lg text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)] transition-all duration-200">
            <svg viewBox="0 0 24 24" width={16} height={16} fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
            </svg>
          </a>
          <LocaleToggle />
          <ThemeToggle />
          <button onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden w-8 h-8 flex items-center justify-center rounded-lg text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)] transition-all duration-200" aria-label="Toggle menu">
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
