import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

interface User {
  id: number
  username: string
  role: string
}

type RegistrationMode = 'admin_only' | 'invite' | 'public'

interface AuthState {
  token: string | null
  user: User | null
  registrationMode: RegistrationMode
  registrationModeLocked: boolean
  registrationPolicyLoaded: boolean
  updateRegistrationMode: (mode: RegistrationMode) => Promise<void>
  login: (username: string, password: string) => Promise<void>
  register: (username: string, password: string, inviteCode?: string) => Promise<void>
  logout: () => void
  isAuthenticated: boolean
}

const AuthContext = createContext<AuthState | null>(null)
const AUTH_KEY = 'evalscope_token'
const USER_KEY = 'evalscope_user'

function getStoredToken(): string | null {
  return localStorage.getItem(AUTH_KEY)
}

function getStoredUser(): User | null {
  try {
    const raw = localStorage.getItem(USER_KEY)
    return raw ? JSON.parse(raw) : null
  } catch { return null }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(getStoredToken)
  const [user, setUser] = useState<User | null>(getStoredUser)
  const [registrationMode, setRegistrationMode] = useState<RegistrationMode>('admin_only')
  const [registrationModeLocked, setRegistrationModeLocked] = useState(false)
  const [registrationPolicyLoaded, setRegistrationPolicyLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false
    fetch('/api/v1/config')
      .then(async (response) => {
        if (!response.ok) throw new Error('Failed to load registration policy')
        return response.json()
      })
      .then((config) => {
        if (cancelled) return
        const mode = config.registration_mode
        setRegistrationMode(mode === 'public' || mode === 'invite' ? mode : 'admin_only')
        setRegistrationModeLocked(config.registration_mode_locked === true)
      })
      .catch(() => {
        if (!cancelled) setRegistrationMode('admin_only')
      })
      .finally(() => {
        if (!cancelled) setRegistrationPolicyLoaded(true)
      })
    return () => { cancelled = true }
  }, [])

  const isAuthenticated = !!(token && user)

  const updateRegistrationMode = useCallback(async (mode: RegistrationMode) => {
    const res = await fetch('/api/v1/auth/settings/registration', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + token },
      body: JSON.stringify({ mode }),
    })
    const data = await res.json().catch(() => ({ error: '保存失败' }))
    if (!res.ok) throw new Error(data.error || '保存失败')
    setRegistrationMode(data.registration_mode)
    setRegistrationModeLocked(data.registration_mode_locked === true)
  }, [token])

  const login = useCallback(async (username: string, password: string) => {
    const res = await fetch('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({ error: '登录失败' }))
      throw new Error(body.error || '登录失败')
    }
    const data = await res.json()
    localStorage.setItem(AUTH_KEY, data.token)
    localStorage.setItem(USER_KEY, JSON.stringify(data.user))
    setToken(data.token)
    setUser(data.user)
  }, [])

  const register = useCallback(async (username: string, password: string, inviteCode?: string) => {
    const res = await fetch('/api/v1/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, ...(inviteCode ? { invite_code: inviteCode } : {}) }),
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({ error: '注册失败' }))
      throw new Error(body.error || '注册失败')
    }
    const data = await res.json()
    localStorage.setItem(AUTH_KEY, data.token)
    localStorage.setItem(USER_KEY, JSON.stringify(data.user))
    setToken(data.token)
    setUser(data.user)
  }, [])

  const logout = useCallback(async () => {
    try {
      await fetch('/api/v1/auth/logout', {
        method: 'POST',
        headers: { Authorization: 'Bearer ' + token },
      })
    } catch { /* ignore network errors */ }
    localStorage.removeItem(AUTH_KEY)
    localStorage.removeItem(USER_KEY)
    setToken(null)
    setUser(null)
  }, [token])

  const value = useMemo(() => ({
    token, user, registrationMode, registrationModeLocked, registrationPolicyLoaded,
    updateRegistrationMode, login, register, logout, isAuthenticated,
  }), [
    token, user, registrationMode, registrationModeLocked, registrationPolicyLoaded,
    updateRegistrationMode, login, register, logout, isAuthenticated,
  ])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// Provider and hook intentionally share one module as the app-wide auth API.
// eslint-disable-next-line react-refresh/only-export-components
export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
