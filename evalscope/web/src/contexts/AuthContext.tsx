import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

interface User {
  id: number
  username: string
  role: string
}

interface AuthState {
  token: string | null
  user: User | null
  login: (username: string, password: string) => Promise<void>
  register: (username: string, password: string) => Promise<void>
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

  const isAuthenticated = !!(token && user)

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

  const register = useCallback(async (username: string, password: string) => {
    const res = await fetch('/api/v1/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
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
    // Invalidate token on server
    try {
      await fetch('/api/v1/auth/logout', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      })
    } catch { /* ignore network errors */ }
    localStorage.removeItem(AUTH_KEY)
    localStorage.removeItem(USER_KEY)
    setToken(null)
    setUser(null)
  }, [token])

  const value = useMemo(() => ({
    token, user, login, register, logout, isAuthenticated,
  }), [token, user, login, register, logout, isAuthenticated])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
