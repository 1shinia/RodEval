import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import { CompareProvider } from '@/contexts/CompareContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { AuthProvider, useAuth } from '@/contexts/AuthContext'
import MainLayout from '@/layouts/MainLayout'
import ErrorBoundary from '@/components/common/ErrorBoundary'
import PageErrorBoundary from '@/components/common/PageErrorBoundary'
import ToastContainer from '@/components/common/Toast'
import { lazy, Suspense } from 'react'
import LoadingSpinner from '@/components/common/LoadingSpinner'
import type { ReactNode } from 'react'

const DashboardPage = lazy(() => import('@/pages/DashboardPage'))
const ReportsLayout = lazy(() => import('@/pages/ReportsLayout'))
const LLMReportsTab = lazy(() => import('@/pages/LLMReportsTab'))
const RAGReportsTab = lazy(() => import('@/pages/RAGReportsTab'))
const AIGCReportsTab = lazy(() => import('@/pages/AIGCReportsTab'))
const AudioReportsTab = lazy(() => import('@/pages/AudioReportsTab'))
const ReportDetailPage = lazy(() => import('@/pages/ReportDetailPage'))
const ComparePage = lazy(() => import('@/pages/ComparePage'))
const EvalLayout = lazy(() => import('@/pages/EvalLayout'))
const EvalLLMTab = lazy(() => import('@/pages/EvalLLMTab'))
const EvalRAGTab = lazy(() => import('@/pages/EvalRAGTab'))
const EvalAIGCTab = lazy(() => import('@/pages/EvalAIGCTab'))
const EvalAudioTab = lazy(() => import('@/pages/EvalAudioTab'))
const PerfTaskPage = lazy(() => import('@/pages/PerfTaskPage'))
const PerfReportsPage = lazy(() => import('@/pages/PerfReportsPage'))
const ReportViewerPage = lazy(() => import('@/pages/ReportViewerPage'))
const BenchmarksPage = lazy(() => import('@/pages/BenchmarksPage'))
const AIGCReportDetailPage = lazy(() => import('@/pages/AIGCReportDetailPage'))
const AudioReportDetailPage = lazy(() => import('@/pages/AudioReportDetailPage'))
const LoginPage = lazy(() => import('@/pages/LoginPage'))
const RegisterPage = lazy(() => import('@/pages/RegisterPage'))

function RequireAuth({ children }: { children: ReactNode }) {
  const { isAuthenticated } = useAuth()
  if (!isAuthenticated) return <Navigate to="/login" replace />
  return <>{children}</>
}

function GuestOnly({ children }: { children: ReactNode }) {
  const { isAuthenticated } = useAuth()
  if (isAuthenticated) return <Navigate to="/dashboard" replace />
  return <>{children}</>
}

function AppRoutes() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        {/* Auth pages — guest only */}
        <Route path="/login" element={<GuestOnly><LoginPage /></GuestOnly>} />
        <Route path="/register" element={<GuestOnly><RegisterPage /></GuestOnly>} />

        {/* All other routes require auth */}
        <Route element={<RequireAuth><MainLayout /></RequireAuth>}>
          <Route path="/dashboard" element={<PageErrorBoundary pageName="dashboard"><DashboardPage /></PageErrorBoundary>} />
          <Route path="/reports" element={<PageErrorBoundary pageName="reports"><ReportsLayout /></PageErrorBoundary>}>
            <Route index element={<Navigate to="/reports/llm" replace />} />
            <Route path="llm" element={<PageErrorBoundary pageName="reports"><LLMReportsTab /></PageErrorBoundary>} />
            <Route path="rag" element={<PageErrorBoundary pageName="reports"><RAGReportsTab /></PageErrorBoundary>} />
            <Route path="aigc" element={<PageErrorBoundary pageName="reports"><AIGCReportsTab /></PageErrorBoundary>} />
            <Route path="audio" element={<PageErrorBoundary pageName="reports"><AudioReportsTab /></PageErrorBoundary>} />
          </Route>
          <Route path="/reports/:reportId" element={<PageErrorBoundary pageName="report-detail"><ReportDetailPage /></PageErrorBoundary>} />
          <Route path="/compare" element={<PageErrorBoundary pageName="compare"><ComparePage /></PageErrorBoundary>} />
          <Route path="/eval" element={<PageErrorBoundary pageName="eval"><EvalLayout /></PageErrorBoundary>}>
            <Route index element={<Navigate to="/eval/llm" replace />} />
            <Route path="llm" element={<PageErrorBoundary pageName="eval"><EvalLLMTab /></PageErrorBoundary>} />
            <Route path="rag" element={<PageErrorBoundary pageName="eval"><EvalRAGTab /></PageErrorBoundary>} />
            <Route path="aigc" element={<PageErrorBoundary pageName="eval"><EvalAIGCTab /></PageErrorBoundary>} />
            <Route path="audio" element={<PageErrorBoundary pageName="eval"><EvalAudioTab /></PageErrorBoundary>} />
          </Route>
          <Route path="/perf" element={<PageErrorBoundary pageName="perf"><PerfTaskPage /></PageErrorBoundary>} />
          <Route path="/perf-reports" element={<PageErrorBoundary pageName="perf-reports"><PerfReportsPage /></PageErrorBoundary>} />
          <Route path="/benchmarks" element={<PageErrorBoundary pageName="benchmarks"><BenchmarksPage /></PageErrorBoundary>} />
          <Route path="/viewer" element={<PageErrorBoundary pageName="viewer"><ReportViewerPage /></PageErrorBoundary>} />
          <Route path="/reports/aigc/:taskId" element={<PageErrorBoundary pageName="aigc-report"><AIGCReportDetailPage /></PageErrorBoundary>} />
          <Route path="/reports/audio/:taskId" element={<PageErrorBoundary pageName="audio-report"><AudioReportDetailPage /></PageErrorBoundary>} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
    </Suspense>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <ErrorBoundary>
        <ThemeProvider>
          <LocaleProvider>
            <AuthProvider>
              <ReportsProvider>
              <CompareProvider>
                <AppRoutes />
                <ToastContainer />
              </CompareProvider>
              </ReportsProvider>
            </AuthProvider>
          </LocaleProvider>
        </ThemeProvider>
      </ErrorBoundary>
    </BrowserRouter>
  )
}
