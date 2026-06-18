import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import MainLayout from '@/layouts/MainLayout'
import ErrorBoundary from '@/components/common/ErrorBoundary'
import PageErrorBoundary from '@/components/common/PageErrorBoundary'
import ToastContainer from '@/components/common/Toast'
import { lazy, Suspense } from 'react'
import LoadingSpinner from '@/components/common/LoadingSpinner'

const DashboardPage = lazy(() => import('@/pages/DashboardPage'))
const ReportsPage = lazy(() => import('@/pages/ReportsPage'))
const ReportDetailPage = lazy(() => import('@/pages/ReportDetailPage'))
const ComparePage = lazy(() => import('@/pages/ComparePage'))
const EvalTaskPage = lazy(() => import('@/pages/EvalTaskPage'))
const PerfTaskPage = lazy(() => import('@/pages/PerfTaskPage'))
const PerfReportsPage = lazy(() => import('@/pages/PerfReportsPage'))
const ReportViewerPage = lazy(() => import('@/pages/ReportViewerPage'))
const BenchmarksPage = lazy(() => import('@/pages/BenchmarksPage'))

function AppRoutes() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route element={<MainLayout />}>
          <Route path="/dashboard" element={<PageErrorBoundary pageName="dashboard"><DashboardPage /></PageErrorBoundary>} />
          <Route path="/reports" element={<PageErrorBoundary pageName="reports"><ReportsPage /></PageErrorBoundary>} />
          <Route path="/reports/:reportId" element={<PageErrorBoundary pageName="report-detail"><ReportDetailPage /></PageErrorBoundary>} />
          <Route path="/compare" element={<PageErrorBoundary pageName="compare"><ComparePage /></PageErrorBoundary>} />
          <Route path="/eval" element={<PageErrorBoundary pageName="eval"><EvalTaskPage /></PageErrorBoundary>} />
          <Route path="/perf" element={<PageErrorBoundary pageName="perf"><PerfTaskPage /></PageErrorBoundary>} />
          <Route path="/perf-reports" element={<PageErrorBoundary pageName="perf-reports"><PerfReportsPage /></PageErrorBoundary>} />
          <Route path="/benchmarks" element={<PageErrorBoundary pageName="benchmarks"><BenchmarksPage /></PageErrorBoundary>} />
          <Route path="/viewer" element={<PageErrorBoundary pageName="viewer"><ReportViewerPage /></PageErrorBoundary>} />
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
            <ReportsProvider>
              <AppRoutes />
              <ToastContainer />
            </ReportsProvider>
          </LocaleProvider>
        </ThemeProvider>
      </ErrorBoundary>
    </BrowserRouter>
  )
}
