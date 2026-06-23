import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import { listReports } from '@/api/reports'
import type { ReportSummary } from '@/api/types'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import Skeleton from '@/components/ui/Skeleton'
import KpiCard from '@/components/ui/KpiCard'
import ScoreChip from '@/components/ui/ScoreChip'
import ScoreBadge from '@/components/ui/ScoreBadge'
import PathBar from '@/components/ui/PathBar'
import ServerBadge from '@/components/ui/ServerBadge'
import EvalRunCard from '@/components/ui/EvalRunCard'
import ModelGroupHeader from '@/components/ui/ModelGroupHeader'
import EmptyState from '@/components/common/EmptyState'
import {
  FileText,
  Cpu,
  Database,
  Clock,
  Inbox,
  FolderOpen,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { cn } from '@/lib/utils'

// ------------------------------------------------------------------ //
// Helpers                                                              //
// ------------------------------------------------------------------ //

/** Format timestamp to YYYY-MM-DD HH:MM:SS */
function formatTimestamp(ts: string): string {
  return ts.replace('T', ' ').slice(0, 19)
}

/** Format timestamp to short form MM-DD HH:MM */
function formatTimestampShort(ts: string): string {
  return ts.replace('T', ' ').slice(5, 16)
}

// ------------------------------------------------------------------ //
// CompactRunRow (Grouped view)                                         //
// ------------------------------------------------------------------ //
interface CompactRunRowProps {
  report: ReportSummary
  onClick: () => void
}

function CompactRunRow({ report, onClick }: CompactRunRowProps) {
  const dsScores = report.dataset_scores

  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 py-2.5 px-3 rounded-[var(--radius-sm)] hover:bg-[var(--bg-card2)] transition-colors w-full text-left"
    >
      {report.timestamp && (
        <span className="type-caption-mono text-[var(--text-muted)] shrink-0 w-[110px]">
          {formatTimestampShort(report.timestamp)}
        </span>
      )}
      <div className="flex flex-wrap gap-1 flex-1 min-w-0">
        {dsScores && Object.keys(dsScores).length > 0 ? (
          Object.entries(dsScores).map(([ds, s]) => <ScoreChip key={ds} label={ds} score={s} />)
        ) : (
          <span className="type-caption-mono text-[var(--text-muted)]">{report.dataset_name}</span>
        )}
      </div>
      <ScoreBadge score={report.score} className="shrink-0 !text-xs !px-2" />
    </button>
  )
}

// ------------------------------------------------------------------ //
// View toggle (segmented control)                                      //
// ------------------------------------------------------------------ //
type DashboardView = 'timeline' | 'grouped' | 'byDataset'

interface ViewToggleProps {
  value: DashboardView
  onChange: (v: DashboardView) => void
  labels: Record<DashboardView, string>
}

function ViewToggle({ value, onChange, labels }: ViewToggleProps) {
  const items: DashboardView[] = ['timeline', 'grouped', 'byDataset']
  return (
    <div className="inline-flex rounded-[var(--radius-sm)] border border-[var(--border-md)] overflow-hidden">
      {items.map((key) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          className={cn(
            'px-3.5 py-1.5 type-button-sm transition-colors cursor-pointer',
            value === key
              ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
              : 'bg-[var(--bg-card2)] text-[var(--text-muted)] hover:text-[var(--text)]',
          )}
        >
          {labels[key]}
        </button>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ //
// KPI Skeleton                                                        //
// ------------------------------------------------------------------ //
function KpiSkeleton() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {Array.from({ length: 4 }).map((_, i) => (
        <div
          key={i}
          className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5"
        >
          <Skeleton width={40} height={40} className="mb-3" />
          <Skeleton width={60} height={28} className="mb-1" />
          <Skeleton width={100} height={14} />
        </div>
      ))}
    </div>
  )
}

// ------------------------------------------------------------------ //
// Pagination                                                           //
// ------------------------------------------------------------------ //
interface PaginationProps {
  page: number
  pageSize: number
  total: number
  onChange: (page: number) => void
}

function Pagination({ page, pageSize, total, onChange }: PaginationProps) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  if (totalPages <= 1) return null

  return (
    <div className="flex items-center justify-center gap-2 pt-3 pb-1">
      <button
        onClick={() => onChange(page - 1)}
        disabled={page <= 1}
        className={cn(
          'p-1.5 rounded-[var(--radius-sm)] border border-[var(--border)] transition-colors',
          page <= 1
            ? 'opacity-40 cursor-not-allowed'
            : 'hover:bg-[var(--bg-card2)] cursor-pointer',
        )}
        aria-label="Previous page"
      >
        <ChevronLeft size={16} />
      </button>
      <span className="type-body-sm text-[var(--text-muted)] tabular-nums">
        {page} / {totalPages}
      </span>
      <button
        onClick={() => onChange(page + 1)}
        disabled={page >= totalPages}
        className={cn(
          'p-1.5 rounded-[var(--radius-sm)] border border-[var(--border)] transition-colors',
          page >= totalPages
            ? 'opacity-40 cursor-not-allowed'
            : 'hover:bg-[var(--bg-card2)] cursor-pointer',
        )}
        aria-label="Next page"
      >
        <ChevronRight size={16} />
      </button>
      <span className="type-body-xs text-[var(--text-dim)] ml-2">
        {t_pageLabel(total)}
      </span>
    </div>
  )
}

/** Helper to avoid needing the locale hook inside Pagination */
function t_pageLabel(total: number): string {
  return `共 ${total} 条`
}

// ------------------------------------------------------------------ //
// Dashboard Page                                                      //
// ------------------------------------------------------------------ //
const PAGE_SIZE = 20

export default function DashboardPage() {
  const { t } = useLocale()
  const { rootPath, setRootPath, serverAddress } = useReports()
  const navigate = useNavigate()

  const [pathInput, setPathInput] = useState(rootPath || './outputs')
  const [scanning, setScanning] = useState(false)
  const [reports, setReports] = useState<ReportSummary[]>([])
  const [scanned, setScanned] = useState(false)

  // Server-side pagination / search / sort state
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [view, setView] = useState<DashboardView>('timeline')
  const evalListRef = useRef<HTMLDivElement>(null)
  const [search, setSearch] = useState('')
  const [searchDebounced, setSearchDebounced] = useState('')
  const [sortBy, setSortBy] = useState<'time' | 'score' | 'model' | 'dataset'>('time')
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())

  // KPI stats (from server metadata)
  const [kpiTotal, setKpiTotal] = useState(0)
  const [kpiModels, setKpiModels] = useState(0)
  const [kpiDatasets, setKpiDatasets] = useState(0)
  const [kpiLatest, setKpiLatest] = useState('')

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearchDebounced(search)
      setPage(1)
    }, 300)
    return () => clearTimeout(timer)
  }, [search])

  // Fetch reports from server with pagination
  const fetchReports = useCallback(async (p: number, sb: string, sq: string) => {
    const trimmed = pathInput.trim()
    if (!trimmed) return
    setScanning(true)
    try {
      const res = await listReports({
        rootPath: trimmed,
        search: sq || undefined,
        sortBy: sb as 'score' | 'model' | 'dataset' | 'time',
        sortOrder: 'desc',
        page: p,
        pageSize: PAGE_SIZE,
      })
      setReports(res.reports)
      setTotal(res.total)
      setKpiTotal(res.total)
      setKpiModels(res.filters.available_models.length)
      setKpiDatasets(res.filters.available_datasets.length)
      // Latest timestamp from first report (already sorted desc by time)
      const latestTs = res.reports.length > 0
        ? formatTimestamp(res.reports[0].timestamp || res.reports[0].name)
        : t('dashboard.neverText')
      setKpiLatest(latestTs)
      setScanned(true)
    } catch {
      setReports([])
      setTotal(0)
      setScanned(true)
    } finally {
      setScanning(false)
    }
  }, [pathInput, t])

  // Scan action (initial load or path change)
  const handleScan = useCallback(async () => {
    const trimmed = pathInput.trim()
    if (!trimmed) return
    setRootPath(trimmed)
    setPage(1)
    setSearch('')
    setSearchDebounced('')
    setSortBy('time')
    await fetchReports(1, 'time', '')
  }, [pathInput, setRootPath, fetchReports])

  // Re-fetch when page, sort, or search changes
  useEffect(() => {
    if (!scanned) return
    fetchReports(page, sortBy, searchDebounced)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, sortBy, searchDebounced])

  // Auto-scan if rootPath is already set on mount
  useEffect(() => {
    if (rootPath && !scanned) {
      setPathInput(rootPath)
      handleScan()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Reset page when view changes (data is the same, just layout)
  const handleViewChange = useCallback((v: DashboardView) => {
    setView(v)
    setExpandedGroups(new Set())
  }, [])

  const toggleGroup = (key: string) => {
    setExpandedGroups(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const navigateToReport = (report: ReportSummary) => {
    navigate(`/reports/${encodeURIComponent(report.name)}?root_path=${encodeURIComponent(rootPath)}`)
  }

  const hasData = scanned && (total > 0 || reports.length > 0)

  const viewLabels: Record<DashboardView, string> = {
    timeline: t('dashboard.timelineView'),
    grouped: t('dashboard.groupedView'),
    byDataset: t('dashboard.byDatasetView'),
  }

  // Grouped by model (client-side grouping of current page)
  const grouped = useMemo(() => {
    const map = new Map<string, ReportSummary[]>()
    for (const r of reports) {
      const list = map.get(r.model_name) || []
      list.push(r)
      map.set(r.model_name, list)
    }
    return Array.from(map.entries())
  }, [reports])

  // Grouped by dataset
  const groupedByDataset = useMemo(() => {
    const map = new Map<string, ReportSummary[]>()
    for (const r of reports) {
      const list = map.get(r.dataset_name) || []
      list.push(r)
      map.set(r.dataset_name, list)
    }
    return Array.from(map.entries())
  }, [reports])

  return (
    <div className="flex flex-col gap-5 min-h-0">
      {/* ── Server Address ── */}
      <div className="flex items-center justify-between">
        <h1 className="type-heading-lg text-[var(--text)]">{t('nav.dashboard')}</h1>
        <ServerBadge address={serverAddress} />
      </div>

      {/* ── Path Bar ── */}
      <PathBar
        value={pathInput}
        onChange={setPathInput}
        onSubmit={handleScan}
        placeholder={t('dashboard.pathPlaceholder')}
        submitLabel={t('dashboard.scanBtn')}
        scanningLabel={t('dashboard.scanning')}
        scanning={scanning}
        disabled={!pathInput.trim()}
      />

      {/* ── KPI Cards ── */}
      {scanning && !scanned ? (
        <KpiSkeleton />
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <KpiCard
            icon={<FileText size={18} strokeWidth={2} />}
            value={String(kpiTotal)}
            label={t('dashboard.totalEvaluations')}
            gradient="var(--kpi-grad-0)"
            delay={0}
            onClick={() => { setView('timeline'); evalListRef.current?.scrollIntoView({ behavior: 'smooth' }) }}
          />
          <KpiCard
            icon={<Cpu size={18} strokeWidth={2} />}
            value={String(kpiModels)}
            label={t('dashboard.modelsEvaluated')}
            gradient="var(--kpi-grad-1)"
            delay={60}
            onClick={() => { setView('grouped'); evalListRef.current?.scrollIntoView({ behavior: 'smooth' }) }}
          />
          <KpiCard
            icon={<Database size={18} strokeWidth={2} />}
            value={String(kpiDatasets)}
            label={t('dashboard.datasetsUsed')}
            gradient="var(--kpi-grad-2)"
            delay={120}
            onClick={() => { setView('byDataset'); evalListRef.current?.scrollIntoView({ behavior: 'smooth' }) }}
          />
          <KpiCard
            icon={<Clock size={18} strokeWidth={2} />}
            value={kpiLatest.length > 20 ? kpiLatest.slice(0, 20) + '…' : kpiLatest}
            label={t('dashboard.latestEval')}
            gradient="var(--kpi-grad-3)"
            delay={180}
            onClick={() => reports.length > 0 && navigateToReport(reports[0])}
          />
        </div>
      )}

      {/* ── Loading skeleton for content ── */}
      {scanning && !scanned && (
        <Card title={t('dashboard.evaluations')}>
          <Skeleton lines={8} height={14} />
        </Card>
      )}

      {/* ── Unified Evaluation List ── */}
      {hasData && !scanning && (
        <div ref={evalListRef}>
          <Card title={t('dashboard.evaluations')} badge={<Badge>{total}</Badge>}>
            {/* Controls bar */}
            <div className="flex items-center gap-3 flex-wrap mb-4">
              <ViewToggle value={view} onChange={handleViewChange} labels={viewLabels} />

              {/* Search */}
              <input
                type="text"
                placeholder={t('dashboard.searchPlaceholder')}
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="flex-1 min-w-[160px] max-w-[300px] px-3 py-1.5 type-body-xs rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder-[var(--text-dim)]"
              />

              {/* Sort */}
              <select
                value={sortBy}
                onChange={e => {
                  setSortBy(e.target.value as 'time' | 'score' | 'model' | 'dataset')
                  setPage(1)
                }}
                className="px-3 py-1.5 type-body-xs rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)]"
              >
                <option value="time">{t('dashboard.sortTime')}</option>
                <option value="score">{t('dashboard.sortScore')}</option>
                <option value="model">{t('dashboard.sortModel')}</option>
              </select>
            </div>

            {/* List content */}
            <div className="overflow-y-auto max-h-[calc(100vh-300px)]">
              {reports.length === 0 ? (
                <div className="text-center py-8 type-body-sm text-[var(--text-muted)]">
                  {t('dashboard.noEvals')}
                </div>
              ) : view === 'timeline' ? (
                /* ── Timeline view ── */
                <div className="flex flex-col gap-3">
                  {reports.map((report) => (
                    <EvalRunCard
                      key={`${report.name}-${report.dataset_name}`}
                      report={report}
                      onClick={() => navigateToReport(report)}
                    />
                  ))}
                </div>
              ) : view === 'grouped' ? (
                /* ── Grouped view ── */
                <div className="flex flex-col gap-2">
                  {grouped.map(([model, runs]) => {
                    const expanded = expandedGroups.has(model)
                    const bestScore = Math.max(...runs.map(r => r.score))

                    return (
                      <div key={model} className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden">
                        <ModelGroupHeader
                          title={model}
                          count={runs.length}
                          runsLabel={t('dashboard.runs')}
                          bestScore={bestScore}
                          bestScoreLabel={t('dashboard.bestScore')}
                          expanded={expanded}
                          onToggle={() => toggleGroup(model)}
                        />

                        {expanded && (
                          <div className="border-t border-[var(--border)]">
                            {runs.map((report) => (
                              <CompactRunRow
                                key={`${report.name}-${report.dataset_name}`}
                                report={report}
                                onClick={() => navigateToReport(report)}
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              ) : view === 'byDataset' ? (
                /* ── By Dataset view ── */
                <div className="flex flex-col gap-2">
                  {groupedByDataset.map(([dataset, runs]) => {
                    const expanded = expandedGroups.has(dataset)
                    const bestScore = Math.max(...runs.map(r => r.score))

                    return (
                      <div key={dataset} className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden">
                        <ModelGroupHeader
                          title={dataset}
                          count={runs.length}
                          runsLabel={t('dashboard.runs')}
                          bestScore={bestScore}
                          bestScoreLabel={t('dashboard.bestScore')}
                          expanded={expanded}
                          onToggle={() => toggleGroup(dataset)}
                        />

                        {expanded && (
                          <div className="border-t border-[var(--border)]">
                            {runs.map((report) => (
                              <CompactRunRow
                                key={`${report.name}-${report.dataset_name}`}
                                report={report}
                                onClick={() => navigateToReport(report)}
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              ) : null}
            </div>

            {/* Pagination */}
            <Pagination page={page} pageSize={PAGE_SIZE} total={total} onChange={setPage} />
          </Card>
        </div>
      )}

      {/* ── Empty state after scan ── */}
      {scanned && !hasData && !scanning && (
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <EmptyState
            icon={<Inbox size={28} strokeWidth={1.5} />}
            title={t('dashboard.noReportsYet')}
            hint={t('dashboard.noReportsHint')}
          />
        </div>
      )}

      {/* ── Welcome state (before any scan) ── */}
      {!scanned && !scanning && (
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <EmptyState
            variant="welcome"
            icon={<FolderOpen size={28} strokeWidth={1.5} />}
            title={t('dashboard.welcomeTitle')}
            hint={t('dashboard.welcomeDesc')}
          />
        </div>
      )}
    </div>
  )
}
