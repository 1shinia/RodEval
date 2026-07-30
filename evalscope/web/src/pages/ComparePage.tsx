import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { useCompare } from '@/contexts/CompareContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPredictions, getChartUrl } from '@/api/reports'
import { comparePerfReports, type PerfCompareResponse, saveCompareReport, listSavedCompareReports, deleteCompareReport, type SavedCompareReport } from '@/api/perf'
import { toast } from '@/components/common/Toast'
import type { ReportData, PredictionRow } from '@/api/types'
import { getDisplayNames, parseReportName } from '@/utils/reportParser'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Card from '@/components/ui/Card'
import Tabs from '@/components/ui/Tabs'
import { scoreColor } from '@/utils/colorScale'
import FilterChip from '@/components/ui/FilterChip'
import Button from '@/components/ui/Button'
import Select from '@/components/ui/Select'
import Skeleton from '@/components/ui/Skeleton'
import { cn } from '@/lib/utils'
import PlotlyChart from '@/components/charts/PlotlyChart'
import ChatView from '@/components/single/ChatView'
import { ChevronLeft, ChevronRight, AlertCircle, CircleCheck, CircleX, ExternalLink, Trash2, Download, X } from 'lucide-react'
import { BarChart as RBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'

// ------------------------------------------------------------------ //
// Types                                                               //
// ------------------------------------------------------------------ //

interface MergedPrediction {
  Index: string
  Input: string
  Gold: string
  models: Record<string, PredictionRow>
}

type PerModelFilter = 'any' | 'pass' | 'fail'

// Dynamic color palette — generates distinct hues for unlimited models
const PALETTE_COLORS = [
  '#816DF8', '#0F9C7E', '#f59e0b', '#ef4444', '#3b82f6',
  '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#06b6d4',
  '#84cc16', '#d946ef', '#22c55e', '#64748b',
]

function modelColor(idx: number): string {
  return PALETTE_COLORS[idx % PALETTE_COLORS.length]
}

function modelBg(idx: number): string {
  return `${modelColor(idx)}15`
}

// ------------------------------------------------------------------ //
// Main Component                                                      //
// ------------------------------------------------------------------ //

export default function ComparePage() {
  const { t } = useLocale()
  const qp = useQueryParams()
  const { selection, clearCompareSelection } = useCompare()
  const { rootPath: ctxRootPath, setRootPath, loadMultiReports, loading, reportCache } = useReports()

  const rootPath = qp.get('root_path') || selection?.rootPath || ctxRootPath

  // ── LLM comparison state (must be before early returns for hook ordering) ──
  const reportNames = useMemo(
    () => (selection?.backend !== 'Perf' ? selection?.reports || [] : []),
    [selection],
  )

  const [reports, setReports] = useState<ReportData[]>([])
  const [dataLoaded, setDataLoaded] = useState(false)
  const [activeTab, setActiveTab] = useState<'score' | 'prediction'>('score')
  const [selectedDs, setSelectedDs] = useState('')
  const [selectedSubset, setSelectedSubset] = useState('')
  const [mergedPredictions, setMergedPredictions] = useState<MergedPrediction[]>([])
  const [perModelFilter, setPerModelFilter] = useState<Record<string, PerModelFilter>>({})
  const [threshold, setThreshold] = useState(0.99)
  const [page, setPage] = useState(1)
  const [predictionsLoading, setPredictionsLoading] = useState(false)

  const reportNamesKey = reportNames.join(';')
  useEffect(() => { setPerModelFilter({}) }, [reportNamesKey])

  useEffect(() => {
    if (rootPath && rootPath !== ctxRootPath) setRootPath(rootPath)
  }, [rootPath, ctxRootPath, setRootPath])

  useEffect(() => {
    if (reportNames.length < 2) return
    setDataLoaded(false)
    loadMultiReports(reportNames)
      .then((list) => { setReports(list); setDataLoaded(true) })
      .catch((e) => {
        toast.error(e instanceof Error ? e.message : t('common.loadFailed'))
        setDataLoaded(true)
      })
  }, [reportNames, loadMultiReports])

  const { scoreTableData, scoreTableColumns, displayNames } = useMemo(() => {
    const displayNames = getDisplayNames(reportNames)
    if (!reports.length) return { scoreTableData: [], scoreTableColumns: [], displayNames }

    const byReport: Record<string, Record<string, number>> = {}
    for (const r of reports) {
      const key = (r as ReportData & { _reportName?: string })._reportName ?? r.model_name
      if (!byReport[key]) byReport[key] = {}
      byReport[key][r.dataset_name] = r.score
    }

    const reportKeys = reportNames.filter((n) => byReport[n])
    const dsLists = reportKeys.map((k) => new Set(Object.keys(byReport[k])))
    const common = dsLists.length
      ? [...dsLists.reduce((a, b) => new Set([...a].filter((x) => b.has(x))))]
      : []
    common.sort()

    const rows: Record<string, unknown>[] = common.map((ds) => {
      const row: Record<string, unknown> = { dataset: ds }
      const scores = reportKeys.map((k) => byReport[k][ds] ?? 0)
      const maxScore = Math.max(...scores)
      reportKeys.forEach((k, i) => {
        row[k] = scores[i]
        row[`${k}_best`] = scores[i] === maxScore && maxScore > 0
      })
      return row
    })

    if (common.length > 0) {
      const avgRow: Record<string, unknown> = { dataset: t('compare.average') }
      reportKeys.forEach((k) => {
        const scores = common.map((ds) => byReport[k][ds] ?? 0)
        avgRow[k] = scores.reduce((a, b) => a + b, 0) / scores.length
        avgRow[`${k}_best`] = false
      })
      let bestAvg = -1
      reportKeys.forEach((k) => { if ((avgRow[k] as number) > bestAvg) bestAvg = avgRow[k] as number })
      reportKeys.forEach((k) => { if ((avgRow[k] as number) === bestAvg && bestAvg > 0) avgRow[`${k}_best`] = true })
      rows.push(avgRow)
    }

    const columns = [
      { key: 'dataset', label: t('compare.dataset') },
      ...reportKeys.map((k) => ({ key: k, label: displayNames[k] })),
    ]

    return { scoreTableData: rows, scoreTableColumns: columns, displayNames }
  }, [reports, reportNames, t])

  // ── Early returns (all hooks called above) ──
  if (selection?.backend === 'Perf') {
    return <PerfCompareView taskIds={selection.reports} rootPath={rootPath} />
  }

  if (!selection || selection.reports.length === 0) {
    return <SavedReportsList />
  }

  // ------------------------------------------------------------------ //
  // Render                                                              //
  // ------------------------------------------------------------------ //

  if (reportNames.length < 2) {
    return (
      <div className="page-enter">
        <Breadcrumb items={[{ label: t('reports.title'), href: '/reports' }, { label: t('compare.title') }]} />
        <div className="flex flex-col items-center justify-center gap-4 py-20">
          {/* text-dim allowed: empty-state alert icon (DESIGN.md §Text) */}
          <AlertCircle size={48} className="text-[var(--text-dim)]" />
          <p className="text-[var(--text-muted)] text-lg">{t('compare.needTwo')}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="page-enter flex flex-col gap-6">
      <Breadcrumb items={[{ label: t('reports.title'), href: '/reports' }, { label: t('compare.title') }]} />

      <div className="flex items-center justify-between">
        <Button size="md" className="bg-[var(--accent-dim)] text-[var(--accent)] hover:bg-[var(--accent-glow)]" onClick={() => clearCompareSelection()}>← 返回列表</Button>
        <Button size="md" onClick={async () => {
          const name = reportNames.map((n) => parseReportName(n).model || n).join(' vs ').slice(0, 80)
          try {
            await saveCompareReport(name, reportNames, 'LLM', rootPath)
            toast.success('报告已保存')
          } catch (e) {
            toast.error(e instanceof Error ? e.message : '保存失败')
          }
        }}>保存报告</Button>
      </div>

      {/* Selected Models */}
      <Card title={t('compare.selectedModels')}>
        <div className="flex flex-wrap items-center gap-2">
          {reportNames.map((name) => (
            <FilterChip
              key={name}
              label={displayNames[name] ?? (parseReportName(name).model || name)}
            />
          ))}
          <span className="text-xs text-[var(--text-muted)] ml-auto">共 {reportNames.length} 个模型</span>
        </div>
      </Card>

      {/* Score Content */}
      {loading && !dataLoaded ? (
        <div className="flex flex-col gap-4">
          <Skeleton height={450} />
          <Skeleton height={300} />
        </div>
      ) : (
        <ScoreTab
          rootPath={rootPath}
          reportNames={reportNames}
          scoreTableColumns={scoreTableColumns}
          scoreTableData={scoreTableData}
          displayNames={displayNames}
          t={t}
        />
      )}
    </div>
  )
}

// ------------------------------------------------------------------ //
// Score Comparison Tab                                                //
// ------------------------------------------------------------------ //

function ScoreTab({
  rootPath,
  reportNames,
  scoreTableColumns,
  scoreTableData,
  displayNames,
  t,
}: {
  rootPath: string
  reportNames: string[]
  scoreTableColumns: { key: string; label: string }[]
  scoreTableData: Record<string, unknown>[]
  displayNames: Record<string, string>
  t: (p: string) => string
}) {
  const reportKeys = scoreTableColumns.slice(1).map((c) => c.key).sort()
  const dataRows = scoreTableData.filter((r) => r.dataset !== t('compare.average'))
  const avgRow = scoreTableData.find((r) => r.dataset === t('compare.average')) ?? null
  const datasetNames = dataRows.map((r) => r.dataset as string)

  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden shadow-[var(--shadow-sm)]">
        <div className="flex items-center border-b border-[var(--border)] px-5 py-3">
          <h3 className="type-label-xs">{t('multi.modelScores')}</h3>
        </div>
        {scoreTableData.length === 0 ? (
          // text-dim allowed: non-essential ≥14px metadata (DESIGN.md §Text)
          <div className="py-12 text-center text-sm text-[var(--text-dim)]">{t('common.noData')}</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="text-sm border-collapse w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="px-3 py-2.5 text-left type-table-xs sticky left-0 bg-[var(--bg-card)] z-10 border-r border-[var(--border)] w-32">
                    Model
                  </th>
                  {datasetNames.map((ds) => (
                    <th key={ds} className="py-2.5 text-center type-table-xs whitespace-nowrap w-[100px]">
                      {ds}
                    </th>
                  ))}
                  {avgRow && (
                    <th className="py-2.5 text-center type-table-xs !text-[var(--accent)] whitespace-nowrap border-l border-[var(--border)] w-[100px]">
                      {t('compare.average')}
                    </th>
                  )}
                </tr>
              </thead>
              <tbody>
                {reportKeys.map((rk, rkIdx) => (
                  <tr key={rk} className="hover:bg-[var(--bg-card2)] transition-colors">
                    <td className="px-3 py-2 text-xs font-medium whitespace-nowrap sticky left-0 bg-[var(--bg-card)] z-10 border-r border-[var(--border)] min-w-[160px]">
                      <div className="flex items-center gap-1.5">
                        <span className="inline-block w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: modelColor(rkIdx) }} />
                        <span className="text-[var(--text-muted)]" title={displayNames[rk] ?? rk}>
                          {displayNames[rk] ?? rk}
                        </span>
                      </div>
                    </td>
                    {datasetNames.map((ds) => {
                      const row = dataRows.find((r) => r.dataset === ds)
                      const score = row ? (row[rk] as number) : null
                      const isBest = row ? !!(row[`${rk}_best`]) : false
                      return (
                        <td key={ds} className="px-1 py-1 w-[100px]">
                          {score != null ? (
                            <div className="w-full py-1.5 px-2 rounded-[var(--radius-xs)] text-xs font-mono font-medium text-center text-white" style={{ backgroundColor: scoreColor(score) }}>
                              {isBest && <span className="inline-block w-1.5 h-1.5 rounded-full bg-white mr-1 align-middle opacity-80" />}
                              {(score).toFixed(4)}
                            </div>
                          ) : (
                            // text-dim allowed: em-dash placeholder, decorative non-essential glyph (DESIGN.md §Text)
                            <div className="w-full py-1.5 px-2 text-xs text-center text-[var(--text-dim)] bg-[var(--bg-deep)] rounded-[var(--radius-xs)]">—</div>
                          )}
                        </td>
                      )
                    })}
                    {avgRow && (() => {
                      const score = avgRow[rk] as number
                      const isBest = !!(avgRow[`${rk}_best`])
                      return (
                        <td className="px-1 py-1 border-l border-[var(--border)] w-[100px]">
                          <div className="w-full py-1.5 px-2 rounded-[var(--radius-xs)] text-xs font-mono font-semibold text-center text-white" style={{ backgroundColor: scoreColor(score) }}>
                            {isBest && <span className="inline-block w-1.5 h-1.5 rounded-full bg-white mr-1 align-middle opacity-80" />}
                            {score.toFixed(4)}
                          </div>
                        </td>
                      )
                    })()}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <Card title="得分对比">
        <ResponsiveContainer width="100%" height={320}>
          <RBarChart
            data={dataRows.map((row) => {
              const entry: Record<string, unknown> = { name: row.dataset }
              reportKeys.forEach((k) => { entry[displayNames[k] ?? k] = row[k] })
              return entry
            })}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} tickFormatter={(v) => v.toFixed(2)} />
            <Tooltip formatter={(v: unknown) => Number(v).toFixed(4)} />
            <Legend wrapperStyle={{ fontSize: 12 }} layout="vertical" align="right" verticalAlign="top" />
            {reportKeys.map((rk, i) => (
              <Bar key={rk} dataKey={displayNames[rk] ?? rk} fill={modelColor(i)} radius={[4, 4, 0, 0]} maxBarSize={32} />
            ))}
          </RBarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

// ------------------------------------------------------------------ //
// Prediction Comparison Tab                                           //
// ------------------------------------------------------------------ //

function PredictionTab({
  reportNames,
  displayNames,
  predCommonDatasets,
  selectedDs,
  setSelectedDs,
  subsets,
  selectedSubset,
  setSelectedSubset,
  perModelFilter,
  setPerModelFilter,
  threshold,
  setThreshold,
  passRates,
  mergedPredictions,
  filtered,
  currentRow,
  page,
  setPage,
  totalPages,
  predictionsLoading,
  t,
}: {
  reportNames: string[]
  displayNames: Record<string, string>
  predCommonDatasets: string[]
  selectedDs: string
  setSelectedDs: (ds: string) => void
  subsets: string[]
  selectedSubset: string
  setSelectedSubset: (s: string) => void
  perModelFilter: Record<string, PerModelFilter>
  setPerModelFilter: (f: Record<string, PerModelFilter>) => void
  threshold: number
  setThreshold: (n: number) => void
  passRates: Record<string, number>
  mergedPredictions: MergedPrediction[]
  filtered: MergedPrediction[]
  currentRow: MergedPrediction | null
  page: number
  setPage: (p: number) => void
  totalPages: number
  predictionsLoading: boolean
  t: (p: string) => string
}) {
  // ── Filter helpers ──────────────────────────────────────────────
  const setModelFilter = (name: string, f: PerModelFilter) =>
    setPerModelFilter({ ...perModelFilter, [name]: f })

  const setAllFilters = (f: PerModelFilter) => {
    const next: Record<string, PerModelFilter> = {}
    reportNames.forEach((n) => { next[n] = f })
    setPerModelFilter(next)
  }

  const isAllAny = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'any')
  const isAllPass = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'pass')
  const isAllFail = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'fail')

  // ── Keyboard navigation ─────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'ArrowLeft' && page > 1) setPage(page - 1)
      else if (e.key === 'ArrowRight' && page < totalPages) setPage(page + 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [page, totalPages, setPage])

  if (predCommonDatasets.length === 0) {
    return (
      <Card>
        <div className="flex flex-col items-center justify-center gap-3 py-12">
          {/* text-dim allowed: empty-state alert icon (DESIGN.md §Text) */}
          <AlertCircle size={32} className="text-[var(--text-dim)]" />
          <p className="text-[var(--text-muted)]">{t('compare.noCommon')}</p>
        </div>
      </Card>
    )
  }

  // Preset buttons config
  const presets = [
    { label: t('common.all'), active: isAllAny, onClick: () => { setPerModelFilter({}); setPage(1) } },
    { label: t('compare.allPass'), active: isAllPass, onClick: () => { setAllFilters('pass'); setPage(1) } },
    { label: t('compare.allFail'), active: isAllFail, onClick: () => { setAllFilters('fail'); setPage(1) } },
  ]

  return (
    <div className="flex flex-col gap-4">

      {/* ── Dataset / Subset / Threshold ── */}
      <Card>
        <div className="flex flex-wrap items-end gap-4">
          <div className="min-w-[200px] flex-1">
            <Select
              label={t('compare.selectDataset')}
              options={predCommonDatasets.map((ds) => ({ value: ds, label: ds }))}
              value={selectedDs}
              onChange={(v) => { setSelectedDs(v); setSelectedSubset('') }}
              placeholder={`-- ${t('compare.selectDataset')} --`}
            />
          </div>
          {subsets.length > 0 && (
            <div className="min-w-[200px] flex-1">
              <Select
                label={t('compare.selectSubset')}
                options={subsets.map((s) => ({ value: s, label: s }))}
                value={selectedSubset}
                onChange={setSelectedSubset}
                placeholder={`-- ${t('compare.selectSubset')} --`}
              />
            </div>
          )}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">
              {t('compare.scoreThreshold')}
            </label>
            <input
              type="number"
              value={threshold}
              step={0.01}
              min={0}
              max={1}
              onChange={(e) => { setThreshold(Number(e.target.value)); setPage(1) }}
              className="w-24 px-3 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] focus:outline-none focus:border-[var(--accent)]"
            />
          </div>
        </div>
      </Card>

      {/* ── Per-model Filter Section ── */}
      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4 flex flex-col gap-3">
        {/* Quick preset row */}
        <div className="flex items-center gap-3 flex-wrap">
          <span className="block text-xs font-semibold uppercase tracking-wider text-[var(--text-dim)] mb-1">{t('compare.filterByModel')}</span>
          <div className="inline-flex rounded-[var(--radius-sm)] border border-[var(--border)] overflow-hidden">
            {presets.map(({ label, active, onClick }, idx, arr) => (
              <button
                key={label}
                onClick={onClick}
                className={cn(
                  'px-3.5 py-1.5 type-button-sm transition-colors cursor-pointer',
                  active
                    ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
                    : 'bg-transparent text-[var(--text-muted)] hover:text-[var(--text)]',
                  idx < arr.length - 1 && 'border-r border-[var(--border)]',
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Per-model tri-state chips */}
        <div className="flex flex-col gap-2.5">
          {reportNames.map((name, idx) => {
            const color = modelColor(idx)
            const bg = modelBg(idx)
            const cur = perModelFilter[name] ?? 'any'
            const rate = passRates[name]
            const chips: { key: PerModelFilter; label: string; icon?: ReactNode; activeBg: string }[] = [
              { key: 'any', label: t('compare.any'), activeBg: 'var(--accent)' },
              { key: 'pass', label: t('common.pass'), icon: <CircleCheck size={12} />, activeBg: 'var(--pass)' },
              { key: 'fail', label: t('common.fail'), icon: <CircleX size={12} />, activeBg: 'var(--fail)' },
            ]
            return (
              <div key={name} className="flex items-center gap-3 flex-wrap">
                {/* Model label */}
                <div className="flex items-center gap-1.5 w-36 shrink-0">
                  <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: palette.dot }} />
                  <span
                    className="text-xs font-medium truncate"
                    style={{ color: palette.dot }}
                    title={displayNames[name] ?? (parseReportName(name).model || name)}
                  >
                    {displayNames[name] ?? (parseReportName(name).model || name)}
                  </span>
                </div>

                {/* Tri-state chips */}
                <div className="inline-flex rounded-[var(--radius-sm)] border border-[var(--border)] overflow-hidden">
                  {chips.map(({ key, label, icon, activeBg }, ci, ca) => {
                    const isActive = cur === key
                    return (
                      <button
                        key={key}
                        onClick={() => { setModelFilter(name, key); setPage(1) }}
                        className={cn(
                          'flex items-center gap-1 px-2.5 py-1.5 type-button-sm transition-colors cursor-pointer',
                          !isActive && 'bg-transparent text-[var(--text-muted)] hover:text-[var(--text)]',
                          ci < ca.length - 1 && 'border-r border-[var(--border)]',
                        )}
                        style={isActive ? { background: activeBg, color: 'var(--text-on-filled)' } : undefined}
                      >
                        {icon}
                        {label}
                      </button>
                    )
                  })}
                </div>

                {/* Pass rate badge */}
                {rate !== undefined && mergedPredictions.length > 0 && (
                  <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                    rate >= 0.5 ? 'bg-green-500/10 text-green-500' : 'bg-yellow-500/10 text-yellow-500'
                  }`}>
                    {(rate * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Stats Bar + Pagination ── */}
      {!predictionsLoading && mergedPredictions.length > 0 && (
        <div className="flex items-center justify-between px-4 py-2.5 rounded-[var(--radius)] bg-[var(--bg-card)] border border-[var(--border)] gap-2 flex-wrap">
          <span className="text-sm text-[var(--text-muted)]">
            {t('compare.showing')}{' '}
            <strong className="text-[var(--text)]">{filtered.length}</strong>{' '}
            {t('compare.of')}{' '}
            <strong className="text-[var(--text)]">{mergedPredictions.length}</strong>{' '}
            {t('compare.predictions')}
            {currentRow && (
              <span className="ml-2 text-xs opacity-50">#{currentRow.Index}</span>
            )}
          </span>
          <div className="flex items-center gap-2">
            <button
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
              className="p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--bg-card2)] disabled:opacity-30 transition-colors cursor-pointer disabled:cursor-not-allowed"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="text-sm text-[var(--text-muted)] min-w-[5rem] text-center tabular-nums">
              {t('compare.sample')} {page} / {totalPages}
            </span>
            <button
              disabled={page >= totalPages}
              onClick={() => setPage(page + 1)}
              className="p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--bg-card2)] disabled:opacity-30 transition-colors cursor-pointer disabled:cursor-not-allowed"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* ── Loading skeleton ── */}
      {predictionsLoading && <Skeleton height={400} />}

      {/* ── ChatView Columns ── */}
      {!predictionsLoading && currentRow && (
        <div
          className="grid gap-4"
          style={{
            gridTemplateColumns: `repeat(${reportNames.length}, minmax(0, 1fr))`,
          }}
        >
          {reportNames.map((name, idx) => {
            const color = modelColor(idx)
            const bg = modelBg(idx)
            const modelRow = currentRow.models[name]
            if (!modelRow) return null
            return (
              <div
                key={name}
                className="flex flex-col rounded-[var(--radius)] border overflow-hidden"
                style={{ borderColor: color, background: bg }}
              >
                {/* Column Header */}
                <div
                  className="flex items-center justify-between px-4 py-2.5 border-b shrink-0"
                  style={{ borderColor: color, background: bg }}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: color }}
                    />
                    <span
                      className="text-xs font-semibold truncate"
                      style={{ color }}
                      title={displayNames[name] ?? (parseReportName(name).model || name)}
                    >
                      {displayNames[name] ?? (parseReportName(name).model || name)}
                    </span>
                  </div>
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono font-medium ${
                      (modelRow.NScore ?? 0) >= threshold
                        ? 'bg-green-500/10 text-green-500'
                        : 'bg-red-500/10 text-red-500'
                    }`}
                  >
                    {(modelRow.NScore * 100).toFixed(1)}%
                  </span>
                </div>

                {/* ChatView */}
                <div
                  className="overflow-y-auto p-3"
                  style={{ maxHeight: 'calc(100vh - 380px)', minHeight: '280px' }}
                >
                  <ChatView prediction={modelRow} threshold={threshold} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* ── Empty state ── */}
      {!predictionsLoading && mergedPredictions.length > 0 && filtered.length === 0 && (
        <Card>
          {/* text-dim allowed: non-essential ≥14px metadata (DESIGN.md §Text) */}
          <div className="text-center py-8 text-[var(--text-dim)]">{t('common.noData')}</div>
        </Card>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ //
// Perf Comparison View                                                //
// ------------------------------------------------------------------ //

function PerfCompareView({ taskIds, rootPath }: { taskIds: string[]; rootPath: string }) {
  const { t } = useLocale()
  const { clearCompareSelection } = useCompare()
  const [data, setData] = useState<PerfCompareResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (taskIds.length < 2) {
      setError('选择至少 2 个压测任务进行对比')
      setLoading(false)
      return
    }
    setLoading(true)
    comparePerfReports(taskIds)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : '加载失败'))
      .finally(() => setLoading(false))
  }, [taskIds])

  if (loading) {
    return (
      <div className="page-enter flex flex-col gap-6">
        <Breadcrumb items={[{ label: '压测报告' }, { label: '对比' }]} />
        <div className="flex flex-col gap-4">
          <Skeleton height={120} />
          <Skeleton height={400} />
        </div>
      </div>
    )
  }

  if (error || !data || data.tasks.length < 2) {
    return (
      <div className="page-enter">
        <Breadcrumb items={[{ label: '压测报告' }, { label: '对比' }]} />
        <div className="flex flex-col items-center justify-center gap-4 py-20">
          <AlertCircle size={48} className="text-[var(--text-dim)]" />
          <p className="text-[var(--text-muted)] text-lg">{error || '选择至少 2 个压测任务进行对比'}</p>
        </div>
      </div>
    )
  }

  const models = data.tasks.map((t) => ({
    label: t.model,
    data: t.runs[0]?.summary || {},
    percentiles: t.runs[0]?.percentiles || [],
    errMsg: (t as any).error as string | undefined,
  }))

  const valid = models.filter((m) => Object.keys(m.data).length > 0)
  const missing = models.filter((m) => Object.keys(m.data).length === 0)

  const totalReqs = valid.reduce((s, m) => s + (m.data['Total Requests'] || 0), 0)
  const totalSucc = valid.reduce((s, m) => s + (m.data['Success Requests'] || 0), 0)
  const avgSuccess = totalReqs ? ((totalSucc / totalReqs) * 100).toFixed(1) : '0'
  const avgLatency = valid.length
    ? (valid.reduce((s, m) => s + (m.data['Avg Latency (s)'] || 0), 0) / valid.length).toFixed(2)
    : '0'
  const avgOutputTps = valid.length
    ? (valid.reduce((s, m) => s + (m.data['Output Throughput (tok/s)'] || 0), 0) / valid.length).toFixed(1)
    : '0'

  const kpis = [
    ['已选任务', models.length],
    ['有效数据', valid.length],
    ['总请求数', totalReqs],
    ['平均成功率', `${avgSuccess}%`],
    ['平均延迟', `${avgLatency}s`],
    ['平均输出 TPS', avgOutputTps],
  ]

  const columns = [
    { key: 'model', label: '模型', render: (m: typeof valid[0]) => m.label },
    { key: 'concurrency', label: '并发', render: (m: typeof valid[0]) => m.data['Concurrency'] ?? '-' },
    { key: 'total', label: '请求数', render: (m: typeof valid[0]) => m.data['Total Requests'] ?? '-' },
    { key: 'success', label: '成功', render: (m: typeof valid[0]) => m.data['Success Requests'] ?? '-' },
    { key: 'failed', label: '失败', render: (m: typeof valid[0]) => m.data['Failed Requests'] ?? '0' },
    { key: 'rate', label: '成功率', render: (m: typeof valid[0]) => {
      const sr = m.data['Success Requests'] && m.data['Total Requests']
        ? m.data['Success Requests'] / m.data['Total Requests'] : 1
      return `${(sr * 100).toFixed(1)}%`
    }},
    { key: 'rps', label: 'RPS', render: (m: typeof valid[0]) => (m.data['Req Throughput (req/s)'] ?? 0).toFixed(4) },
    { key: 'rpm', label: 'RPM', render: (m: typeof valid[0]) => `${((m.data['Req Throughput (req/s)'] ?? 0) * 60).toFixed(1)}` },
    { key: 'tpm', label: 'TPM', render: (m: typeof valid[0]) => `${((m.data['Output Throughput (tok/s)'] ?? 0) * 60).toFixed(0)}` },
    { key: 'latency_avg', label: '延迟 Avg(s)', render: (m: typeof valid[0]) => (m.data['Avg Latency (s)'] ?? 0).toFixed(2) },
    { key: 'ttft', label: 'TTFT Avg(ms)', render: (m: typeof valid[0]) => (m.data['TTFT (ms)'] ?? 0).toFixed(1) },
    { key: 'tpot', label: 'TPOT Avg(ms)', render: (m: typeof valid[0]) => (m.data['TPOT (ms)'] ?? 0).toFixed(1) },
    { key: 'output_tps', label: '输出 tok/s', render: (m: typeof valid[0]) => (m.data['Output Throughput (tok/s)'] ?? 0).toFixed(1) },
    { key: 'total_tps', label: '总 tok/s', render: (m: typeof valid[0]) => (m.data['Total Throughput (tok/s)'] ?? 0).toFixed(1) },
  ]

  return (
    <div className="page-enter flex flex-col gap-6">
      <Breadcrumb items={[{ label: '压测报告' }, { label: '对比' }]} />

      <div className="flex items-center justify-between">
        <Button size="md" className="bg-[var(--accent-dim)] text-[var(--accent)] hover:bg-[var(--accent-glow)]" onClick={() => clearCompareSelection()}>← 返回列表</Button>
        <Button size="md" onClick={async () => {
          const name = `${valid.map((m) => m.label).join(' vs ')}`.slice(0, 80)
          try {
            await saveCompareReport(name, taskIds)
            toast.success('报告已保存')
          } catch (e) {
            toast.error(e instanceof Error ? e.message : '保存失败')
          }
        }}>保存报告</Button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
        {kpis.map(([label, value]) => (
          <div key={label} className="px-4 py-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] border-l-[3px] border-l-[var(--accent)]">
            <div className="text-lg font-bold text-[var(--text)]">{value}</div>
            <div className="text-xs text-[var(--text-muted)] mt-0.5">{label}</div>
          </div>
        ))}
      </div>

      {missing.length > 0 && (
        <div className="px-4 py-3 rounded-[var(--radius)] bg-[var(--accent-dim)] border border-[var(--accent)] text-sm text-[var(--text)]">
          以下 {missing.length} 个任务缺少压测数据，已排除在图表和对比表之外：
          <span className="text-[var(--text-muted)] ml-2">
            {missing.map((m) => m.label).join('、')}
          </span>
        </div>
      )}

      <Card title="压测指标对比">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b-2 border-[var(--border)]">
                {columns.map((c) => (
                  <th key={c.key} className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium whitespace-nowrap">
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {valid.map((m, i) => (
                <tr key={i} className="border-b border-[var(--border)] hover:bg-[var(--bg-card2)] transition-colors">
                  {columns.map((c) => (
                    <td key={c.key} className="py-2 px-3 text-[var(--text)] whitespace-nowrap">
                      {c.render(m)}
                    </td>
                  ))}
                </tr>
              ))}
              {missing.map((m, i) => (
                <tr key={`miss-${i}`} className="border-b border-[var(--border)] bg-[var(--bg-deep)] opacity-60">
                  <td className="py-2 px-3 text-[var(--text-muted)] whitespace-nowrap">{m.label}</td>
                  <td colSpan={columns.length - 1} className="py-2 px-3 text-xs text-[var(--text-dim)]">无数据</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-5">
        <Card title="吞吐能力 (RPM)">
          <ResponsiveContainer width="100%" height={280}>
            <RBarChart data={valid.map((m) => ({
              name: m.label.length > 14 ? m.label.slice(0, 12) + '…' : m.label,
              'RPM': (m.data['Req Throughput (req/s)'] || 0) * 60,
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <Tooltip formatter={(v: unknown) => Number(v).toFixed(1)} />
              <Bar dataKey="RPM" radius={[4, 4, 0, 0]}>
                {valid.map((_, i) => <Cell key={i} fill={modelColor(i)} />)}
              </Bar>
            </RBarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="吞吐能力 (TPM)">
          <ResponsiveContainer width="100%" height={280}>
            <RBarChart data={valid.map((m) => ({
              name: m.label.length > 14 ? m.label.slice(0, 12) + '…' : m.label,
              'TPM': (m.data['Output Throughput (tok/s)'] || 0) * 60,
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <Tooltip formatter={(v: unknown) => Number(v).toFixed(0)} />
              <Bar dataKey="TPM" radius={[4, 4, 0, 0]}>
                {valid.map((_, i) => <Cell key={i} fill={modelColor(i)} />)}
              </Bar>
            </RBarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Avg 延迟 (s)">
          <ResponsiveContainer width="100%" height={280}>
            <RBarChart data={valid.map((m) => ({
              name: m.label.length > 14 ? m.label.slice(0, 12) + '…' : m.label,
              '延迟': (m.data['Avg Latency (s)'] || 0),
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} unit="s" />
              <Tooltip formatter={(v: unknown) => `${Number(v).toFixed(2)}s`} />
              <Bar dataKey="延迟" radius={[4, 4, 0, 0]}>
                {valid.map((_, i) => <Cell key={i} fill={modelColor(i)} />)}
              </Bar>
            </RBarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="TTFT 首字延迟 (ms)">
          <ResponsiveContainer width="100%" height={280}>
            <RBarChart data={valid.map((m) => ({
              name: m.label.length > 14 ? m.label.slice(0, 12) + '…' : m.label,
              'TTFT': (m.data['TTFT (ms)'] || 0),
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} unit="ms" />
              <Tooltip formatter={(v: unknown) => `${Number(v).toFixed(0)}ms`} />
              <Bar dataKey="TTFT" radius={[4, 4, 0, 0]}>
                {valid.map((_, i) => <Cell key={i} fill={modelColor(i)} />)}
              </Bar>
            </RBarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="TPOT 生成间隔 (ms)">
          <ResponsiveContainer width="100%" height={280}>
            <RBarChart data={valid.map((m) => ({
              name: m.label.length > 14 ? m.label.slice(0, 12) + '…' : m.label,
              'TPOT': (m.data['TPOT (ms)'] || 0),
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} unit="ms" />
              <Tooltip formatter={(v: unknown) => `${Number(v).toFixed(1)}ms`} />
              <Bar dataKey="TPOT" radius={[4, 4, 0, 0]}>
                {valid.map((_, i) => <Cell key={i} fill={modelColor(i)} />)}
              </Bar>
            </RBarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="成功率 (%)">
          <ResponsiveContainer width="100%" height={280}>
            <RBarChart data={valid.map((m) => {
              const sr = m.data['Success Requests'] && m.data['Total Requests']
                ? (m.data['Success Requests'] / m.data['Total Requests']) * 100 : 100
              return { name: m.label.length > 14 ? m.label.slice(0, 12) + '…' : m.label, '成功率': sr }
            })}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <Tooltip formatter={(v: unknown) => `${Number(v).toFixed(1)}%`} />
              <Bar dataKey="成功率" radius={[4, 4, 0, 0]}>
                {valid.map((_, i) => <Cell key={i} fill={modelColor(i)} />)}
              </Bar>
            </RBarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card title="原始报告">
        <div className="flex flex-wrap gap-2">
          {data.tasks.map((t) => (
            <a
              key={t.task_id}
              href={`/api/v1/perf/report?task_id=${encodeURIComponent(t.task_id)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-sm text-[var(--accent)] hover:bg-[var(--accent-dim)] transition-colors"
            >
              {t.model}
              <ExternalLink size={12} />
            </a>
          ))}
        </div>
      </Card>
    </div>
  )
}

// ------------------------------------------------------------------ //
// Saved Reports List                                                   //
// ------------------------------------------------------------------ //

function SavedReportsList() {
  const [reports, setReports] = useState<SavedCompareReport[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<string>('all')
  const { setCompareSelection } = useCompare()

  const load = useCallback(() => {
    setLoading(true)
    listSavedCompareReports()
      .then((r) => setReports(r.reports || []))
      .catch(() => toast.error('加载报告列表失败'))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { load() }, [load])

  const handleDelete = async (id: number) => {
    if (!window.confirm('确定删除该对比报告？')) return
    try {
      await deleteCompareReport(id)
      toast.success('已删除')
      load()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '删除失败')
    }
  }

  const handleOpen = (r: SavedCompareReport) => {
    const ids = JSON.parse(r.task_ids) as string[]
    setCompareSelection({ reports: ids, rootPath: r.root_path || '', backend: r.backend === 'LLM' ? 'Native' : 'Perf' })
  }

  const filtered = filter === 'all' ? reports : reports.filter((r) => r.backend === filter)

  return (
    <div className="page-enter flex flex-col gap-6">
      <Breadcrumb items={[{ label: '对比报告列表' }]} />

      <Tabs
        tabs={[
          { key: 'all', label: `全部 (${reports.length})` },
          { key: 'Perf', label: `性能压测 (${reports.filter((r) => r.backend === 'Perf').length})` },
          { key: 'LLM', label: `模型评估 (${reports.filter((r) => r.backend === 'LLM').length})` },
        ]}
        activeKey={filter}
        onChange={(k) => setFilter(k as string)}
      />

      {loading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} height={60} />)}
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 py-16 text-[var(--text-dim)]">
          <AlertCircle size={36} />
          <p className="text-sm">暂无保存的对比报告</p>
        </div>
      ) : (
        <Card title={`已保存 ${filtered.length} 份对比报告`}>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b-2 border-[var(--border)]">
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">名称</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">类型</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">任务数</th>
                  <th className="text-left py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">创建时间</th>
                  <th className="text-right py-2.5 px-3 text-xs text-[var(--text-muted)] font-medium">操作</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.id} className="border-b border-[var(--border)] hover:bg-[var(--bg-card2)] transition-colors cursor-pointer" onClick={() => handleOpen(r)}>
                    <td className="py-2.5 px-3 text-[var(--text)]">{r.name}</td>
                    <td className="py-2.5 px-3">
                      <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${
                        r.backend === 'LLM' ? 'bg-[var(--accent-dim)] text-[var(--accent)]' : 'bg-[var(--green-dim)] text-[var(--green)]'
                      }`}>
                        {r.backend === 'LLM' ? '模型评估' : '性能压测'}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-[var(--text-muted)]">{r.task_count}</td>
                    <td className="py-2.5 px-3 text-xs text-[var(--text-dim)]">{r.created_at}</td>
                    <td className="py-2.5 px-3 text-right">
                      <a
                        href={`/api/v1/perf/compare/saved/${r.id}/download`}
                        onClick={(e) => e.stopPropagation()}
                        className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded text-[var(--accent)] hover:bg-[var(--accent-dim)] transition-colors"
                        title="下载"
                      >
                        <Download size={14} />
                      </a>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(r.id) }}
                        className="p-1 rounded cursor-pointer opacity-40 hover:opacity-100 hover:bg-[var(--danger-bg)] hover:text-[var(--danger)] transition-all ml-1"
                        title="删除"
                      >
                        <Trash2 size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
