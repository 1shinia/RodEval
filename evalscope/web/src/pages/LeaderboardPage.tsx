import { useEffect, useMemo, useState, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import Select from '@/components/ui/Select'
import Table from '@/components/ui/Table'
import LoadingSpinner from '@/components/common/LoadingSpinner'
import { toast } from '@/components/common/Toast'
import { Trophy } from 'lucide-react'
import type { LbMeta, LbColumn, LbPeriodOption, LlmLeaderboard, MmLeaderboard } from '@/api/leaderboard'
import {
  getLeaderboardMeta,
  getLlmLeaderboard,
  getMultimodalLeaderboard,
} from '@/api/leaderboard'

type LbType = 'llm' | 'multimodal'

const typeOptions = [
  { value: 'llm', label: 'LLM 榜单' },
  { value: 'multimodal', label: '多模态榜单' },
]

export default function LeaderboardPage() {
  const { locale } = useLocale()
  const [searchParams] = useSearchParams()
  const typeFromUrl = searchParams.get('type') === 'multimodal' ? 'multimodal' : 'llm'

  const [meta, setMeta] = useState<LbMeta | null>(null)
  const [type, setType] = useState<LbType>(typeFromUrl)
  const [llmPeriod, setLlmPeriod] = useState('')
  const [mmTab, setMmTab] = useState('official')
  const [category, setCategory] = useState('')
  const [llm, setLlm] = useState<LlmLeaderboard | null>(null)
  const [mm, setMm] = useState<MmLeaderboard | null>(null)
  const [loading, setLoading] = useState(true)

  // 首次拉取 meta（榜单类型 + LLM 时间段 + 多模态榜单类别）
  useEffect(() => {
    getLeaderboardMeta()
      .then((m) => {
        setMeta(m)
        setLlmPeriod((p) => p || m.default_llm || '')
        setMmTab((t) => (m.vlm_tabs?.some((x) => x.value === t) ? t : m.vlm_tabs?.[0]?.value ?? 'official'))
      })
      .catch((e) => { toast.error(e instanceof Error ? e.message : '加载榜单失败') })
      .finally(() => setLoading(false))
  }, [])

  // 按当前类型拉数据（切换下拉自动触发）
  useEffect(() => {
    if (!meta) return
    if (type === 'llm' && llmPeriod) {
      let stale = false
      getLlmLeaderboard(llmPeriod)
        .then((d) => {
          if (stale) return
          setLlm(d)
          setCategory((c) => (d.tabs?.some((t) => t.key === c) ? c : d.tabs?.[0]?.key ?? ''))
        })
        .catch((e) => { toast.error(e instanceof Error ? e.message : '加载 LLM 榜单失败') })
      return () => { stale = true }
    }
    if (type === 'multimodal' && mmTab) {
      let stale = false
      getMultimodalLeaderboard(mmTab)
        .then((d) => { if (!stale) setMm(d) })
        .catch((e) => { toast.error(e instanceof Error ? e.message : '加载多模态榜单失败') })
      return () => { stale = true }
    }
    return undefined
  }, [meta, type, llmPeriod, mmTab])

  const llmPeriods: LbPeriodOption[] = useMemo(() => meta?.llm_periods ?? [], [meta])
  const mmTabs: LbPeriodOption[] = useMemo(() => meta?.vlm_tabs ?? [], [meta])

  const changePeriod = (v: string) => setLlmPeriod(v)
  const changeType = (v: string) => setType(v === 'multimodal' ? 'multimodal' : 'llm')

  // 当前激活的数据表与列
  const activeTable = type === 'llm'
    ? (llm?.tables.find((t) => t.key === category) ?? llm?.tables[0] ?? null)
    : null
  const activeCols: LbColumn[] = type === 'llm'
    ? (activeTable?.columns ?? [])
    : (mm?.columns ?? [])
  const rows: Record<string, unknown>[] = type === 'llm'
    ? (activeTable?.rows ?? [])
    : (mm?.rows ?? [])

  // 把后端列配置映射成 Table 组件可用的列（数值列可排序，文本列只展示）
  const buildTableCols = useCallback(
    (cols: LbColumn[], rowData: Record<string, unknown>[]) => cols.map((c) => {
      const sample = rowData.find((r) => typeof r[c.key] === 'number')
      const isNum = sample !== undefined
      return {
        key: c.key,
        label: locale === 'zh' ? c.zh || c.en : c.en || c.zh,
        sortable: isNum,
        render: (row: Record<string, unknown>) => {
          const v = row[c.key]
          if (isNum) return <span className="tabular-nums font-medium">{v as number}</span>
          if (Array.isArray(v)) return <span>{v[0] ?? ''}</span>
          if (v == null || v === '') return <span className="text-[var(--text-dim)]">—</span>
          return <span>{String(v)}</span>
        },
      }
    }),
    [locale],
  )

  // 多模态默认按「最常展示的均分列」排序（image-vlm 用 Avg_Score_* 前缀）
  const mmSortKey = useMemo(
    () => (mm?.columns.find((c) => c.key.startsWith('Avg_Score'))?.key ?? 'Avg Score') as string,
    [mm],
  )

  if (loading && !meta) return <LoadingSpinner />

  const llmTabs = llm?.tabs ?? []

  return (
    <div className="page-enter space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h1 className="flex items-center gap-2 text-xl font-semibold">
          <Trophy size={20} className="text-[var(--accent)]" />
          评测榜单
        </h1>
      </div>

      {/* 下拉框：榜单类型 / 分类维度(LLM) / 时间段(LLM) / 榜单类别(多模态) */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5 flex-wrap">
        <div className="w-44">
          <Select label="榜单类型" options={typeOptions} value={type} onChange={changeType} />
        </div>

        {type === 'llm' && (
          <div className="w-44">
            <Select
              label="分类维度"
              options={llmTabs.map((t) => ({ value: t.key, label: locale === 'zh' ? t.zh : t.en }))}
              value={category || undefined}
              onChange={setCategory}
            />
          </div>
        )}

        {type === 'llm' && llmPeriods.length > 0 && (
          <div className="w-56">
            <Select label="时间段" options={llmPeriods} value={llmPeriod} onChange={changePeriod} />
          </div>
        )}

        {type === 'multimodal' && mmTabs.length > 0 && (
          <div className="w-56">
            <Select label="榜单" options={mmTabs} value={mmTab} onChange={setMmTab} />
          </div>
        )}

        {type === 'llm' && llm && (
          <div className="flex items-center gap-2 pt-5 text-xs text-[var(--text-muted)]">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[var(--accent-dim)] text-[var(--accent)]">
              {llmTabs.length} 个分类维度
            </span>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[var(--bg-card2)] text-[var(--text-muted)]">
              {activeCols.length} 列
            </span>
          </div>
        )}
      </div>

      {/* 数据表 */}
      {meta && (
        type === 'llm'
          ? (llm ? (
              <Table
                columns={buildTableCols(activeCols, rows)}
                data={rows}
                defaultSort={{ key: 'Average', dir: 'desc' }}
              />
            ) : <LoadingSpinner />)
          : (mm ? (
              <Table
                columns={buildTableCols(activeCols, rows)}
                data={rows}
                defaultSort={{ key: mmSortKey, dir: 'desc' }}
              />
            ) : <LoadingSpinner />)
      )}

      {/* 空态 */}
      {meta && rows.length === 0 && !loading && (
        <div className="text-center py-16 text-sm text-[var(--text-muted)]">
          未找到榜单数据（可能缺少 OpenCompass 快照）
        </div>
      )}
    </div>
  )
}