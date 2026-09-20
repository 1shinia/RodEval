import { useEffect, useMemo, useState, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import Select from '@/components/ui/Select'
import Table from '@/components/ui/Table'
import LoadingSpinner from '@/components/common/LoadingSpinner'
import { toast } from '@/components/common/Toast'
import { Trophy } from 'lucide-react'
import type { LbMeta, LbColumn, LbPeriodOption, LlmLeaderboard, MmLeaderboard } from '@/api/leaderboard'
import { getLeaderboardMeta, getLlmLeaderboard, getBoard } from '@/api/leaderboard'

export default function LeaderboardPage() {
  const { locale } = useLocale()
  const [searchParams] = useSearchParams()
  const urlType = searchParams.get('type') ?? 'llm'

  const [meta, setMeta] = useState<LbMeta | null>(null)
  const [type, setType] = useState<string>(urlType === 'llm' ? 'llm' : urlType)
  const [rawOssTab, setRawOssTab] = useState('')
  const [llmPeriod, setLlmPeriod] = useState('')
  const [category, setCategory] = useState('')
  const [llm, setLlm] = useState<LlmLeaderboard | null>(null)
  const [oss, setOss] = useState<MmLeaderboard | null>(null)
  const [loading, setLoading] = useState(true)

  // 首次拉取 meta（榜单类型清单 + LLM 时间段 + 各榜页签）
  useEffect(() => {
    getLeaderboardMeta()
      .then((m) => {
        setMeta(m)
        setLlmPeriod((p) => p || m.default_llm || '')
      })
      .catch((e) => { toast.error(e instanceof Error ? e.message : '加载榜单失败') })
      .finally(() => setLoading(false))
  }, [])

  // 有效榜单类型：URL 里带非法 id 时回退 llm（渲染期派生，不在 effect 里改状态）
  const validType = useMemo(() => {
    if (type === 'llm') return 'llm'
    return meta?.boards.some((b) => b.id === type) ? type : 'llm'
  }, [meta, type])

  // 榜单类型选项（llm 在最前，其余按后端返回顺序）
  const typeOptions: LbPeriodOption[] = useMemo(
    () => (meta?.boards ?? []).map((b) => ({ value: b.id, label: b.label })),
    [meta],
  )

  // 当前 OSS 榜（非 llm）的注册项与页签
  const currentBoard = useMemo(
    () => (validType === 'llm' ? null : (meta?.boards.find((b) => b.id === validType) ?? null)),
    [meta, validType],
  )
  const ossTabs: LbPeriodOption[] = useMemo(() => currentBoard?.tabs ?? [], [currentBoard])

  // 当前有效页签：所选页签对当前榜单无效时回退到该榜第一个（渲染期派生）
  const ossTab = useMemo(() => {
    if (!currentBoard?.tabs?.length) return ''
    return currentBoard.tabs.some((x) => x.value === rawOssTab)
      ? rawOssTab
      : currentBoard.tabs[0].value
  }, [currentBoard, rawOssTab])

  // 切换榜单类型：清掉旧数据
  const changeType = (v: string) => {
    if (v === validType) return
    setType(v)
    setLlm(null)
    setOss(null)
  }

  // 按当前类型拉数据（切换下拉自动触发）
  useEffect(() => {
    if (!meta) return
    if (validType === 'llm' && llmPeriod) {
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
    if (validType !== 'llm' && ossTab) {
      let stale = false
      getBoard(validType, ossTab)
        .then((d) => { if (!stale) setOss(d) })
        .catch((e) => { toast.error(e instanceof Error ? e.message : '加载榜单失败') })
      return () => { stale = true }
    }
    return undefined
  }, [meta, validType, llmPeriod, ossTab])

  const llmPeriods: LbPeriodOption[] = useMemo(() => meta?.llm_periods ?? [], [meta])

  const changePeriod = (v: string) => setLlmPeriod(v)

  // 当前激活的数据表与列
  const activeTable = validType === 'llm'
    ? (llm?.tables.find((t) => t.key === category) ?? llm?.tables[0] ?? null)
    : null
  const activeCols: LbColumn[] = validType === 'llm'
    ? (activeTable?.columns ?? [])
    : (oss?.columns ?? [])
  const rows: Record<string, unknown>[] = validType === 'llm'
    ? (activeTable?.rows ?? [])
    : (oss?.rows ?? [])

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

  // OSS 榜默认按「上游标记 sorter=descend」的主分列排序（ai4science=Score、agent=Avg_Score、
  // physical=Avg_Success_Rate、image-vlm=Avg_Score_wo_Agent），回退到 Avg_Score 前缀猜
  const ossSortKey = useMemo(
    () => (oss?.columns.find((c) => c.sorter === 'descend')?.key
      ?? oss?.columns.find((c) => c.key.startsWith('Avg_Score'))?.key
      ?? 'Avg Score') as string,
    [oss],
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

      {/* 下拉框：榜单类型 / 分类维度(LLM) / 时间段(LLM) / 子榜单(其余) */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5 flex-wrap">
        <div className="w-44">
          <Select label="榜单类型" options={typeOptions} value={validType} onChange={changeType} />
        </div>

        {validType === 'llm' && (
          <div className="w-44">
            <Select
              label="分类维度"
              options={llmTabs.map((t) => ({ value: t.key, label: locale === 'zh' ? t.zh : t.en }))}
              value={category || undefined}
              onChange={setCategory}
            />
          </div>
        )}

        {validType === 'llm' && llmPeriods.length > 0 && (
          <div className="w-56">
            <Select label="时间段" options={llmPeriods} value={llmPeriod} onChange={changePeriod} />
          </div>
        )}

        {validType !== 'llm' && ossTabs.length > 0 && (
          <div className="w-56">
            <Select label="榜单" options={ossTabs} value={ossTab} onChange={setRawOssTab} />
          </div>
        )}

        {validType === 'llm' && llm && (
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
        validType === 'llm'
          ? (llm ? (
              <Table
                columns={buildTableCols(activeCols, rows)}
                data={rows}
                defaultSort={{ key: 'Average', dir: 'desc' }}
              />
            ) : <LoadingSpinner />)
          : (oss ? (
              <Table
                columns={buildTableCols(activeCols, rows)}
                data={rows}
                defaultSort={{ key: ossSortKey, dir: 'desc' }}
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
