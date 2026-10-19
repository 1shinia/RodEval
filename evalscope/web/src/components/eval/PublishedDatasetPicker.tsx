import { useEffect, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listPublishedDatasets, type PublishedDataset } from '@/api/datasetCatalog'

interface Props {
  categories: PublishedDataset['category'][]
  value: string
  onChange: (value: string) => void
  onEntriesChange?: (entries: PublishedDataset[]) => void
}

const datasetKey = (entry: PublishedDataset) => JSON.stringify([entry.category, entry.benchmark])

export default function PublishedDatasetPicker({ categories, value, onChange, onEntriesChange }: Props) {
  const { locale } = useLocale()
  const zh = locale === 'zh'
  const [items, setItems] = useState<PublishedDataset[]>([])
  const [selectedKeys, setSelectedKeys] = useState<string[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  useEffect(() => {
    let active = true
    listPublishedDatasets().then(data => { if (active) setItems(data.items) })
      .catch(e => { if (active) setError(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])
  const categoryKey = categories.join(',')
  const pool = useMemo(() => items.filter(item => item.published && categoryKey.split(',').includes(item.category)), [items, categoryKey])
  useEffect(() => {
    const names = value.split(/[,，]/).map(name => name.trim()).filter(Boolean)
    setSelectedKeys(previous => {
      const next = previous.filter(key => pool.some(item => datasetKey(item) === key && names.includes(item.benchmark)))
      for (const name of names) {
        const matches = pool.filter(item => item.benchmark === name)
        if (matches.length === 1 && !next.includes(datasetKey(matches[0]))) next.push(datasetKey(matches[0]))
      }
      return next.length === previous.length && next.every((key, i) => key === previous[i]) ? previous : next
    })
  }, [pool, value])
  const selected = useMemo(() => pool.filter(item => selectedKeys.includes(datasetKey(item)) && value.split(/[,，]/).some(name => name.trim() === item.benchmark)), [pool, selectedKeys, value])
  useEffect(() => { onEntriesChange?.(selected) }, [selected, onEntriesChange])
  const names = value.split(/[,，]/).map(name => name.trim()).filter(Boolean)
  const unresolved = names.filter(name => !selected.some(item => item.benchmark === name))
  const filtered = pool.filter(item => `${item.category} ${item.benchmark}`.toLowerCase().includes(search.toLowerCase()))
  const toggle = (entry: PublishedDataset) => {
    const next = selected.some(item => datasetKey(item) === datasetKey(entry))
      ? selected.filter(item => datasetKey(item) !== datasetKey(entry))
      : [...selected, entry]
    setSelectedKeys(next.map(datasetKey))
    onChange([...new Set(next.map(item => item.benchmark))].join(', '))
  }
  return <div className="md:col-span-2 space-y-3 rounded-xl border border-[var(--border)] p-4 text-sm">
    <div className="font-medium">{zh ? '已发布数据集' : 'Published datasets'} <span className="text-red-500">*</span></div>
    <p className="text-[var(--text-muted)]">{zh ? '只使用管理员已发布的数据集，可多选' : 'Use administrator-published datasets only; multiple selections allowed.'}</p>
    {selected.length > 0 && <div className="space-y-2 rounded-lg bg-[var(--bg)] p-3">
      <div className="font-medium">{zh ? '已选择' : 'Selected'} ({selected.length})</div>
      <div className="flex flex-wrap gap-2">
        {selected.map(item => <button key={datasetKey(item)} type="button"
          onClick={() => toggle(item)}
          aria-label={zh ? `移除 ${item.benchmark}` : `Remove ${item.benchmark}`}
          className="inline-flex max-w-full items-center gap-2 rounded-lg border border-[var(--border)] bg-white px-3 py-2 text-sm text-[var(--text)] hover:border-[var(--accent)] focus-visible:outline-2 focus-visible:outline-[var(--accent)]">
          <span className="min-w-0 break-all text-left">{item.benchmark}</span>
          <span aria-hidden="true" className="shrink-0 text-[var(--text-muted)]">×</span>
        </button>)}
      </div>
    </div>}
    <input className="w-full rounded-lg border border-[var(--border)] bg-[var(--bg-card)] px-3 py-2" value={search} onChange={e => setSearch(e.target.value)} placeholder={zh ? '搜索数据集或类别' : 'Search datasets or categories'} aria-label={zh ? '搜索已发布数据集' : 'Search published datasets'} />
    {error && <p role="alert" className="text-red-600">{error}</p>}
    {loading ? <p role="status">{zh ? '加载中…' : 'Loading…'}</p> : error ? null : !pool.length ? <p className="text-[var(--text-muted)]">{zh ? '暂无已发布数据集，请联系管理员准备并发布。' : 'No published datasets. Contact an administrator to prepare and publish data.'}</p> : !filtered.length ? <p>{zh ? '没有匹配的数据集。' : 'No matching datasets.'}</p> : <div className="max-h-64 overflow-auto divide-y divide-[var(--border)]">{filtered.map(item => <label key={datasetKey(item)} className="flex gap-3 items-start py-3 cursor-pointer"><input type="checkbox" checked={selected.some(entry => datasetKey(entry) === datasetKey(item))} onChange={() => toggle(item)} className="mt-1 accent-[var(--accent)]" /><span><span className="font-medium">{item.benchmark}</span><span className="block text-[var(--text-muted)] mt-1">{item.category.toUpperCase()}</span></span></label>)}</div>}
    {!loading && !error && unresolved.length > 0 && <div role="alert" className="text-red-600">{zh ? '请选择以下名称对应的已发布数据集（若不可用请移除）：' : 'Select a published dataset for these names (or remove unavailable names): '}{unresolved.join(', ')} <button type="button" className="underline" onClick={() => onChange([...new Set(selected.map(item => item.benchmark))].join(', '))}>{zh ? '移除未选择项' : 'Remove unresolved selections'}</button></div>}
  </div>
}
