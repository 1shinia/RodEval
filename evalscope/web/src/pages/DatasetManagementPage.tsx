import { useEffect, useState } from 'react'
import { Database, Search, RefreshCw, X, FolderSearch } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { listInventory, scanCatalog, reviewDataset, publishDataset as setDatasetPublication, type CatalogEntry, type CatalogDefinition } from '@/api/datasetCatalog'

export default function DatasetManagementPage() {
  const { locale } = useLocale()
  const zh = locale === 'zh'
  const [items, setItems] = useState<CatalogDefinition[]>([])
  const [query, setQuery] = useState('')
  const [unmapped, setUnmapped] = useState<CatalogEntry[]>([])
  const [category, setCategory] = useState('')
  const [publication, setPublication] = useState('')
  const [preparation, setPreparation] = useState('')
  const [loading, setLoading] = useState(true)
  const [loadFailed, setLoadFailed] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<CatalogEntry | null>(null)
  const [page, setPage] = useState(1)
  const load = async () => {
    setLoading(true)
    try {
      const data = await listInventory()
      setItems(data.items)
      setUnmapped(data.unmapped)
      setSelected(old => old ? data.items.flatMap(item => item.versions).find(item => item.id === old.id) ?? null : null)
      setLoadFailed(false)
    } catch (e) {
      setLoadFailed(true)
      throw e
    } finally { setLoading(false) }
  }
  useEffect(() => { load().catch(e => setError(e.message)) }, [])
  const run = async (key: string, action: () => Promise<unknown>, reload = true) => {
    if (busy) return
    setBusy(key); setError('')
    try {
      await action()
      if (reload && key !== 'refresh') await load()
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { setBusy('') }
  }
  const markDataset = (item: CatalogDefinition, passed: boolean) => run(`${item.category}:${item.benchmark}`, async () => {
    const result = await reviewDataset(item.category, item.benchmark, passed)
    const update = (current: CatalogDefinition): CatalogDefinition => ({
      ...current,
      manual_review: result.manual_review,
      published: passed ? current.published : false,
      versions: passed ? current.versions : current.versions.map(version => ({ ...version, published: false })),
    })
    setItems(current => current.map(entry => entry.category === item.category && entry.benchmark === item.benchmark ? update(entry) : entry))
    setSelected(current => current && current.category === item.category && current.benchmark === item.benchmark && !passed ? { ...current, published: false } : current)
  }, false)
  const publishDataset = (item: CatalogDefinition) => run(`publish:${item.category}:${item.benchmark}`, async () => {
    const result = await setDatasetPublication(item.category, item.benchmark, !item.published)
    setItems(current => current.map(entry => entry.category === item.category && entry.benchmark === item.benchmark
      ? { ...entry, published: result.published, manual_review: result.manual_review } : entry))
  }, false)
  const label = (item: { category: CatalogEntry['category'] }) => ({ llm: zh ? '文本' : 'Text', vlm: zh ? '多模态' : 'Multimodal', rag: zh ? '向量检索' : 'Vector Retrieval', agent: zh ? '智能体' : 'Agent' })[item.category]
  const status = (item: { status: string }) => {
    const labels: Record<string, string> = zh
      ? { not_registered: '未发现已登记缓存', discovered: '待校验', candidate: '待校验', available: '校验通过', ready: '校验通过', valid: '校验通过', missing: '缓存缺失', invalid: '校验未通过', unsupported: '暂不支持发布', stale: '缓存已变化' }
      : { not_registered: 'No registered cache', discovered: 'Needs validation', candidate: 'Needs validation', available: 'Validated', ready: 'Validated', valid: 'Validated', missing: 'Missing cache', invalid: 'Invalid', unsupported: 'Publishing unsupported', stale: 'Cache changed' }
    return labels[item.status] ?? item.status
  }
  const preparationState = (item: CatalogDefinition) => {
    if (item.manual_review) return item.manual_review.passed ? 'validated' : 'rejected'
    if (item.versions.some(version => version.status === 'available')) return 'validated'
    if (item.versions.some(version => version.details?.manual_review?.passed === false)) return 'rejected'
    if (item.historical_test) return 'tested'
    return 'unconfirmed'
  }
  const preparationLabel = (state: string) => {
    const labels: Record<string, string> = zh
      ? { unconfirmed: '待确认', tested: '有历史测试记录', validated: '校验通过', rejected: '校验不通过' }
      : { unconfirmed: 'Unconfirmed', tested: 'Historical test recorded', validated: 'Validated', rejected: 'Rejected' }
    return labels[state] ?? state
  }
  const visible = items.filter(item =>
    `${item.benchmark} ${item.dataset_id}`.toLowerCase().includes(query.toLowerCase()) &&
    (!category || item.category === category) &&
    (!publication || item.published === (publication === 'published')) &&
    (!preparation || preparationState(item) === preparation))
  const pageSize = 50
  const pageCount = Math.max(1, Math.ceil(visible.length / pageSize))
  const currentPage = Math.min(page, pageCount)
  const pageItems = visible.slice((currentPage - 1) * pageSize, currentPage * pageSize)
  const button = 'inline-flex items-center justify-center gap-2 rounded-lg border border-[var(--border)] px-3 py-2 text-sm disabled:cursor-not-allowed hover:bg-[var(--bg-card2)]'
  const field = 'rounded-lg border border-[var(--border)] bg-white text-[var(--text)] px-3 py-2 text-sm'

  return <div className="max-w-[1400px] mx-auto p-6 space-y-6 text-sm">
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div><h1 className="text-2xl font-semibold flex items-center gap-3"><Database size={25} />{zh ? '数据集管理' : 'Dataset management'}</h1>
        <p className="mt-2 text-[var(--text-muted)]">{zh ? '复用已有缓存，校验后发布。普通用户只能使用已发布的数据集。' : 'Reuse existing cache, validate, then publish. Users can only use published datasets.'}</p></div>
      <div className="flex gap-2">
        <button className={button} disabled={!!busy || loading} onClick={() => run('refresh', load)}><RefreshCw size={16} />{zh ? '刷新' : 'Refresh'}</button>
        <button className={button + ' bg-[var(--accent)] text-white'} disabled={!!busy} onClick={() => run('scan', scanCatalog)}><FolderSearch size={16} />{busy === 'scan' ? (zh ? '扫描中…' : 'Scanning…') : (zh ? '扫描已有缓存' : 'Scan existing cache')}</button>
      </div>
    </header>
    <div className="rounded-xl border border-[var(--border)] p-4 text-[var(--text-muted)]">
      <div className="mt-2">{(['llm', 'vlm', 'rag', 'agent'] as const).map(value => <span className="mr-5" key={value}>{label({ category: value })}：{items.filter(item => item.category === value).length}</span>)}</div>
    </div>
    {error && <div role="alert" className="rounded-lg border border-red-200 bg-red-50 text-red-700 p-3">{error}</div>}
    <div className="flex flex-wrap gap-3">
      <div className="relative flex-1 min-w-60"><Search size={17} className="absolute left-3 top-3 text-[var(--text-muted)]" /><input className={field + ' w-full pl-10'} aria-label={zh ? '搜索数据集' : 'Search datasets'} placeholder={zh ? '搜索数据集名称或标识' : 'Search name or dataset ID'} value={query} onChange={e => { setQuery(e.target.value); setPage(1) }} /></div>
      <select className={field} aria-label={zh ? '类型' : 'Category'} value={category} onChange={e => { setCategory(e.target.value); setPage(1) }}><option value="">{zh ? '全部类型' : 'All categories'}</option><option value="llm">{zh ? '文本' : 'Text'}</option><option value="vlm">{zh ? '多模态' : 'Multimodal'}</option><option value="rag">{label({ category: 'rag' })}</option><option value="agent">{label({ category: 'agent' })}</option></select>
      <select className={field} aria-label={zh ? '发布状态' : 'Publication'} value={publication} onChange={e => { setPublication(e.target.value); setPage(1) }}><option value="">{zh ? '全部发布状态' : 'All publication states'}</option><option value="published">{zh ? '已发布' : 'Published'}</option><option value="unpublished">{zh ? '未发布' : 'Unpublished'}</option></select>
      <select className={field} aria-label={zh ? '准备状态' : 'Preparation'} value={preparation} onChange={e => { setPreparation(e.target.value); setPage(1) }}><option value="">{zh ? '全部准备状态' : 'All preparation states'}</option>{['unconfirmed', 'validated', 'rejected'].map(state => <option key={state} value={state}>{preparationLabel(state)}</option>)}</select>
    </div>
    <div className="overflow-x-auto rounded-xl border border-[var(--border)] bg-[var(--bg-card)]">
      <table className="w-full min-w-[1120px] table-fixed text-left [&_td]:align-top [&_td]:[overflow-wrap:anywhere] [&_th]:whitespace-nowrap [&_button]:whitespace-nowrap"><colgroup><col className="w-[27%]" /><col className="w-[8%]" /><col className="w-[23%]" /><col className="w-[10%]" /><col className="w-[32%]" /></colgroup><thead className="bg-[var(--bg-card2)] text-[var(--text-muted)]"><tr>{(zh ? ['数据集 / 版本', '类型', '准备状态', '发布状态', '操作'] : ['Dataset / version', 'Category', 'Preparation', 'Publication', 'Actions']).map(text => <th className="p-4 font-medium" key={text}>{text}</th>)}</tr></thead>
        <tbody>{!loading && !loadFailed && pageItems.map(item => <tr key={`${item.category}:${item.benchmark}`} className={`border-t border-[var(--border)] ${busy === `${item.category}:${item.benchmark}` || busy === `publish:${item.category}:${item.benchmark}` ? '[&_button:disabled]:opacity-50' : ''}`}>
          <td className="p-4"><div className="font-medium">{item.benchmark}</div><div className="mt-1 text-[var(--text-muted)]">{item.dataset_id}{item.task_type ? ` · ${item.task_type}` : ''}</div>{!!item.dependencies?.length && <details className="mt-2 text-[var(--text-muted)]"><summary className="cursor-pointer">{zh ? '依赖缓存' : 'Dependency cache'} ({item.dependencies.length})</summary>{item.dependencies.map(dependency => <p key={dependency.id}>{dependency.dataset_id}</p>)}<p>{zh ? '已确认用途；尚未验证当前加载器能否复用，不代表主数据集校验通过。' : 'Purpose identified; reuse by the current loader is unverified. This does not validate the main dataset.'}</p></details>}</td>
          <td className="p-4">{label(item)}</td><td className="p-4"><span title={zh ? '待确认不代表未下载；历史测试只证明当时有样本结果，不证明当前缓存完整；校验通过不等于已发布。' : 'Unconfirmed does not mean not downloaded. Historical results do not verify current cache completeness. Validation is separate from publication.'}>{preparationLabel(preparationState(item))}</span>{item.historical_test && <details className="mt-2 text-[var(--text-muted)]"><summary className="cursor-pointer">{zh ? '历史测试依据' : 'Historical evidence'}</summary><p>{zh ? '最近报告样本数' : 'Latest report samples'}: {item.historical_test.samples}</p><p>{new Date(item.historical_test.timestamp * 1000).toLocaleString(zh ? 'zh-CN' : 'en-US')}</p><p className="break-all">{item.historical_test.report}</p></details>}</td><td className="p-4">{item.published ? (zh ? '已发布' : 'Published') : (zh ? '未发布' : 'Unpublished')}</td>
          <td className="p-4 space-y-3">
            <div className="flex flex-wrap gap-2">
              <button className={button + ' !border-green-200 bg-green-50 text-green-700 hover:!bg-green-100'} disabled={!!busy || loading} onClick={() => markDataset(item, true)}>{zh ? '校验通过' : 'Mark passed'}</button>
              <button className={button + ' !border-red-200 bg-red-50 text-red-700 hover:!bg-red-100'} disabled={!!busy || loading} onClick={() => {
                if (item.published && !window.confirm(zh ? '校验不通过会同时取消该数据集的发布，确定继续？' : 'This will also unpublish this dataset. Continue?')) return
                void markDataset(item, false)
              }}>{zh ? '校验不通过' : 'Mark failed'}</button>
              <button className={button + (item.published ? ' !border-gray-200 bg-gray-50 text-gray-700 hover:!bg-gray-100' : ' !border-violet-200 bg-violet-50 text-violet-700 hover:!bg-violet-100')} disabled={!!busy || loading} onClick={() => publishDataset(item)}>{busy === `publish:${item.category}:${item.benchmark}` ? (item.published ? (zh ? '取消中…' : 'Unpublishing…') : (zh ? '发布中…' : 'Publishing…')) : item.published ? (zh ? '取消发布' : 'Unpublish') : (zh ? '发布' : 'Publish')}</button>
            </div>
            {item.manual_review && <p className="text-[var(--text-muted)]">{zh ? '人工确认' : 'Manual review'} · {item.manual_review.passed ? (zh ? '通过' : 'Passed') : (zh ? '不通过' : 'Failed')} · {zh ? '操作人编号' : 'User ID'} {item.manual_review.user_id} · {new Date(item.manual_review.at).toLocaleString(zh ? 'zh-CN' : 'en-US')}</p>}
          </td>
        </tr>)}</tbody></table>
      {loading ? <div role="status" className="py-16 text-center">{zh ? '正在加载数据集…' : 'Loading datasets…'}</div> : loadFailed ? <div className="py-16 text-center text-red-600">{zh ? '列表加载失败，请查看错误详情并刷新重试。' : 'Failed to load datasets. Review the error and refresh to retry.'}</div> : !visible.length && <div className="py-16 px-6 text-center text-[var(--text-muted)]"><Database size={32} className="mx-auto mb-3" /><p>{items.length ? (zh ? '没有匹配的数据集。' : 'No matching datasets.') : (zh ? '暂无登记记录。点击“扫描已有缓存”开始；不会自动下载数据。' : 'No registered datasets. Scan existing cache to begin; no data will be downloaded.')}</p></div>}
    </div>
    {!loading && !loadFailed && visible.length > 0 && <nav aria-label={zh ? '数据集分页' : 'Dataset pagination'} className="flex flex-wrap items-center justify-between gap-3">
      <p className="text-[var(--text-muted)]">{zh ? `共 ${visible.length} 条，每页 ${pageSize} 条，当前显示 ${(currentPage - 1) * pageSize + 1}–${Math.min(currentPage * pageSize, visible.length)} 条` : `${visible.length} results · ${pageSize} per page · showing ${(currentPage - 1) * pageSize + 1}–${Math.min(currentPage * pageSize, visible.length)}`}</p>
      <div className="flex items-center gap-3"><button className={button} disabled={currentPage === 1} onClick={() => setPage(currentPage - 1)}>{zh ? '上一页' : 'Previous'}</button><span>{currentPage} / {pageCount}</span><button className={button} disabled={currentPage === pageCount} onClick={() => setPage(currentPage + 1)}>{zh ? '下一页' : 'Next'}</button></div>
    </nav>}
    {unmapped.length > 0 && <details className="rounded-xl border border-[var(--border)] p-4"><summary className="cursor-pointer">{zh ? '待确认缓存' : 'Caches to review'} ({unmapped.length})</summary><p className="mt-3 text-[var(--text-muted)]">{zh ? '这些缓存尚未关联到当前数据集目录，点击查看原因。' : 'These caches are not linked to the current dataset catalog. Select one to view the reason.'}</p>{unmapped.map(version => <button key={version.id} className={button + ' mt-3 mr-3'} onClick={() => setSelected(version)}>{version.dataset_id}</button>)}</details>}
    {selected && <div className="fixed inset-0 z-50 bg-black/20 flex justify-end" onClick={() => setSelected(null)}><section role="dialog" aria-modal="true" aria-label={zh ? '数据集详情' : 'Dataset details'} onClick={e => e.stopPropagation()} className="w-full max-w-xl h-full bg-[var(--bg-card)] shadow-xl p-6 overflow-y-auto">
      <div className="flex justify-between items-center mb-6"><h2 className="text-2xl font-semibold">{selected.benchmark}</h2><button aria-label={zh ? '关闭' : 'Close'} onClick={() => setSelected(null)}><X /></button></div>
      <dl className="space-y-5">{[
        [zh ? '准备状态' : 'Preparation', status(selected)], [zh ? '发布状态' : 'Publication', selected.published ? (zh ? '已发布' : 'Published') : (zh ? '未发布' : 'Unpublished')],
        [zh ? '数据来源' : 'Source', selected.source], [zh ? '数据集标识' : 'Dataset ID', selected.dataset_id], [zh ? '版本' : 'Revision', selected.revision],
        [zh ? '子集 / 划分' : 'Subset / split', [selected.subset, selected.split].filter(Boolean).join(' / ')],
        [zh ? '服务器缓存路径（仅管理员可见）' : 'Server cache path (admin only)', selected.cache_path],
        [zh ? '样本数量' : 'Samples', (selected.details?.rows ?? selected.details?.sample_count) == null ? undefined : String(selected.details?.rows ?? selected.details?.sample_count)],
        [zh ? '自动校验说明' : 'Automatic validation notes', selected.details?.automatic_validation?.reason ?? selected.details?.reason],
        [zh ? '人工确认结论' : 'Manual verdict', selected.details?.manual_review ? (selected.details.manual_review.passed ? (zh ? '通过' : 'Passed') : (zh ? '不通过' : 'Failed')) : undefined],
      ].map(([title, value]) => <div key={title}><dt className="text-[var(--text-muted)] mb-1">{title}</dt><dd className="break-words whitespace-pre-wrap">{value || '—'}</dd></div>)}</dl>
    </section></div>}
  </div>
}
