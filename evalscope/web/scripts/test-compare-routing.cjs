const { readFileSync } = require('node:fs')
const { resolve } = require('node:path')
const vm = require('node:vm')
const assert = require('node:assert/strict')
const ts = require('typescript')
const base = resolve(__dirname, '..')
const filename = resolve(base, 'src/pages/ComparePage.tsx')
const source = readFileSync(filename, 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS } }).outputText
let pathname = '/compare', reportId, draftId = 'draft-1', selection = { backend: 'Perf', rootPath: '', reports: ['old-a', 'old-b'], draftId: 'draft-1' }
let states = [], cursor = 0, effects = [], requested = [], fail = false
const navigations = []
const jsx = (type, props, key) => ({ type, props: props || {}, key })
const react = {
  useState(initial) { const i = cursor++; if (!(i in states)) states[i] = initial; return [states[i], value => { states[i] = value }] },
  useEffect(fn) { effects.push(fn) }, useRef(value) { return { current: value } },
  useCallback(fn) { return fn }, useMemo(fn) { return fn() }, createContext() { return {} },
}
const exportsObject = {}
const context = {
  exports: exportsObject,
  require(id) {
    if (id === 'react') return react
    if (id === 'react/jsx-runtime') return { jsx, jsxs: jsx }
    if (id === 'react-router-dom') return {
      useParams: () => ({ reportId }), useLocation: () => ({ pathname }),
      useNavigate: () => value => navigations.push(value), useSearchParams: () => [new URLSearchParams(draftId ? { id: draftId } : {})],
    }
    if (id.endsWith('/CompareContext')) return { useCompare: () => ({ selection }) }
    if (id.endsWith('/LocaleContext')) return { useLocale: () => ({ t: key => key, locale: 'en' }) }
    if (id === '@/api/perf') return { getSavedCompareReport: async id => {
      requested.push(id)
      if (fail) throw new Error('Report not found')
      return { report: { task_ids: '["a","b"]', backend: 'LLM', root_path: '/saved-root' } }
    } }
    return new Proxy({}, { get: (_, key) => key === '__esModule' ? false : (() => null) })
  },
}
vm.runInNewContext(compiled, context, { filename })
const Page = exportsObject.default
async function run() {
  assert.equal(Page().type.name, 'SavedReportsList', 'list must ignore stale cached selection')
  pathname = '/compare/draft'
  assert.equal(Page().type.name, 'ComparisonView')
  draftId = 'different-draft'
  assert.equal(Page().type.name, 'ComparisonMessage', 'stale selection cannot open a different draft')
  draftId = null
  assert.equal(Page().type.name, 'ComparisonMessage', 'draft without ID cannot use stale selection')
  draftId = 'draft-1'
  selection = null
  assert.equal(Page().type.name, 'ComparisonMessage', 'empty draft must not show stale details')
  reportId = '42'; pathname = '/compare/saved/42'
  const saved = Page()
  assert.equal(saved.type.name, 'SavedComparison')
  assert.equal(saved.key, '42')
  cursor = 0; saved.type(saved.props)
  effects.pop()()
  await new Promise(resolve => setImmediate(resolve))
  cursor = 0
  const detail = saved.type(saved.props)
  assert.equal(detail.type.name, 'ComparisonView')
  assert.equal(requested[0], 42)
  assert.equal(detail.props.selection.rootPath, '/saved-root')
  assert.equal(detail.props.selection.backend, 'Native')
  assert.equal(detail.props.selection.reports.join(','), 'a,b')
  reportId = '43'
  assert.equal(Page().key, '43', 'different IDs must reset detail state')
  reportId = undefined; pathname = '/compare'
  assert.equal(Page().type.name, 'SavedReportsList', 'return navigation must show list')
  states = []; cursor = 0; effects = []; fail = true
  saved.type(saved.props); effects.pop()()
  await new Promise(resolve => setImmediate(resolve)); cursor = 0
  assert.equal(saved.type(saved.props).type.name, 'ComparisonMessage', '404 must not display cached report')
  const app = readFileSync(resolve(base, 'src/App.tsx'), 'utf8')
  for (const route of ['/compare', '/compare/draft', '/compare/saved/:reportId']) assert.ok(app.includes(`path="${route}"`))
  const barSource = readFileSync(resolve(base, 'src/components/reports/CompareBar.tsx'), 'utf8')
  const barModule = ts.transpileModule(barSource, { compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS } }).outputText
  let storedSelection, openedUrl
  const barExports = {}
  vm.runInNewContext(barModule, {
    exports: barExports,
    window: { location: { origin: 'http://localhost:5173' }, open: url => { openedUrl = url } },
    URL,
    require(id) {
      if (id === 'react/jsx-runtime') return { jsx, jsxs: jsx }
      if (id.endsWith('/CompareContext')) return { useCompare: () => ({ setCompareSelection: s => { storedSelection = s } }) }
      if (id.endsWith('/LocaleContext')) return { useLocale: () => ({ t: key => key }) }
      return {}
    },
  })
  const bar = barExports.default({ selected: ['first', 'second'], totalCount: 2, rootPath: '/example', backend: 'Native', compareUrl: '/compare?root_path=%2Fexample' })
  // The last JSX child is the compare Button component; invoke its click handler.
  const compareButton = bar.props.children.at(-1)
  compareButton.props.onClick()
  const opened = new URL(openedUrl, 'http://localhost:5173')
  assert.equal(opened.pathname, '/compare/draft')
  assert.equal(opened.searchParams.get('id'), storedSelection.draftId)
  assert.equal(opened.searchParams.get('root_path'), '/example')
  assert.deepEqual(Array.from(storedSelection.reports), ['first', 'second'])
  console.log('PASS: list ignores cache; draft identity; empty draft; saved ID loading; ID reset; return to list; missing report; route registration; compare entry')
}
run().catch(error => { console.error(error); process.exitCode = 1 })
