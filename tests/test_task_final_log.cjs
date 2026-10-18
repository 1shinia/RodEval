const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const web = path.resolve(__dirname, '../evalscope/web')
const ts = require(path.join(web, 'node_modules/typescript'))

function harness(api) {
  const slots = [], effects = []
  let cursor = 0, pending = []
  const hooks = {
    useState(initial) {
      const i = cursor++
      if (!(i in slots)) slots[i] = initial
      return [slots[i], v => { slots[i] = typeof v === 'function' ? v(slots[i]) : v }]
    },
    useRef(initial) {
      const i = cursor++
      if (!(i in slots)) slots[i] = { current: initial }
      return slots[i]
    },
    useMemo(fn) { cursor++; return fn() },
    useCallback(fn, deps) {
      const i = cursor++
      const previous = slots[i]
      if (!previous || deps.some((v, j) => v !== previous.deps[j])) slots[i] = { fn, deps }
      return slots[i].fn
    },
    useEffect(fn, deps) {
      const i = cursor++
      const previous = effects[i]
      if (!previous || deps.some((v, j) => v !== previous.deps[j])) {
        pending.push(() => {
          previous?.cleanup?.()
          effects[i] = { deps, cleanup: fn() }
        })
      }
    },
  }
  const exports = {}
  const code = ts.transpileModule(fs.readFileSync(path.join(web, 'src/hooks/useTaskRunner.ts'), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  }).outputText
  vm.runInNewContext(code, { exports, setTimeout, clearTimeout, queueMicrotask,
    sessionStorage: { setItem() {}, removeItem() {}, getItem() {} },
    require(name) {
      if (name === 'react') return hooks
      if (name.includes('useQueryParams')) return { useQueryParams: () => new Map() }
      if (name.includes('useSSE')) return { useSSE: () => ({ connectionState: {} }) }
      if (name.includes('Toast')) return { toast: { error() {}, warning() {} } }
      if (name.includes('taskId')) return { createTaskId: () => 'eval-test' }
      if (name.includes('taskLifecycle')) return {
        classifyProgress: d => d.status === 'completed' ? { status: 'success' } : null,
        progressRetryDelay: () => 3000, shouldShowProgressError: () => false,
      }
      throw new Error(name)
    },
  })
  return () => {
    cursor = 0
    const result = exports.useTaskRunner({ api, taskPrefix: 'eval' })
    const work = pending; pending = []; work.forEach(fn => fn())
    return result
  }
}

const tick = () => new Promise(resolve => setImmediate(resolve))

test('terminal progress does not discard delayed final logs during effect cleanup', async () => {
  let resolveLog
  const api = {
    launch: async () => ({ status: 'launched' }),
    getProgress: async () => ({ status: 'completed', percent: 100 }),
    getLog: () => new Promise(resolve => { resolveLog = resolve }),
    getReportUrl: () => '',
  }
  const render = harness(api)
  await render().handleSubmit({})
  render()
  await tick()
  render() // React commits the running=false update, if one was issued too early.
  resolveLog({ text: 'Evaluation results: final score\nResults saved in: test' })
  await tick()
  const final = render()
  assert.match(final.logText, /Evaluation results: final score/)
  assert.equal(final.running, false)
  assert.equal(final.result.status, 'success')
})
