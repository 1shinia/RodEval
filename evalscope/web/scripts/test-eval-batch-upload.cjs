const assert = require('node:assert/strict')
const fs = require('node:fs')
const vm = require('node:vm')
const ts = require('typescript')
const { test } = require('node:test')
const path = require('node:path')
const source = fs.readFileSync(path.join(__dirname, '../src/pages/EvalLayout.tsx'), 'utf8')
const tree = ts.createSourceFile('EvalLayout.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
function callback(name, context) {
  let found
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === name) found = node.initializer.arguments[0]
    ts.forEachChild(node, visit)
  }
  visit(tree)
  assert.ok(found, `callback ${name} exists`)
  const js = ts.transpileModule(`(${found.getText(tree)})`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
  return vm.runInNewContext(js, context)
}
function setup() {
  const state = { info: { batch_id: 'old' }, batch: { status: 'stopped', resumable: true }, error: 'old error', log: 'old log', selected: 'old-task', selectedLog: 'old result', uploading: false, running: false, generation: 0 }
  const storage = new Map([['evalBatchId', 'old']])
  const context = {
    batchIdRef: { current: 'old' }, batchFileRef: { current: null }, batchConfigRef: { current: { old: true } },
    logGenerationRef: { current: 4 }, lastLogTaskIdRef: { current: 'old-task' },
    clearBatchPoll: () => { state.generation++ },
    sessionStorage: { removeItem: key => storage.delete(key) },
    setBatchInfo: v => { state.info = v }, setBatchState: v => { state.batch = v },
    setBatchError: v => { state.error = v }, setBatchLogText: v => { state.log = v },
    setSelectedTaskId: v => { state.selected = v }, setSelectedTaskLog: v => { state.selectedLog = v },
    setBatchUploading: v => { state.uploading = v }, setBatchRunning: v => { state.running = v },
    uploadEvalBatchCsv: async () => ({ batch_id: 'new', model_count: 1, models: ['new-model'] }),
    toast: { info() {}, error() {} }, tRef: { current: x => x },
  }
  return { state, storage, context }
}
test('explicit CSV upload detaches old batch before awaiting upload and keeps new upload ready', async () => {
  const { state, storage, context } = setup()
  let finish
  context.uploadEvalBatchCsv = () => new Promise(resolve => { finish = resolve })
  const promise = callback('handleBatchUpload', context)({ name: 'changed.csv' })
  assert.equal(context.batchIdRef.current, null)
  assert.equal(storage.has('evalBatchId'), false)
  assert.equal(state.batch, null)
  assert.equal(state.info, null)
  assert.equal(state.log, '')
  assert.equal(state.selected, '')
  assert.equal(state.selectedLog, '')
  assert.equal(context.lastLogTaskIdRef.current, '')
  assert.equal(context.logGenerationRef.current, 5)
  assert.equal(state.generation, 1)
  finish({ batch_id: 'new', model_count: 1, models: ['new-model'] })
  await promise
  assert.equal(state.info.batch_id, 'new')
  assert.equal(state.uploading, false)
})
test('replacement upload invalidates a late status response from the old batch', async () => {
  const { state, context } = setup()
  context.pollGenerationRef = { current: 0 }
  context.clearBatchPoll = () => { context.pollGenerationRef.current++ }
  let finishStatus
  context.getEvalBatchStatus = () => new Promise(resolve => { finishStatus = resolve })
  callback('monitorBatch', context)('old')
  await callback('handleBatchUpload', context)({ name: 'replacement.csv' })
  finishStatus({ status: 'stopped', resumable: true, results: [] })
  await new Promise(resolve => setImmediate(resolve))
  assert.equal(state.batch, null)
  assert.equal(state.info.batch_id, 'new')
})
test('replacement upload invalidates a late log response from the old batch', async () => {
  const { state, context } = setup()
  let finishLog
  context.getEvalLog = () => new Promise(resolve => { finishLog = resolve })
  const oldLog = callback('fetchTaskLog', context)('old-task')
  await callback('handleBatchUpload', context)({ name: 'replacement.csv' })
  finishLog({ text: 'late old result' })
  await oldLog
  assert.equal(state.selectedLog, '')
  assert.equal(state.selected, '')
})
test('failed replacement upload cannot fall back to old batch', async () => {
  const { state, storage, context } = setup()
  context.uploadEvalBatchCsv = async () => { throw new Error('invalid CSV') }
  await callback('handleBatchUpload', context)({ name: 'invalid.csv' })
  assert.equal(context.batchIdRef.current, null)
  assert.equal(storage.has('evalBatchId'), false)
  assert.equal(state.batch, null)
  assert.equal(state.info, null)
  assert.match(state.error, /invalid CSV/)
})
test('replacement CSV launches the new batch with the current configuration', async () => {
  const { state, storage, context } = setup()
  context.sessionStorage.setItem = (key, value) => storage.set(key, value)
  let launched, monitored
  context.launchEvalBatch = async (...args) => { launched = args; return { total: 1 } }
  context.monitorBatch = id => { monitored = id }
  await callback('handleBatchUpload', context)({ name: 'replacement.csv' })
  const config = { datasets: ['aime24'], limit: 5 }
  await callback('onBatchSubmit', context)(state.info.batch_id, config)
  assert.equal(launched[0], 'new')
  assert.equal(launched[1], config)
  assert.equal(storage.get('evalBatchId'), 'new')
  assert.equal(monitored, 'new')
  assert.equal(state.running, true)
})
test('explicit resume still passes old batch ID with newly uploaded CSV ID', async () => {
  const { state, context } = setup()
  let args, monitored
  context.resumeEvalBatch = async (...values) => { args = values }
  context.monitorBatch = id => { monitored = id }
  const config = { datasets: ['aime24'] }
  await callback('onBatchResume', context)({ name: 'original.csv' }, config)
  assert.equal(args[0], 'old')
  assert.equal(args[1], 'new')
  assert.equal(args[2], config)
  assert.equal(monitored, 'old')
  assert.equal(state.running, true)
})
