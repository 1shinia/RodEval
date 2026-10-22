// Run: node --test tests/rag/test_reranker_form.cjs
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const root = path.resolve(__dirname, '../..')
const web = path.join(root, 'evalscope/web')
const ts = require(path.join(web, 'node_modules/typescript'))
const React = require(path.join(web, 'node_modules/react'))
const { renderToStaticMarkup } = require(path.join(web, 'node_modules/react-dom/server'))

function loadForm(tool, onSubmit) {
  let stateIndex = 0
  const preset = [tool, 'test-model', 'https://example.invalid/v1', 'test-credential']
  const exports = {}
  const source = fs.readFileSync(path.join(web, 'src/components/eval/RAGEvalForm.tsx'), 'utf8')
  const code = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: false,
  }}).outputText
  const hooks = { ...React,
    useState: initial => [stateIndex < preset.length ? preset[stateIndex++] : (stateIndex++, initial), () => {}],
    useEffect: () => {}, useRef: () => ({ current: null }),
  }
  vm.runInNewContext(code, { exports, require: name => {
    if (name === 'react') return hooks
    if (name === 'react/jsx-runtime') return require(path.join(web, 'node_modules/react/jsx-runtime'))
    if (name.includes('LocaleContext')) return { useLocale: () => ({ t: key => key }) }
    if (name.includes('AuthContext')) return { useAuth: () => ({ user: { role: 'admin' } }) }
    if (name.includes('formStyles')) return { inputClass: () => '', FORM_INPUT_CLASS: '', FORM_LABEL_CLASS: '' }
    if (name.includes('PublishedDatasetPicker')) return { default: () => null }
    if (name.includes('FormField')) return { default: props => React.createElement('div', null, props.label, props.children) }
    if (name.includes('/Button')) return { default: props => React.createElement('button', { type: props.type }, props.children) }
    throw new Error(`Unexpected import: ${name}`)
  }})
  return exports.default({ onSubmit })
}

test('reranker form exposes no two-stage or unused Top K controls', () => {
  const html = renderToStaticMarkup(loadForm('reranker', () => {}))
  for (const key of ['eval.ragTwoStage', 'eval.ragTopK', 'evalRagExtra.encoderStageOne']) {
    assert.ok(!html.includes(key), `Unexpected control: ${key}`)
  }
  assert.ok(html.includes('Reranking'))
  assert.ok(!html.includes('InstructionReranking'))
  assert.ok(html.includes('eval.rerankerTaskTypesTitle'))
  assert.ok(html.includes('eval.rerankerDatasetNames'))
})

test('customer RAG selector exposes only embedding and reranker', () => {
  const form = loadForm('embedding', () => {})
  const html = renderToStaticMarkup(form)
  const selector = html.match(/<select\b[^>]*>[\s\S]*?<\/select>/)[0]
  const options = [...selector.matchAll(/<option value="([^"]+)"/g)].map(match => match[1])
  assert.deepEqual(options, ['embedding', 'reranker'])
})

for (const tool of ['reranker', 'embedding']) {
  test(`${tool} submits one API model without a Top K override`, () => {
    let submitted
    loadForm(tool, config => { submitted = config }).props.onSubmit({ preventDefault() {} })
    assert.ok(submitted)
    assert.equal(submitted.eval_config.models.length, 1)
    const model = submitted.eval_config.models[0]
    assert.equal(model.is_cross_encoder, tool === 'reranker')
    assert.equal(model.model_name, 'test-model')
    assert.equal(model.api_base, 'https://example.invalid/v1')
    assert.ok(!('top_k' in submitted.eval_config.eval))
  })
}
