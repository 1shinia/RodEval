const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const web = path.resolve(__dirname, '../../evalscope/web')
const ts = require(path.join(web, 'node_modules/typescript'))
const React = require(path.join(web, 'node_modules/react'))
const { renderToStaticMarkup } = require(path.join(web, 'node_modules/react-dom/server'))

function loadForm(metrics, onSubmit) {
  let source = fs.readFileSync(path.join(web, 'src/components/eval/RAGEvalForm.tsx'), 'utf8')
  const states = { ragTool: 'ragas', ragasTestset: '/data/input.json', ragasLlmModel: 'offline',
    ragasMetrics: metrics, ragasEmbModel: '' }
  for (const [name, value] of Object.entries(states)) {
    source = source.replace(new RegExp(`(const \\[${name},[^\\n]*?= useState(?:<[^>]+>)?\\()([^\\n]*)(\\))`),
      (_, prefix, initial, end) => prefix + JSON.stringify(value) + end)
  }
  const code = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: false,
  }}).outputText
  const exports = {}
  vm.runInNewContext(code, { exports, require: name => {
    if (name === 'react') return { ...React, useState: value => [value, () => {}], useEffect() {}, useRef: () => ({ current: null }) }
    if (name === 'react/jsx-runtime') return require(path.join(web, 'node_modules/react/jsx-runtime'))
    if (name.includes('LocaleContext')) return { useLocale: () => ({ t: key => key }) }
    if (name.includes('AuthContext')) return { useAuth: () => ({ user: { role: 'admin' } }) }
    if (name.includes('formStyles')) return { inputClass: () => '', FORM_INPUT_CLASS: '', FORM_LABEL_CLASS: '' }
    if (name.includes('FormField')) return { default: p => React.createElement('div', null, p.label, p.children) }
    if (name.includes('/Button')) return { default: p => React.createElement('button', { type: p.type }, p.children) }
    throw new Error(name)
  }})
  return exports.default({ onSubmit })
}

test('RAGAS offers only implemented metrics and no ineffective language control', () => {
  const html = renderToStaticMarkup(loadForm(['faithfulness'], () => {}))
  assert.ok(!html.includes('context_relevancy'))
  assert.ok(!html.includes('eval.ragLanguage'))
  assert.ok(!html.includes('eval.ragEmbeddingModel'))
})

test('faithfulness submits without an embedding model or unused language', () => {
  let submitted
  loadForm(['faithfulness'], config => { submitted = config }).props.onSubmit({ preventDefault() {} })
  assert.ok(submitted)
  assert.equal(submitted.eval_config.eval.testset_file, '/data/input.json')
  assert.ok(!submitted.eval_config.eval.embeddings)
  assert.ok(!('language' in submitted.eval_config.eval))
})

test('answer relevancy still requires an embedding model', () => {
  let submitted
  const form = loadForm(['answer_relevancy'], config => { submitted = config })
  assert.ok(renderToStaticMarkup(form).includes('eval.ragEmbeddingModel'))
  form.props.onSubmit({ preventDefault() {} })
  assert.equal(submitted, undefined)
})
