import React, { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Components } from 'react-markdown'
import { useTheme } from '@/contexts/ThemeContext'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter'
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash'
import cpp from 'react-syntax-highlighter/dist/esm/languages/prism/cpp'
import java from 'react-syntax-highlighter/dist/esm/languages/prism/java'
import javascript from 'react-syntax-highlighter/dist/esm/languages/prism/javascript'
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json'
import markdown from 'react-syntax-highlighter/dist/esm/languages/prism/markdown'
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python'
import sql from 'react-syntax-highlighter/dist/esm/languages/prism/sql'
import typescript from 'react-syntax-highlighter/dist/esm/languages/prism/typescript'
import yaml from 'react-syntax-highlighter/dist/esm/languages/prism/yaml'
import { vscDarkPlus, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import ImageLightbox from './ImageLightbox'

interface Props {
  content: string
}

const INLINE_IMG_STYLE = { maxHeight: 200, maxWidth: 320, display: 'inline-block', verticalAlign: 'top' as const }

SyntaxHighlighter.registerLanguage('bash', bash)
SyntaxHighlighter.registerLanguage('cpp', cpp)
SyntaxHighlighter.registerLanguage('java', java)
SyntaxHighlighter.registerLanguage('javascript', javascript)
SyntaxHighlighter.registerLanguage('json', json)
SyntaxHighlighter.registerLanguage('markdown', markdown)
SyntaxHighlighter.registerLanguage('python', python)
SyntaxHighlighter.registerLanguage('sql', sql)
SyntaxHighlighter.registerLanguage('typescript', typescript)
SyntaxHighlighter.registerLanguage('yaml', yaml)

const LANGUAGE_ALIASES: Record<string, string> = {
  c: 'cpp',
  'c++': 'cpp',
  js: 'javascript',
  jsx: 'javascript',
  md: 'markdown',
  py: 'python',
  sh: 'bash',
  shell: 'bash',
  ts: 'typescript',
  tsx: 'typescript',
  yml: 'yaml',
}

const SUPPORTED_LANGUAGES = new Set([
  'bash', 'cpp', 'java', 'javascript', 'json', 'markdown', 'python', 'sql', 'typescript', 'yaml',
])

function MarkdownRenderer({ content }: Props) {
  const { theme } = useTheme()

  const markdownComponents = useMemo<Components>(() => ({
    img: ({ src, alt }) => <ImageLightbox src={src ?? ''} alt={alt} style={INLINE_IMG_STYLE} />,
    code: ({ className, children }) => {
      const match = /language-(\w+)/.exec(className || '')
      const language = match ? (LANGUAGE_ALIASES[match[1].toLowerCase()] ?? match[1].toLowerCase()) : ''
      if (language && SUPPORTED_LANGUAGES.has(language)) {
          return (
            <SyntaxHighlighter
            language={language}
            style={theme === 'dark' ? vscDarkPlus : oneLight}
            PreTag="div"
            customStyle={{
              margin: 0,
              borderRadius: '0.5rem',
              padding: '1rem',
              fontSize: '0.8125rem',
              lineHeight: 1.6,
            }}
            codeTagProps={{ style: { fontFamily: 'var(--font-mono)' } }}
          >
            {String(children).replace(/\n$/, '')}
          </SyntaxHighlighter>
        )
      }
      return match ? (
        <pre className="bg-[var(--bg-card)] p-4 rounded-lg overflow-x-auto text-[0.8125rem] leading-relaxed">
          <code className="font-mono">{children}</code>
        </pre>
      ) : (
        <code className="bg-[var(--bg-card)] px-1.5 py-0.5 rounded text-[0.85em] font-mono">{children}</code>
      )
    },
    pre: ({ children }) => <div className="not-prose my-3">{children}</div>,
    table: ({ children }) => (
      <div className="overflow-x-auto my-4">
        <table className="w-full">{children}</table>
      </div>
    ),
  }), [theme])

  if (!content) return null

  return (
    <div className={`prose prose-sm max-w-none break-words${theme === 'dark' ? ' prose-invert' : ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        urlTransform={(url) => {
          // Only allow safe URL protocols to prevent XSS via javascript:/data: URIs
          const allowed = ['http:', 'https:', 'mailto:', '#', '/']
          try {
            const parsed = new URL(url, window.location.origin)
            if (allowed.some(p => p.endsWith(':') ? parsed.protocol === p : url.startsWith(p))) {
              return url
            }
          } catch {
            // Relative URLs or fragments are safe
            if (url.startsWith('/') || url.startsWith('#') || url.startsWith('.')) return url
          }
          return ''
        }}
        components={markdownComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

export default React.memo(MarkdownRenderer)
