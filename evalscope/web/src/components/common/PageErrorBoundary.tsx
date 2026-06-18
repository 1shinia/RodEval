import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangle, RotateCw } from 'lucide-react'

interface Props {
  children: ReactNode
  pageName?: string
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class PageErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[PageErrorBoundary${this.props.pageName ? `:${this.props.pageName}` : ''}]`, error, info.componentStack)
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center min-h-[40vh] p-8">
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-[var(--radius)] p-8 max-w-[480px] w-full text-center shadow-[var(--shadow)]">
            <div className="w-12 h-12 rounded-xl bg-[var(--warning-bg)] border border-[var(--warning-border)] inline-flex items-center justify-center mb-4">
              <AlertTriangle size={22} className="text-[var(--warning)]" />
            </div>
            <h2 className="text-[var(--text)] text-base font-semibold mb-2 mt-0">
              Page Error
            </h2>
            <p className="text-[var(--text-muted)] text-sm mb-5 mt-0 leading-normal break-words">
              {this.state.error?.message || 'An unexpected error occurred in this page.'}
            </p>
            <button
              onClick={this.handleRetry}
              className="inline-flex items-center gap-1.5 bg-[var(--accent)] text-[var(--bg)] border-0 rounded-[var(--radius-sm)] px-5 py-2 text-sm font-medium cursor-pointer transition-opacity duration-150 hover:opacity-85"
            >
              <RotateCw size={14} />
              Retry
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
