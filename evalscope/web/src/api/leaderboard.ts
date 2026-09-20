import { api } from './client'

export interface LbPeriodOption {
  value: string
  label: string
}

export interface LbBoardOption {
  id: string
  label: string
  tabs?: LbPeriodOption[]
}

export interface LbMeta {
  llm_periods: LbPeriodOption[]
  mm_periods: LbPeriodOption[]
  vlm_tabs: LbPeriodOption[]
  boards: LbBoardOption[]
  default_llm: string | null
  default_mm: string | null
}

export interface LbColumn {
  key: string
  zh: string
  en: string
  width?: number | null
  sorter?: string | null
}

export interface LbTab {
  key: string
  zh: string
  en: string
}

export interface LbTable {
  key: string
  period: string
  columns: LbColumn[]
  rows: Record<string, unknown>[]
}

export interface LlmLeaderboard {
  period: string
  tabs: LbTab[]
  tables: LbTable[]
}

export interface MmLeaderboard {
  tab?: string
  name?: string | null
  tabs?: LbTab[]
  columns: LbColumn[]
  rows: Record<string, unknown>[]
}

export async function getLeaderboardMeta(): Promise<LbMeta> {
  return api<LbMeta>('/api/v1/leaderboard/meta')
}

export async function getLlmLeaderboard(period?: string): Promise<LlmLeaderboard> {
  return api<LlmLeaderboard>('/api/v1/leaderboard/llm', period ? { period } : {})
}

export async function getMultimodalLeaderboard(tab?: string): Promise<MmLeaderboard> {
  return api<MmLeaderboard>('/api/v1/leaderboard/multimodal', tab ? { tab } : {})
}

export async function getBoard(board: string, tab?: string): Promise<MmLeaderboard> {
  return api<MmLeaderboard>('/api/v1/leaderboard/board', tab ? { board, tab } : { board })
}