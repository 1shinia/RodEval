export const loadReportsLayout = () => import('@/pages/ReportsLayout')
export const loadLLMReportsTab = () => import('@/pages/LLMReportsTab')

export function preloadDefaultReportsRoute(): void {
  void loadReportsLayout()
  void loadLLMReportsTab()
}
