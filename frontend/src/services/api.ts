import type { ApiResponse, ChangeEvidence, DashboardSummary, EvaluationSummary, EvidenceDetail, EvolutionChange, EmergingPosition, GraphData, GraphMode, GraphNodeDetail, GraphRoot, GraphSearchItem, JdBatch, MatchRanking, MatchReport, PositionProfile, ResumeSkill, ResumeSkillPatch, ResumeTask, ReviewItem, ReviewStatus } from '../types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

type Method = 'GET' | 'POST' | 'PATCH'
type QueryValue = string | number | boolean | null | undefined

interface RequestOptions {
  method?: Method
  params?: Record<string, QueryValue>
  body?: unknown
  timeoutMs?: number
}

const withQuery = (path: string, params?: RequestOptions['params']) => {
  const query = new URLSearchParams()
  Object.entries(params ?? {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') query.set(key, String(value))
  })
  const suffix = query.toString()
  return suffix ? `${path}?${suffix}` : path
}

const request = async <T>(path: string, options: RequestOptions = {}): Promise<ApiResponse<T>> => {
  const headers = new Headers()
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), options.timeoutMs ?? 30_000)
  const init: RequestInit = { method: options.method ?? 'GET', signal: controller.signal }

  if (options.body instanceof FormData) {
    init.body = options.body
  } else if (options.body !== undefined) {
    headers.set('Content-Type', 'application/json')
    init.body = JSON.stringify(options.body)
  }

  init.headers = headers
  try {
    const response = await fetch(`${API_BASE}${withQuery(path, options.params)}`, init)
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`
      try {
        const payload = await response.json() as { detail?: string }
        if (payload.detail) detail = payload.detail
      } catch { /* keep HTTP status as fallback */ }
      throw new Error(detail)
    }
    return response.json() as Promise<ApiResponse<T>>
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw new Error('请求超时，请检查后端服务或简历文件')
    throw error
  } finally {
    window.clearTimeout(timeout)
  }
}

export const api = {
  getJdBatches: (): Promise<ApiResponse<JdBatch[]>> => request<JdBatch[]>('/api/v1/jd-batches'),
  createJdBatch: (file: File): Promise<ApiResponse<JdBatch>> => {
    const body = new FormData()
    body.set('file', file)
    return request<JdBatch>('/api/v1/jd-batches', { method: 'POST', body, timeoutMs: 300_000 })
  },
  getDashboard: (): Promise<ApiResponse<DashboardSummary>> => request<DashboardSummary>('/api/v1/dashboard'),
  getDashboardSummary: (): Promise<ApiResponse<DashboardSummary>> => request<DashboardSummary>('/api/v1/dashboard/summary'),
  getEvaluationSummary: (): Promise<ApiResponse<EvaluationSummary>> => request<EvaluationSummary>('/api/v1/evaluations/summary'),
  getGraph: (mode: GraphMode = 'panorama', params: { keyword?: string; maxNodes?: number } = {}): Promise<ApiResponse<GraphData>> => request<GraphData>('/api/v1/graph', { params: { mode, keyword: params.keyword, max_nodes: params.maxNodes } }),
  getGraphRoots: (mode: GraphMode = 'panorama'): Promise<ApiResponse<GraphRoot[]>> => request<GraphRoot[]>('/api/v1/graph/roots', { params: { mode } }),
  getGraphNodeDetail: (nodeId: string): Promise<ApiResponse<GraphNodeDetail>> => request<GraphNodeDetail>(`/api/v1/graph/nodes/${nodeId}`),
  searchGraph: (mode: GraphMode, keyword: string, limit = 10): Promise<ApiResponse<GraphSearchItem[]>> => request<GraphSearchItem[]>('/api/v1/graph/search', { params: { mode, keyword, limit } }),
  getPosition: (id: string): Promise<ApiResponse<PositionProfile>> => request<PositionProfile>(`/api/v1/positions/${id}`),
  getEvolutionChanges: (): Promise<ApiResponse<{ items: EvolutionChange[]; total: number; page: number; pageSize: number }>> => request<{ items: EvolutionChange[]; total: number; page: number; pageSize: number }>('/api/v1/evolution/changes', { params: { page: 1, page_size: 20 } }),
  getChangeEvidence: (changeId: string): Promise<ApiResponse<ChangeEvidence>> => request<ChangeEvidence>(`/api/v1/evolution/changes/${changeId}/evidence`),
  getEvidenceDetail: (evidenceId: string): Promise<ApiResponse<EvidenceDetail>> => request<EvidenceDetail>(`/api/v1/evolution/evidence/${evidenceId}`),
  getEmergingPositions: (): Promise<ApiResponse<{ items: EmergingPosition[]; total: number; page: number; pageSize: number }>> => request<{ items: EmergingPosition[]; total: number; page: number; pageSize: number }>('/api/v1/emerging-positions', { params: { page: 1, page_size: 20 } }),
  getReviews: (params: { status?: ReviewStatus; type?: ReviewItem['type']; keyword?: string } = {}): Promise<ApiResponse<ReviewItem[]>> => request<ReviewItem[]>('/api/v1/reviews', { params }),
  decideReview: (id: string, status: ReviewStatus, note = ''): Promise<ApiResponse<{ id: string; status: ReviewStatus; note: string }>> => request<{ id: string; status: ReviewStatus; note: string }>(`/api/v1/reviews/${id}/decision`, { method: 'POST', body: { status, note } }),
  review: (id: string, status: ReviewStatus, note = ''): Promise<ApiResponse<{ id: string; status: ReviewStatus; note: string }>> => api.decideReview(id, status, note),
  createResumeTask: (file: File): Promise<ApiResponse<{ taskId: string; status: string; progress: number }>> => {
    const body = new FormData()
    body.set('file', file)
    return request<{ taskId: string; status: string; progress: number }>('/api/v1/resume-tasks', { method: 'POST', body })
  },
  getResumeTask: (taskId: string): Promise<ApiResponse<ResumeTask>> => request<ResumeTask>(`/api/v1/resume-tasks/${taskId}`),
  updateResumeSkills: (taskId: string, skills: ResumeSkill[] | ResumeSkillPatch): Promise<ApiResponse<{ taskId: string; skills: ResumeSkill[] }>> => {
    const body = Array.isArray(skills) ? { skills } : skills
    return request<{ taskId: string; skills: ResumeSkill[] }>(`/api/v1/resume-tasks/${taskId}/skills`, { method: 'PATCH', body })
  },
  createMatch: (resumeTaskId: string, positionId: string): Promise<ApiResponse<MatchReport>> => request<MatchReport>('/api/v1/matches', { method: 'POST', body: { resumeTaskId, positionId } }),
  rankMatches: (resumeTaskId: string, limit = 50): Promise<ApiResponse<MatchRanking>> => request<MatchRanking>('/api/v1/matches/rank', { method: 'POST', body: { resumeTaskId, limit } }),
  getLearningPath: (matchId: string): Promise<ApiResponse<{ matchId: string; items: MatchReport['learningPath'] }>> => request<{ matchId: string; items: MatchReport['learningPath'] }>(`/api/v1/matches/${matchId}/learning-path`),
}
