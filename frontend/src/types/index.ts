export type Trend = 'new' | 'rising' | 'stable' | 'declining'
export type RequirementType = 'required' | 'preferred'
export type ReviewStatus = 'pending' | 'approved' | 'rejected'
export type GraphMode = 'panorama' | 'skill_reverse'

export interface DashboardSummary {
  sourceCount: number
  validCount: number
  emergingCount: number
  changedCount: number
  graphTrainCount?: number
  jdTestCount?: number
  holdoutCount?: number
  metrics: EvaluationMetric[]
}

export interface JdBatch {
  id: string
  filename: string
  status: 'classified' | 'reviewing' | 'applied'
  inputCount: number
  validCount: number
  rejectedCount: number
  newPositionCount: number
  changeCount: number
  noChangeCount: number
  pendingReviewCount: number
  createdAt: string
  appliedAt?: string | null
}

export interface EvaluationSummary {
  metrics: EvaluationMetric[]
  pendingReviewCount: number
  highPriorityReviewCount: number
  testedAt: string
}

export interface EvaluationMetric {
  name: string
  value: number
  target: number
  sampleCount: number
  source?: string
}

export interface GraphNode {
  id: string
  name: string
  type: 'position' | 'skill' | 'cluster' | 'stack'
  x?: number
  y?: number
  trend?: Trend
  weight?: number
  sampleCount?: number
  firstSeen?: string
  confidence?: number
}

export interface GraphEdge {
  source: string
  target: string
  relationship: 'REQUIRES' | 'BELONGS_TO'
  requirementType?: RequirementType
  weight?: number
  confidence?: number
}

export interface GraphData {
  mode: GraphMode
  hierarchy: GraphNode['type'][]
  nodes: GraphNode[]
  edges: GraphEdge[]
  summary: {
    positionClusterCount: number
    techStackCount: number
    skillClusterCount: number
    positionCount: number
    skillCount: number
  }
  updatedAt: string
  graphVersion: string
  truncated?: boolean
}

export interface GraphRoot {
  id: string
  name: string
  nodeCount: number
}

export interface GraphSearchItem {
  id: string
  type: GraphNode['type']
  name: string
}

export interface GraphSkillLink {
  skillId: string
  name: string
  requirementType: RequirementType
  weight: number
  confidence: number
}

export interface GraphNodeDetail {
  id: string
  type: GraphNode['type']
  name: string
  description: string
  sampleCount: number
  firstSeen: string
  confidence: number
  directNodes: GraphSearchItem[]
  requiredSkills: GraphSkillLink[]
  preferredSkills: GraphSkillLink[]
  relatedPositionCount: number
  skillCount: number
  clusterCount: number
  weight?: number
}

export interface SkillRequirement {
  id: string
  name: string
  type: RequirementType
  weight: number
  frequency: number
  confidence: number
  trend: Trend
  firstSeen: string
  evidenceCount: number
}

export interface PositionProfile {
  id: string
  name: string
  category: string
  techStack: string
  level: string
  status: 'emerging' | 'existing' | 'inactive'
  description: string
  firstSeen: string
  lastSeen: string
  confidence: number
  sampleCount: number
  aliases: string[]
  responsibilities: string[]
  scenarios: string[]
  requirements: SkillRequirement[]
}

export type ChangeType = 'new' | 'rising' | 'declining'

export interface RequirementSnapshot {
  requirementType: 'required' | 'preferred'
  weight: number
}

export interface EvolutionChange {
  id: string
  positionId: string
  positionName: string
  skillId: string
  skillName: string
  changeType: ChangeType
  before: RequirementSnapshot | null
  after: RequirementSnapshot
  evidenceCount: number
  confidence: number
  detectedAt: string
}

export interface ChangeEvidence {
  changeId: string
  positionId: string
  positionName: string
  skillId: string
  skillName: string
  before: RequirementSnapshot | null
  after: RequirementSnapshot
  confidence: number
  sourceSupport: {
    companyCount: number
    jobCount: number
  }
  windowContinuity: {
    continuousWindowCount: number
    passed: boolean
  }
  semanticConsistency: number
  evidenceIds: string[]
}

export interface EmergingSkill {
  id: string
  name: string
}

export interface EmergingPosition {
  id: string
  positionId: string
  name: string
  description: string
  growthRate: number
  confidence: number
  firstSeen: string
  sourceCount: number
  sampleCount: number
  skills: EmergingSkill[]
}

export interface EvidenceDetail {
  evidenceId: string
  company: string
  positionTitle: string
  sourcePlatform: string
  publishedAt: string
  url: string
  jdText: string
  excerpt: string
  matchedSkill: string
}

export interface ChangeRecord {
  id: string
  position: string
  skill: string
  changeType: '新增' | '增强' | '修改' | '下降'
  before: string
  after: string
  date: string
  evidenceCount: number
  confidence: number
}

export interface ResumeSkill {
  id?: string
  name: string
  level: '熟悉' | '掌握' | '精通'
  source: string
  confidence: number
}

export interface ResumeSkillPatch {
  skills?: ResumeSkill[]
  added?: ResumeSkill[]
  removed?: string[]
  updated?: Partial<ResumeSkill>[]
}

export interface ResumeExperience {
  period: string
  title: string
  description: string
  skills: string[]
}

export interface ParsedResumeProfile {
  candidateName: string
  targetPosition: string
  education: string
  experienceYears: number
  direction: string
  completeness: number
  skills: ResumeSkill[]
  experiences: ResumeExperience[]
  analysisSource?: string
  llmAnalysis?: {
    enabled?: boolean
    status?: 'completed' | 'degraded' | 'not_enabled' | string
    model?: string
    inputMode?: 'text' | 'vision' | string
    fallbackSource?: string
    error?: string
  }
}

export interface ResumeTask {
  taskId: string
  id?: string
  filename?: string
  fileSize?: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  createdAt?: string
  updatedAt?: string
  error?: string
  result?: ParsedResumeProfile
}

export interface MatchDimension {
  name: string
  value: number
  color: string
}

export interface LearningStep {
  stage: number
  title: string
  duration: string
  skills: string[]
  goal: string
}

export interface SkillGap {
  name: string
  priority: string
  requirement: string
  current: string
  weight: number
}

export interface MatchReport {
  matchId: string
  resumeTaskId: string
  positionId: string
  positionName: string
  candidateName: string
  overallScore: number
  fitLevel: string
  benchmarkRank: string
  benchmarkSampleCount: number
  summary: string
  dimensions: MatchDimension[]
  strengths: string[]
  gaps: SkillGap[]
  evidence: {
    skillEvidenceCount: number
    projectEvidenceCount: number
    jobSampleCount: number
  }
  suggestions: string[]
  learningPath: LearningStep[]
  guidanceSource?: 'llm' | 'deterministic_fallback'
  skillAlignment?: SkillAlignment[]
}

export interface SkillAlignment {
  rawName: string
  standardSkillId: string
  standardSkillName: string
  level: string
  confidence: number
  reason?: string
  source: 'llm' | 'deterministic_fallback'
}

export interface MatchRankingItem {
  positionId: string
  positionName: string
  score: number
  fitLevel: string
  matchedSkillCount: number
  totalSkillCount: number
  strengths: string[]
  gapCount: number
}

export interface MatchRanking {
  resumeTaskId: string
  bestPositionId: string
  bestPositionName: string
  bestScore: number
  skillAlignment?: SkillAlignment[]
  items: MatchRankingItem[]
}

export interface ReviewItem {
  id: string
  type: '新岗位' | '能力变更' | '技能归一'
  title: string
  description: string
  confidence: number
  sources: string[]
  createdAt: string
  status: ReviewStatus
  targetId?: string
  note?: string
}

export interface ApiResponse<T> {
  data: T
  requestId: string
}
