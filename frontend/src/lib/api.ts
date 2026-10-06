/**
 * SafeMed AI API Client.
 * Connects frontend directly to the SafeMed AI backend.
 * Uses Vite proxy (/api) or direct backend URL (http://127.0.0.1:8000).
 * No hardcoded responses or static fallbacks.
 */

// ---------------------------------------------------------------------------
// Evaluation results served by GET /api/evaluation-results (backend/results/evaluation_results.json)
// produced by experiments/evaluation/run_evaluation.py
// ---------------------------------------------------------------------------

export interface WilcoxonTestResult {
  test_type: string
  statistic: number
  p_value: number
  p_value_holm?: number
  significant: boolean
  mean_diff?: number
  median_diff?: number
  hodges_lehmann?: number
  rank_biserial?: number
  ci_95?: [number, number]
  n_pairs?: number
  n_samples?: number
  diff_proportions?: number
  discordant_pairs?: number
  contingency_table?: [[number, number], [number, number]]
  cohens_dz?: number
  shapiro_wilk_p?: number
}

/** Result shape for McNemar's test or Exact Binomial fallback (legacy support). */
export type BinaryTestResult = WilcoxonTestResult

/** Result shape for Paired t-test or Permutation fallback (legacy support). */
export type ContinuousTestResult = WilcoxonTestResult

/** Fact-level verification metrics for one condition (control or treatment). */
export interface FactLevelConditionMetrics {
  precision: number   // Answer Accuracy = 1 − hr
  recall: number
  f1: number
  hr: number          // Hallucination Rate
  hdihr: number       // Harmful-Document-Induced Hallucination Rate
}

/** Filter performance metrics from experiments/evaluation/metrics.py */
export interface FilterMetrics {
  TPR_Recall: number
  FPR: number
  Precision: number
  F1_score: number
  AUROC: number
  PR_AUC: number
}

/** Per-query document entry (control top-5 format). */
export interface QueryDocumentControl {
  id: string
  pmid?: string
  title: string
  source: string
  text?: string
  true_label?: string
  topic?: string
  retrieval_score?: number
  block_reason?: string
}

/** Per-query document entry (treatment top-5 format). */
export interface QueryDocumentTreatment {
  id: string
  title: string
  source: string
  verification_status: string
  relevance_score: number
  snippet: string
}

/** Per-query condition result. */
export interface QueryConditionResult {
  her: number
  gtrr: number
  answer: string
  top5: (QueryDocumentControl | QueryDocumentTreatment)[]
  fact_metrics?: {
    precision: number
    recall: number
    f1: number
    hallucination_rate: number
    hdihr: number
  }
}

/** Shape of the legacy pilot results file (the Stage 5 benchmark now writes a different schema) */
export interface EvaluationResults {
  num_queries: number
  Mean_Pre_Filter_Harmful_Density?: number
  HD_Exposure_Top5_Control: number
  HD_Exposure_Top5_Treatment: number
  HER_Reduction?: number
  GD_Retention_Top5_Control: number
  GD_Retention_Top5_Treatment: number
  Inferential_Statistics: {
    H1_HER_Reduction?: WilcoxonTestResult
    H2_GTRR_Retention?: WilcoxonTestResult
    H3_Pre_vs_Final_HER?: WilcoxonTestResult
    H4_Hallucination_Rate?: WilcoxonTestResult
    Hit_at_5?: WilcoxonTestResult
    Hallucination_Rate?: WilcoxonTestResult
    HDIHR?: WilcoxonTestResult
    HD_at_Top5?: WilcoxonTestResult
    [key: string]: WilcoxonTestResult | undefined
  }
  Filter_Metrics: FilterMetrics
  Fact_Level_Metrics: {
    control: FactLevelConditionMetrics
    treatment: FactLevelConditionMetrics
  }
  query_details: Array<{
    query_id: string
    query: string
    reference_answer: string
    control: QueryConditionResult
    treatment: QueryConditionResult
  }>
}

export interface DocumentEvaluationEntry {
  id?: string
  title: string
  source: string
  category: 'GD' | 'HD' | 'MD'
  status: 'TRUSTED SOURCE' | 'BLOCKED: UNSAFE'
  is_blocked: boolean
  block_reason?: string | null
  harmful_probability: number
  snippet: string
  text?: string | null
  pmid?: string | null
  reinstated?: boolean
  scorer?: string | null
  retrieval_score?: number | null
}

export interface SafetyScanSummary {
  total_scanned: number
  blocked_count: number
  safe_count: number
  evaluation_log: DocumentEvaluationEntry[]
}

export interface TrustedDocument {
  id?: string
  title: string
  source: string
  verification_status: string
  relevance_score: number
  snippet: string
  text?: string | null
  pmid?: string | null
  harmful_probability?: number | null
  retrieval_score?: number | null
}

export interface SingleQueryDocument {
  id?: string
  title: string
  source: string
  category: 'GD' | 'HD' | 'MD' | string
  status?: string
  is_blocked_in_proposed: boolean
  block_reason?: string | null
  harmful_probability: number
  relevance_score?: number | null
  snippet: string
}

export interface SingleQueryConditionMetrics {
  condition: string
  answer: string
  top5_documents: SingleQueryDocument[]
  hd_count: number
  hd_at_top5: number
  gd_count: number
  hit_at_5: number
  gd_retention_rate: number
  claims: string[]
  has_harmful_exposure: boolean
  hallucination_risk: string
}

export interface SingleQueryEvaluation {
  query: string
  tau_threshold: number
  total_screened: number
  total_blocked: number
  blocked_document_ids: string[]
  control: SingleQueryConditionMetrics
  proposed: SingleQueryConditionMetrics
  hd_exposure_reduction: number
  fidelity_status: string
}

export interface AttributionSupport {
  source: number
  sentence: number
  score: number
}

export interface AttributionAnswerSentence {
  index: number
  text: string
  cited: number[]
  supported_by: AttributionSupport[]
}

export interface AttributionSourceSentence {
  text: string
  answer_sentences: number[]
  score: number
}

export interface Attribution {
  method: 'embedding' | 'lexical'
  threshold: number
  answer_sentences: AttributionAnswerSentence[]
  sources: { rank: number; sentences: AttributionSourceSentence[] }[]
}

export interface ClinicalResponse {
  query: string
  consensus_status: string
  clinical_summary: string
  trusted_sources: TrustedDocument[]
  atomic_claims: string[]
  safety_scan: SafetyScanSummary
  evaluation_breakdown?: SingleQueryEvaluation
  attribution?: Attribution | null
  llm_model?: string | null
  llm_endpoint?: string | null
}

export interface SampleCase {
  case_id: string
  title: string
  query: string
  description: string
}

export interface SystemHealth {
  status: string
  service: string
  pipeline: string
  stage_1_retriever: string
  stage_2_filter: {
    architecture: string
    threshold: number
    cutoff_k: number
  }
  stage_3_reranker: string
  stage_4_generator: string
}

const API_BASE = window.location.port === '8000' ? '' : 'http://127.0.0.1:8000'

export async function fetchEvaluationResults(): Promise<EvaluationResults> {
  const res = await fetch(`${API_BASE}/api/evaluation-results`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to load evaluation results (HTTP ${res.status})`)
  }
  return res.json()
}

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/api/health`)
  if (!res.ok) throw new Error(`Health check failed with HTTP ${res.status}`)
  return res.json()
}

export async function fetchSampleCases(): Promise<SampleCase[]> {
  const res = await fetch(`${API_BASE}/api/sample-cases`)
  if (!res.ok) throw new Error(`Failed to load preset cases with HTTP ${res.status}`)
  return res.json()
}

export async function executeClinicalQuery(query: string, case_id?: string): Promise<ClinicalResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: query.trim(), case_id }),
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    throw new Error(errorData.detail || `Server responded with status ${res.status}`)
  }

  return await res.json()
}

export interface HighlightedPdf {
  url: string
  found: number
  requested: number
  firstPage: number
}

/**
 * Fetches the article's open-access PDF with the given sentences highlighted.
 * Throws with the server's reason (e.g. no open-access PDF) so the UI can fall back to the abstract.
 */
export async function fetchHighlightedPdf(pmid: string, sentences: string[], notes: string[]): Promise<HighlightedPdf> {
  const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(pmid)}/highlighted-pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sentences, notes }),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    const detail = data.detail
    throw new Error(typeof detail === 'string' ? detail : detail?.reason || `Server responded with status ${res.status}`)
  }
  const blob = await res.blob()
  return {
    url: URL.createObjectURL(blob),
    found: Number(res.headers.get('X-Highlights-Found') ?? 0),
    requested: Number(res.headers.get('X-Highlights-Requested') ?? 0),
    firstPage: Number(res.headers.get('X-First-Page') ?? 1),
  }
}

export interface FullTextStatus {
  pmid: string
  pmcid?: string | null
  is_open_access: boolean
  license?: string | null
  has_pdf: boolean
  error?: string | null
}

/** Whether the article has an open-access PDF in PubMed Central (the backend downloads and caches it). */
export async function fetchFullTextStatus(pmid: string): Promise<FullTextStatus> {
  const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(pmid)}/fulltext`)
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(typeof data.detail === 'string' ? data.detail : `Server responded with status ${res.status}`)
  }
  return await res.json()
}
