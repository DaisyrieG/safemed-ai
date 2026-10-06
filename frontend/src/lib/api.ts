/**
 * SafeMed AI API client for the backend at http://127.0.0.1:8000.
 */

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
  decision?: string | null
  tau_safe?: number | null
  control_answer?: string | null
  hallucination_check?: HallucinationCheck | null
  hallucination_error?: string | null
}

export type ClaimStatus = 'SUPPORTED' | 'UNSUPPORTED' | 'CONTRADICTED'

export interface ClaimVerdict {
  claim: string
  status: ClaimStatus
  reasoning: string
  induced_by?: string | null
}

export interface AnswerCheck {
  claims: ClaimVerdict[]
  n_claims: number
  n_supported: number
  n_unsupported: number
  n_contradicted: number
  hallucinated: boolean
  hd_induced: boolean
  unsupported_claim_rate: number
  decision: string
  answer_correct: boolean | null
}

/** Stage 5 judge results for one query (Chapter 3 Hallucination Scoring Protocol). */
export interface HallucinationCheck {
  reference: 'pubmedqa' | 'sources'
  reference_answer?: string | null
  source_id?: string | null
  judge_model: string
  proposed: AnswerCheck
  control: AnswerCheck
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
  stage_4_ready?: boolean
  stage_5_judge?: string
  stage_5_ready?: boolean
}

const API_BASE = window.location.port === '8000' ? '' : 'http://127.0.0.1:8000'

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

export async function executeClinicalQuery(query: string, case_id?: string, check_hallucination = true): Promise<ClinicalResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: query.trim(), case_id, check_hallucination }),
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

/** Runs the Stage 5 judge on both answers of a query already answered by executeClinicalQuery. */
export async function runHallucinationCheck(response: ClinicalResponse, case_id?: string): Promise<HallucinationCheck> {
  const res = await fetch(`${API_BASE}/api/hallucination-check`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: response.query,
      case_id,
      proposed: { answer: response.clinical_summary, doc_ids: response.trusted_sources.map((s) => String(s.id ?? '')) },
      control: {
        answer: response.control_answer ?? '',
        doc_ids: (response.evaluation_breakdown?.control.top5_documents ?? []).map((d) => String(d.id ?? '')),
      },
    }),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(typeof data.detail === 'string' ? data.detail : `Server responded with status ${res.status}`)
  }
  return res.json()
}
