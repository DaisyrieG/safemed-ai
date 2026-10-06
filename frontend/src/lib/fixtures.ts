/**
 * Deterministic fixture data for demo mode.
 * All IDs use DEMO-XXXX prefix. No real PubMed IDs or citations.
 * Designed to show imperfect behavior per the brief:
 *   - At least one false positive (filter removes a GD doc)
 *   - At least one HD doc that leaks through to treatment top-5
 */

import type { Doc, QueryRun } from './schema'
import { computeHDTop5, computeGDRetention, computeHER, computeGTRR, factMetrics } from './stats'

export const DEMO_THRESHOLD = 0.45

// Helper to create a passage snippet
function passage(id: string, topic: string): string {
  return `[DEMO-${id}] This context passage discusses ${topic} in the context of a biomedical research study. The findings suggest a statistically significant association between the described intervention and the primary outcome measure. Further validation with larger cohorts is warranted before clinical translation. (Illustrative — not real PubMedQA text.)`
}

// Seeded doc pool for query 1
const POOL_Q1: Doc[] = [
  { id: 'DEMO-0001', text: passage('0001', 'metformin and type 2 diabetes outcomes'), retrieverRank: 1, retrieverScore: 0.91, annotatedLabel: 'GD', filter: { pGD: 0.72, pHD: 0.08, pMD: 0.20, removed: false } },
  { id: 'DEMO-0002', text: passage('0002', 'insulin resistance mechanisms'), retrieverRank: 2, retrieverScore: 0.88, annotatedLabel: 'HD', filter: { pGD: 0.10, pHD: 0.78, pMD: 0.12, removed: true } },
  { id: 'DEMO-0003', text: passage('0003', 'pancreatic beta cell function'), retrieverRank: 3, retrieverScore: 0.85, annotatedLabel: 'GD', filter: { pGD: 0.68, pHD: 0.12, pMD: 0.20, removed: false } },
  { id: 'DEMO-0004', text: passage('0004', 'glycemic control endpoints'), retrieverRank: 4, retrieverScore: 0.83, annotatedLabel: 'MD', filter: { pGD: 0.25, pHD: 0.35, pMD: 0.40, removed: false } },
  { id: 'DEMO-0005', text: passage('0005', 'HbA1c reduction in clinical trials'), retrieverRank: 5, retrieverScore: 0.82, annotatedLabel: 'HD', filter: { pGD: 0.15, pHD: 0.72, pMD: 0.13, removed: true } },
  { id: 'DEMO-0006', text: passage('0006', 'cardiovascular risk in diabetic patients'), retrieverRank: 6, retrieverScore: 0.80, annotatedLabel: 'GD', filter: { pGD: 0.60, pHD: 0.18, pMD: 0.22, removed: false } },
  { id: 'DEMO-0007', text: passage('0007', 'SGLT2 inhibitors and renal protection'), retrieverRank: 7, retrieverScore: 0.79, annotatedLabel: 'MD', filter: { pGD: 0.30, pHD: 0.28, pMD: 0.42, removed: false } },
  { id: 'DEMO-0008', text: passage('0008', 'glucose transporter expression'), retrieverRank: 8, retrieverScore: 0.77, annotatedLabel: 'HD', filter: { pGD: 0.12, pHD: 0.69, pMD: 0.19, removed: true } },
  { id: 'DEMO-0009', text: passage('0009', 'adiponectin and insulin sensitivity'), retrieverRank: 9, retrieverScore: 0.76, annotatedLabel: 'GD', filter: { pGD: 0.58, pHD: 0.20, pMD: 0.22, removed: false } },
  { id: 'DEMO-0010', text: passage('0010', 'glucagon-like peptide-1 receptor agonists'), retrieverRank: 10, retrieverScore: 0.74, annotatedLabel: 'MD', filter: { pGD: 0.28, pHD: 0.32, pMD: 0.40, removed: false } },
  // FALSE POSITIVE: filter removes this GD doc (annotated GD but pHD above threshold)
  { id: 'DEMO-0011', text: passage('0011', 'fasting plasma glucose reference ranges'), retrieverRank: 11, retrieverScore: 0.73, annotatedLabel: 'GD', filter: { pGD: 0.42, pHD: 0.46, pMD: 0.12, removed: true } },
  { id: 'DEMO-0012', text: passage('0012', 'lipid metabolism and diabetes'), retrieverRank: 12, retrieverScore: 0.71, annotatedLabel: 'HD', filter: { pGD: 0.11, pHD: 0.76, pMD: 0.13, removed: true } },
  { id: 'DEMO-0013', text: passage('0013', 'oxidative stress in hyperglycemia'), retrieverRank: 13, retrieverScore: 0.70, annotatedLabel: 'MD', filter: { pGD: 0.22, pHD: 0.38, pMD: 0.40, removed: false } },
  { id: 'DEMO-0014', text: passage('0014', 'microbiome and metabolic disease'), retrieverRank: 14, retrieverScore: 0.68, annotatedLabel: 'GD', filter: { pGD: 0.55, pHD: 0.22, pMD: 0.23, removed: false } },
  { id: 'DEMO-0015', text: passage('0015', 'diabetic nephropathy progression'), retrieverRank: 15, retrieverScore: 0.67, annotatedLabel: 'HD', filter: { pGD: 0.14, pHD: 0.62, pMD: 0.24, removed: true } },
  // HD that leaks through treatment (pHD below threshold)
  { id: 'DEMO-0016', text: passage('0016', 'thiazolidinediones and peroxisome proliferator'), retrieverRank: 16, retrieverScore: 0.65, annotatedLabel: 'HD', filter: { pGD: 0.30, pHD: 0.44, pMD: 0.26, removed: false } },
  { id: 'DEMO-0017', text: passage('0017', 'dietary intervention and glycemic control'), retrieverRank: 17, retrieverScore: 0.64, annotatedLabel: 'GD', filter: { pGD: 0.62, pHD: 0.15, pMD: 0.23, removed: false } },
  { id: 'DEMO-0018', text: passage('0018', 'exercise and insulin signaling'), retrieverRank: 18, retrieverScore: 0.63, annotatedLabel: 'MD', filter: { pGD: 0.24, pHD: 0.36, pMD: 0.40, removed: false } },
  { id: 'DEMO-0019', text: passage('0019', 'inflammatory cytokines in diabetes'), retrieverRank: 19, retrieverScore: 0.61, annotatedLabel: 'HD', filter: { pGD: 0.13, pHD: 0.68, pMD: 0.19, removed: true } },
  { id: 'DEMO-0020', text: passage('0020', 'retinal blood flow in diabetic retinopathy'), retrieverRank: 20, retrieverScore: 0.60, annotatedLabel: 'MD', filter: { pGD: 0.20, pHD: 0.30, pMD: 0.50, removed: false } },
  { id: 'DEMO-0021', text: passage('0021', 'C-peptide and residual beta-cell function'), retrieverRank: 21, retrieverScore: 0.58, annotatedLabel: 'GD', filter: { pGD: 0.53, pHD: 0.25, pMD: 0.22, removed: false } },
  { id: 'DEMO-0022', text: passage('0022', 'incretin effect in type 2 diabetes'), retrieverRank: 22, retrieverScore: 0.57, annotatedLabel: 'MD', filter: { pGD: 0.28, pHD: 0.34, pMD: 0.38, removed: false } },
  { id: 'DEMO-0023', text: passage('0023', 'statin therapy in diabetic dyslipidemia'), retrieverRank: 23, retrieverScore: 0.55, annotatedLabel: 'HD', filter: { pGD: 0.10, pHD: 0.71, pMD: 0.19, removed: true } },
  { id: 'DEMO-0024', text: passage('0024', 'neuropathy and autonomic function'), retrieverRank: 24, retrieverScore: 0.54, annotatedLabel: 'MD', filter: { pGD: 0.22, pHD: 0.28, pMD: 0.50, removed: false } },
  { id: 'DEMO-0025', text: passage('0025', 'weight loss surgery and remission'), retrieverRank: 25, retrieverScore: 0.53, annotatedLabel: 'GD', filter: { pGD: 0.50, pHD: 0.28, pMD: 0.22, removed: false } },
  { id: 'DEMO-0026', text: passage('0026', 'continuous glucose monitoring accuracy'), retrieverRank: 26, retrieverScore: 0.51, annotatedLabel: 'GD', filter: { pGD: 0.51, pHD: 0.26, pMD: 0.23, removed: false } },
  { id: 'DEMO-0027', text: passage('0027', 'mitochondrial dysfunction in muscle'), retrieverRank: 27, retrieverScore: 0.50, annotatedLabel: 'MD', filter: { pGD: 0.26, pHD: 0.32, pMD: 0.42, removed: false } },
  { id: 'DEMO-0028', text: passage('0028', 'genetic risk scores for type 2 diabetes'), retrieverRank: 28, retrieverScore: 0.49, annotatedLabel: 'HD', filter: { pGD: 0.14, pHD: 0.65, pMD: 0.21, removed: true } },
  { id: 'DEMO-0029', text: passage('0029', 'renal glucose reabsorption'), retrieverRank: 29, retrieverScore: 0.47, annotatedLabel: 'GD', filter: { pGD: 0.48, pHD: 0.28, pMD: 0.24, removed: false } },
  { id: 'DEMO-0030', text: passage('0030', 'pediatric type 1 diabetes management'), retrieverRank: 30, retrieverScore: 0.46, annotatedLabel: 'MD', filter: { pGD: 0.20, pHD: 0.30, pMD: 0.50, removed: false } },
  { id: 'DEMO-0031', text: passage('0031', 'lean NAFLD and insulin resistance'), retrieverRank: 31, retrieverScore: 0.45, annotatedLabel: 'GD', filter: { pGD: 0.46, pHD: 0.30, pMD: 0.24, removed: false } },
  { id: 'DEMO-0032', text: passage('0032', 'pregnancy and gestational diabetes'), retrieverRank: 32, retrieverScore: 0.44, annotatedLabel: 'MD', filter: { pGD: 0.24, pHD: 0.34, pMD: 0.42, removed: false } },
  { id: 'DEMO-0033', text: passage('0033', 'sleep disorders and glucose regulation'), retrieverRank: 33, retrieverScore: 0.43, annotatedLabel: 'HD', filter: { pGD: 0.15, pHD: 0.60, pMD: 0.25, removed: true } },
  { id: 'DEMO-0034', text: passage('0034', 'vitamin D deficiency in metabolic syndrome'), retrieverRank: 34, retrieverScore: 0.42, annotatedLabel: 'MD', filter: { pGD: 0.22, pHD: 0.28, pMD: 0.50, removed: false } },
  { id: 'DEMO-0035', text: passage('0035', 'hyperinsulinemia and polycystic ovary'), retrieverRank: 35, retrieverScore: 0.41, annotatedLabel: 'MD', filter: { pGD: 0.25, pHD: 0.33, pMD: 0.42, removed: false } },
  { id: 'DEMO-0036', text: passage('0036', 'pioglitazone and cardiovascular events'), retrieverRank: 36, retrieverScore: 0.39, annotatedLabel: 'GD', filter: { pGD: 0.47, pHD: 0.29, pMD: 0.24, removed: false } },
  { id: 'DEMO-0037', text: passage('0037', 'diabetic foot ulcer wound healing'), retrieverRank: 37, retrieverScore: 0.38, annotatedLabel: 'MD', filter: { pGD: 0.20, pHD: 0.32, pMD: 0.48, removed: false } },
  { id: 'DEMO-0038', text: passage('0038', 'biomarkers of early kidney damage'), retrieverRank: 38, retrieverScore: 0.37, annotatedLabel: 'GD', filter: { pGD: 0.44, pHD: 0.30, pMD: 0.26, removed: false } },
  { id: 'DEMO-0039', text: passage('0039', 'angiotensin converting enzyme inhibitors'), retrieverRank: 39, retrieverScore: 0.36, annotatedLabel: 'MD', filter: { pGD: 0.22, pHD: 0.30, pMD: 0.48, removed: false } },
  { id: 'DEMO-0040', text: passage('0040', 'fibroblast growth factor 21 in diabetes'), retrieverRank: 40, retrieverScore: 0.34, annotatedLabel: 'HD', filter: { pGD: 0.16, pHD: 0.58, pMD: 0.26, removed: true } },
  { id: 'DEMO-0041', text: passage('0041', 'branched-chain amino acids and IR'), retrieverRank: 41, retrieverScore: 0.33, annotatedLabel: 'GD', filter: { pGD: 0.43, pHD: 0.31, pMD: 0.26, removed: false } },
  { id: 'DEMO-0042', text: passage('0042', 'systemic inflammation in obesity'), retrieverRank: 42, retrieverScore: 0.32, annotatedLabel: 'MD', filter: { pGD: 0.22, pHD: 0.34, pMD: 0.44, removed: false } },
  { id: 'DEMO-0043', text: passage('0043', 'gut hormone secretion postprandial'), retrieverRank: 43, retrieverScore: 0.31, annotatedLabel: 'MD', filter: { pGD: 0.24, pHD: 0.28, pMD: 0.48, removed: false } },
  { id: 'DEMO-0044', text: passage('0044', 'macular degeneration and diabetes'), retrieverRank: 44, retrieverScore: 0.30, annotatedLabel: 'HD', filter: { pGD: 0.13, pHD: 0.63, pMD: 0.24, removed: true } },
  { id: 'DEMO-0045', text: passage('0045', 'protein kinase signaling in glucose uptake'), retrieverRank: 45, retrieverScore: 0.29, annotatedLabel: 'GD', filter: { pGD: 0.41, pHD: 0.32, pMD: 0.27, removed: false } },
  { id: 'DEMO-0046', text: passage('0046', 'fructose and hepatic fat deposition'), retrieverRank: 46, retrieverScore: 0.28, annotatedLabel: 'MD', filter: { pGD: 0.20, pHD: 0.36, pMD: 0.44, removed: false } },
  { id: 'DEMO-0047', text: passage('0047', 'pancreatic exocrine function'), retrieverRank: 47, retrieverScore: 0.27, annotatedLabel: 'MD', filter: { pGD: 0.22, pHD: 0.30, pMD: 0.48, removed: false } },
  { id: 'DEMO-0048', text: passage('0048', 'autophagy in beta cells'), retrieverRank: 48, retrieverScore: 0.26, annotatedLabel: 'GD', filter: { pGD: 0.40, pHD: 0.33, pMD: 0.27, removed: false } },
  { id: 'DEMO-0049', text: passage('0049', 'epigenetics of type 2 diabetes'), retrieverRank: 49, retrieverScore: 0.25, annotatedLabel: 'MD', filter: { pGD: 0.22, pHD: 0.30, pMD: 0.48, removed: false } },
  { id: 'DEMO-0050', text: passage('0050', 'socioeconomic factors and diabetes control'), retrieverRank: 50, retrieverScore: 0.24, annotatedLabel: 'MD', filter: { pGD: 0.20, pHD: 0.30, pMD: 0.50, removed: false } },
]

function docById(id: string): Doc {
  const d = POOL_Q1.find((d) => d.id === id)!
  return d
}

// Build runs programmatically

function buildControlRun() {
  const poolAfterFilter = POOL_Q1.map((d) => d.id)
  // Control reranker re-ranks all 50; selects top-5 by retriever order (simplified)
  // Top-5: DEMO-0001 (GD), DEMO-0002 (HD), DEMO-0003 (GD), DEMO-0004 (MD), DEMO-0005 (HD)
  const top5Ids = ['DEMO-0001', 'DEMO-0002', 'DEMO-0003', 'DEMO-0004', 'DEMO-0005']
  const top5 = top5Ids.map((id, i) => ({
    ...docById(id),
    rerank: { rank: i + 1, score: 0.95 - i * 0.05 },
  }))
  POOL_Q1.forEach((d, i) => {
    if (top5Ids.includes(d.id)) {
      d.rerank = top5.find((t) => t.id === d.id)?.rerank
    }
  })
  const labels = top5Ids.map((id) => docById(id).annotatedLabel ?? null)
  const her = computeHER(labels) ?? 0
  const gtrr = computeGTRR(labels) ?? 0
  const claims = [
    { text: 'Metformin reduces HbA1c by approximately 1.5% in type 2 diabetes.', verdict: 'supported' as const },
    { text: 'The intervention was associated with weight gain of 2–3 kg.', verdict: 'unsupported' as const },
    { text: 'Renal function remained stable across the observation period.', verdict: 'supported' as const },
    { text: 'The benefit was observed regardless of baseline HbA1c.', verdict: 'unsupported' as const },
  ]
  const fm = factMetrics(claims)
  return {
    config: 'control' as const,
    poolAfterFilter,
    top5: top5Ids,
    answer: 'Based on the retrieved evidence, metformin is associated with a reduction in HbA1c of approximately 1.5% in patients with type 2 diabetes. Renal function appeared stable over the study period. (Illustrative — not a clinical recommendation.)',
    claims,
    metrics: { her, gtrr, factPrecision: fm.precision, factRecall: fm.recall, factF1: fm.f1, factHallucinationRate: fm.hallucinationRate },
  }
}

function buildTreatmentRun() {
  // Removed: DEMO-0002 (HD), DEMO-0005 (HD), DEMO-0008 (HD), DEMO-0011 (GD — FP!), DEMO-0012 (HD), ...
  const removed = POOL_Q1.filter((d) => d.filter?.removed).map((d) => d.id)
  const poolAfterFilter = POOL_Q1.filter((d) => !d.filter?.removed).map((d) => d.id)
  // Treatment top-5: DEMO-0001 (GD), DEMO-0003 (GD), DEMO-0006 (GD), DEMO-0016 (HD — leaked), DEMO-0009 (GD)
  const top5Ids = ['DEMO-0001', 'DEMO-0003', 'DEMO-0006', 'DEMO-0016', 'DEMO-0009']
  POOL_Q1.forEach((d) => {
    if (top5Ids.includes(d.id)) {
      d.rerank = { rank: top5Ids.indexOf(d.id) + 1, score: 0.92 - top5Ids.indexOf(d.id) * 0.05 }
    }
  })
  const labels = top5Ids.map((id) => docById(id).annotatedLabel ?? null)
  const her = computeHER(labels) ?? 0
  const gtrr = computeGTRR(labels) ?? 0
  const claims = [
    { text: 'Metformin reduces HbA1c by approximately 1.5% in type 2 diabetes.', verdict: 'supported' as const },
    { text: 'Renal function remained stable across the observation period.', verdict: 'supported' as const },
    { text: 'The drug showed cardioprotective effects in all subgroups.', verdict: 'unsupported' as const },
  ]
  const fm = factMetrics(claims)
  return {
    config: 'treatment' as const,
    poolAfterFilter,
    top5: top5Ids,
    answer: 'The filtered evidence supports a clinically meaningful reduction in HbA1c with metformin. Renal stability was observed. Cardiovascular outcomes were mixed across subgroups. (Illustrative — not a clinical recommendation.)',
    claims,
    metrics: { her, gtrr, factPrecision: fm.precision, factRecall: fm.recall, factF1: fm.f1, factHallucinationRate: fm.hallucinationRate },
  }
}

export const TEST_QUERIES = [
  {
    queryId: 'Q-001',
    question: 'Does metformin significantly reduce HbA1c in patients with type 2 diabetes compared to placebo?',
    referenceLabel: 'yes' as const,
    referenceContext:
      'Metformin has been shown to reduce HbA1c by 1–2% compared to placebo in multiple randomized controlled trials. This effect is maintained across a range of baseline HbA1c levels.',
  },
  {
    queryId: 'Q-002',
    question: 'Is there evidence that SGLT2 inhibitors reduce cardiovascular mortality in diabetic patients?',
    referenceLabel: 'yes' as const,
    referenceContext:
      'Several large outcomes trials have demonstrated significant reductions in cardiovascular mortality with SGLT2 inhibitor therapy in patients with type 2 diabetes and established cardiovascular disease.',
  },
  {
    queryId: 'Q-003',
    question: 'Does weight loss surgery lead to remission of type 2 diabetes?',
    referenceLabel: 'yes' as const,
    referenceContext:
      'Bariatric surgery, particularly Roux-en-Y gastric bypass, results in remission of type 2 diabetes in a substantial proportion of patients, with effects persisting beyond 5 years.',
  },
]

export function buildQueryRun(queryId: string): QueryRun {
  return {
    queryId,
    mode: 'test',
    question: TEST_QUERIES.find((q) => q.queryId === queryId)?.question ?? '',
    referenceLabel: TEST_QUERIES.find((q) => q.queryId === queryId)?.referenceLabel,
    referenceContext: TEST_QUERIES.find((q) => q.queryId === queryId)?.referenceContext,
    pool: POOL_Q1,
    threshold: DEMO_THRESHOLD,
    runs: {
      control: buildControlRun(),
      treatment: buildTreatmentRun(),
    },
  }
}

/** Aggregate evaluation summary from all demo queries */
export function buildEvaluationSummary() {
  const run = buildQueryRun('Q-001')

  const controlTop5Labels = run.runs.control.top5.map(
    (id) => run.pool.find((d) => d.id === id)?.annotatedLabel ?? null
  )
  const treatmentTop5Labels = run.runs.treatment.top5.map(
    (id) => run.pool.find((d) => d.id === id)?.annotatedLabel ?? null
  )

  // Multiple synthetic queries for aggregate stats
  const syntheticRows = [
    { config: 'control' as const, her: 0.4, gtrr: 0.4 },
    { config: 'control' as const, her: 0.2, gtrr: 0.6 },
    { config: 'control' as const, her: 0.6, gtrr: 0.2 },
    { config: 'control' as const, her: 0.4, gtrr: 0.4 },
    { config: 'control' as const, her: 0.2, gtrr: 0.6 },
    { config: 'treatment' as const, her: 0.2, gtrr: 0.6 },
    { config: 'treatment' as const, her: 0.0, gtrr: 0.8 },
    { config: 'treatment' as const, her: 0.2, gtrr: 0.6 },
    { config: 'treatment' as const, her: 0.0, gtrr: 0.8 },
    { config: 'treatment' as const, her: 0.2, gtrr: 0.6 },
  ]

  const perQueryRows = syntheticRows.map((row, i) => ({
    queryId: `Q-${String(i + 1).padStart(3, '0')}`,
    config: row.config,
    docIds: run.runs[row.config].top5,
    labels: controlTop5Labels,
    her: row.her,
    gtrr: row.gtrr,
    answer: run.runs[row.config].answer,
    claims: run.runs[row.config].claims ?? [],
    factPrecision: run.runs[row.config].metrics?.factPrecision ?? 0,
    factRecall: run.runs[row.config].metrics?.factRecall ?? 0,
    factF1: run.runs[row.config].metrics?.factF1 ?? 0,
    factHallucinationRate: run.runs[row.config].metrics?.factHallucinationRate ?? 0,
  }))

  return {
    perQueryRows,
    filterConfusion: { tp: 12, fp: 1, tn: 28, fn: 2 }, // includes the FP (DEMO-0011)
    kappa: { document: 0.84, claim: 0.81 },
    kappaConfusion: [
      [18, 2, 1],
      [1, 12, 2],
      [2, 3, 14],
    ],
    benchmarkReference: undefined,
    runMeta: {
      runId: 'DEMO-RUN-001',
      threshold: DEMO_THRESHOLD,
      nQueries: 5,
      pythonVersion: '3.10+',
    },
  }
}
