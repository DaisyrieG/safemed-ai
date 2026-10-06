import { z } from 'zod'

export const LabelSchema = z.enum(['GD', 'HD', 'MD'])
export type Label = z.infer<typeof LabelSchema>

export const ConfigSchema = z.enum(['control', 'treatment'])
export type Config = z.infer<typeof ConfigSchema>

export const ModeSchema = z.enum(['test', 'free'])
export type Mode = z.infer<typeof ModeSchema>

export const VerdictSchema = z.enum(['supported', 'unsupported'])
export type Verdict = z.infer<typeof VerdictSchema>

export const DocSchema = z.object({
  id: z.string(),
  text: z.string(),
  retrieverRank: z.number().int().min(1).max(50),
  retrieverScore: z.number(),
  annotatedLabel: LabelSchema.optional(),
  filter: z
    .object({
      pGD: z.number(),
      pHD: z.number(),
      pMD: z.number(),
      removed: z.boolean(),
    })
    .optional(),
  rerank: z
    .object({
      rank: z.number().int(),
      score: z.number(),
    })
    .optional(),
})
export type Doc = z.infer<typeof DocSchema>

export const ClaimSchema = z.object({
  text: z.string(),
  verdict: VerdictSchema,
})
export type Claim = z.infer<typeof ClaimSchema>

export const ConfigRunSchema = z.object({
  config: ConfigSchema,
  poolAfterFilter: z.array(z.string()),
  top5: z.array(z.string()),
  answer: z.string(),
  claims: z.array(ClaimSchema).optional(),
  metrics: z
    .object({
      her: z.number(),
      gtrr: z.number(),
      factPrecision: z.number(),
      factRecall: z.number(),
      factF1: z.number(),
      factHallucinationRate: z.number(),
    })
    .optional(),
})
export type ConfigRun = z.infer<typeof ConfigRunSchema>

export const QueryRunSchema = z.object({
  queryId: z.string(),
  mode: ModeSchema,
  question: z.string(),
  referenceLabel: z.enum(['yes', 'no', 'maybe']).optional(),
  referenceContext: z.string().optional(),
  pool: z.array(DocSchema),
  threshold: z.number(),
  runs: z.object({
    control: ConfigRunSchema,
    treatment: ConfigRunSchema,
  }),
})
export type QueryRun = z.infer<typeof QueryRunSchema>

export const EvaluationSummarySchema = z.object({
  perQueryRows: z.array(
    z.object({
      queryId: z.string(),
      config: ConfigSchema,
      docIds: z.array(z.string()),
      labels: z.array(LabelSchema.nullable()),
      her: z.number(),
      gtrr: z.number(),
      answer: z.string(),
      claims: z.array(ClaimSchema),
      factPrecision: z.number(),
      factRecall: z.number(),
      factF1: z.number(),
      factHallucinationRate: z.number(),
    })
  ),
  filterConfusion: z.object({ tp: z.number(), fp: z.number(), tn: z.number(), fn: z.number() }),
  kappa: z.object({ document: z.number(), claim: z.number() }),
  kappaConfusion: z.array(z.array(z.number())),
  benchmarkReference: z
    .object({ value: z.number(), citation: z.string() })
    .optional(),
  runMeta: z.object({
    runId: z.string(),
    threshold: z.number(),
    nQueries: z.number(),
    pythonVersion: z.string(),
  }),
})
export type EvaluationSummary = z.infer<typeof EvaluationSummarySchema>
