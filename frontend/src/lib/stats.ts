/**
 * Pure, framework-free statistics module.
 * All functions are deterministic and side-effect-free.
 * Implements only the tests described in the thesis (section 1.6).
 */

/** erfc approximation for chi2 p-value at df=1 */
export function erfc(x: number): number {
  // Abramowitz & Stegun approximation (accurate to ~1.5e-7)
  const t = 1 / (1 + 0.3275911 * x)
  const poly =
    t *
    (0.254829592 +
      t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
  return poly * Math.exp(-x * x)
}

/**
 * Chi-square test of independence for a 2x2 contingency table.
 * No continuity correction. df = 1.
 * Returns chi2, p-value, and phi effect size.
 *
 * table = [[a, b], [c, d]]
 *   rows = configurations, cols = outcome present/absent
 */
export function chiSquareTest(table: [[number, number], [number, number]]): {
  chi2: number
  p: number
  phi: number
  N: number
} {
  const [[a, b], [c, d]] = table
  const N = a + b + c + d
  if (N === 0) return { chi2: 0, p: 1, phi: 0, N: 0 }

  const rowTotals = [a + b, c + d]
  const colTotals = [a + c, b + d]

  const expected = [
    [(rowTotals[0] * colTotals[0]) / N, (rowTotals[0] * colTotals[1]) / N],
    [(rowTotals[1] * colTotals[0]) / N, (rowTotals[1] * colTotals[1]) / N],
  ]

  const observed = [
    [a, b],
    [c, d],
  ]

  let chi2 = 0
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const e = expected[i][j]
      if (e > 0) {
        chi2 += Math.pow(observed[i][j] - e, 2) / e
      }
    }
  }

  const p = erfc(Math.sqrt(chi2 / 2))
  const phi = Math.sqrt(chi2 / N)
  return { chi2, p, phi, N }
}

/** Per-query HD@Top-5 (Harmful Document Exposure): harmful count in top-5 / 5. Discrete: {0, 0.2, 0.4, 0.6, 0.8, 1.0} */
export function computeHDTop5(labels: (string | null | undefined)[]): number | null {
  const top5 = labels.slice(0, 5)
  if (top5.some((l) => l == null)) return null
  const harmfulCount = top5.filter((l) => l === 'HD').length
  return harmfulCount / 5
}

/** Per-query Hit@5: 1 if at least one ground-truth document is in top-5, else 0 */
export function computeHit5(labels: (string | null | undefined)[]): number | null {
  const top5 = labels.slice(0, 5)
  if (top5.some((l) => l == null)) return null
  const gtCount = top5.filter((l) => l === 'GD').length
  return gtCount > 0 ? 1 : 0
}

/** Per-query GD Retention Rate: ground-truth count in top-5 / 5 */
export function computeGDRetention(labels: (string | null | undefined)[]): number | null {
  const top5 = labels.slice(0, 5)
  if (top5.some((l) => l == null)) return null
  const gtCount = top5.filter((l) => l === 'GD').length
  return gtCount / 5
}

// V3 Aliases for backwards compatibility
export const computeHER = computeHDTop5
export const computeGTRR = computeGDRetention

/** Mean and standard deviation */
export function meanSD(values: number[]): { mean: number; sd: number } {
  if (values.length === 0) return { mean: 0, sd: 0 }
  const mean = values.reduce((s, v) => s + v, 0) / values.length
  const variance =
    values.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / values.length
  return { mean, sd: Math.sqrt(variance) }
}

/** Filter classification metrics. Harmful = positive class. */
export function filterMetrics(confusion: {
  tp: number
  fp: number
  tn: number
  fn: number
}): {
  fpr: number
  fnr: number
  precision: number
  recall: number
  f1: number
} {
  const { tp, fp, tn, fn } = confusion
  const fpr = fp + tn > 0 ? fp / (fp + tn) : 0
  const fnr = tp + fn > 0 ? fn / (tp + fn) : 0
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0
  const f1 =
    precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0
  return { fpr, fnr, precision, recall, f1 }
}

/** Cohen's kappa for two flat arrays of categorical labels */
export function cohensKappa(
  predicted: string[],
  actual: string[]
): number {
  if (predicted.length !== actual.length || predicted.length === 0) return 0
  const n = predicted.length
  const categories = Array.from(new Set([...predicted, ...actual]))

  let po = 0
  for (let i = 0; i < n; i++) {
    if (predicted[i] === actual[i]) po++
  }
  po /= n

  let pe = 0
  for (const cat of categories) {
    const pPred = predicted.filter((x) => x === cat).length / n
    const pAct = actual.filter((x) => x === cat).length / n
    pe += pPred * pAct
  }

  if (pe === 1) return 1
  return (po - pe) / (1 - pe)
}

/**
 * Build 2x2 contingency table for H1 (harmful presence) from per-query rows.
 * Rows = [Standard RAG, Proposed]; Cols = [harmful_present, harmful_absent]
 */
export function buildH1Table(
  controlLabels: (string | null)[],
  treatmentLabels: (string | null)[]
): [[number, number], [number, number]] {
  const countHD = (labels: (string | null)[]) =>
    labels.filter((l) => l === 'HD').length
  const cHD = countHD(controlLabels)
  const cTotal = controlLabels.length
  const tHD = countHD(treatmentLabels)
  const tTotal = treatmentLabels.length
  return [
    [cHD, cTotal - cHD],
    [tHD, tTotal - tHD],
  ]
}

/**
 * Build 2x2 contingency table for H2 (ground-truth presence).
 */
export function buildH2Table(
  controlLabels: (string | null)[],
  treatmentLabels: (string | null)[]
): [[number, number], [number, number]] {
  const countGD = (labels: (string | null)[]) =>
    labels.filter((l) => l === 'GD').length
  const cGD = countGD(controlLabels)
  const cTotal = controlLabels.length
  const tGD = countGD(treatmentLabels)
  const tTotal = treatmentLabels.length
  return [
    [cGD, cTotal - cGD],
    [tGD, tTotal - tGD],
  ]
}

/** Fact-level metrics from claims */
export function factMetrics(
  claims: { verdict: 'supported' | 'unsupported' }[]
): {
  precision: number
  recall: number
  f1: number
  hallucinationRate: number
} {
  if (claims.length === 0)
    return { precision: 0, recall: 0, f1: 0, hallucinationRate: 0 }
  const supported = claims.filter((c) => c.verdict === 'supported').length
  const precision = supported / claims.length
  const hallucinationRate = 1 - precision
  // Recall and F1 are approximated using supported as TP and total as TP+FP+FN
  const recall = precision // simplified: recall = precision when reference == claims
  const f1 =
    precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0
  return { precision, recall, f1, hallucinationRate }
}
