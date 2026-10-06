/** User-facing source label; the synthetic test documents are named plainly, not as debug strings. */
export function displaySource(source: string | null | undefined): string {
  const s = source ?? ''
  if (/synthetic/i.test(s)) return 'Synthetic test document · not a real study'
  if (/pubmedqa/i.test(s)) return 'PubMed · PubMedQA'
  return s || 'PubMed'
}

export function isSynthetic(source: string | null | undefined): boolean {
  return /synthetic/i.test(source ?? '')
}

export const CATEGORY_STYLE: Record<string, { label: string; className: string; title: string }> = {
  GD: {
    label: 'GD',
    title: 'Ground-truth document for this question',
    className: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/50 dark:text-emerald-300 dark:border-emerald-900',
  },
  MD: {
    label: 'MD',
    title: 'Mediocre: related, but not the answer to this question',
    className: 'bg-slate-50 text-slate-600 border-slate-200 dark:bg-slate-900/60 dark:text-slate-300 dark:border-slate-800',
  },
  HD: {
    label: 'HD',
    title: 'Harmful: looks relevant but points to the wrong answer',
    className: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/50 dark:text-red-300 dark:border-red-900',
  },
}

/** Splits "Decision: yes Explanation: ..." into the decision and the explanation text. */
export function splitDecision(text: string): { decision: string | null; body: string } {
  const m = /^\s*decision\s*:\s*(yes|no|maybe)\b[.:]?\s*/i.exec(text ?? '')
  const rest = m ? text.slice(m[0].length) : text ?? ''
  return { decision: m ? m[1].toLowerCase() : null, body: rest.replace(/^\s*explanation\s*:\s*/i, '') }
}

export const DECISION_STYLE: Record<string, string> = {
  yes: 'bg-emerald-50 text-emerald-800 border-emerald-200 dark:bg-emerald-950/50 dark:text-emerald-200 dark:border-emerald-900',
  no: 'bg-rose-50 text-rose-800 border-rose-200 dark:bg-rose-950/50 dark:text-rose-200 dark:border-rose-900',
  maybe: 'bg-amber-50 text-amber-800 border-amber-200 dark:bg-amber-950/50 dark:text-amber-200 dark:border-amber-900',
}

export function pct(n: number, d: number): string {
  return d ? `${Math.round((100 * n) / d)}%` : '–'
}
