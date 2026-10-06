import { useState } from 'react'
import type { AnswerCheck, ClaimStatus, HallucinationCheck } from '../../lib/api'
import { DECISION_STYLE, splitDecision } from '../../lib/display'
import { IconAlert, IconCheck, IconChevron } from '../icons'

const STATUS_STYLE: Record<ClaimStatus, string> = {
  SUPPORTED: 'text-emerald-700 bg-emerald-50 border-emerald-200 dark:text-emerald-300 dark:bg-emerald-950/50 dark:border-emerald-900',
  UNSUPPORTED: 'text-amber-800 bg-amber-50 border-amber-200 dark:text-amber-200 dark:bg-amber-950/50 dark:border-amber-900',
  CONTRADICTED: 'text-red-700 bg-red-50 border-red-200 dark:text-red-300 dark:bg-red-950/50 dark:border-red-900',
}

function Column({ title, check, answer, referenceAnswer }: {
  title: string
  check: AnswerCheck
  answer?: string | null
  referenceAnswer?: string | null
}) {
  const [open, setOpen] = useState(false)
  const [showAnswer, setShowAnswer] = useState(false)
  const bad = check.n_unsupported + check.n_contradicted

  return (
    <div className="flex-1 min-w-0 p-4">
      <div className="text-xs text-muted-foreground mb-2">{title}</div>
      <div className={`flex items-center gap-2 text-sm font-semibold ${check.hallucinated ? 'text-red-700 dark:text-red-300' : 'text-emerald-700 dark:text-emerald-300'}`}>
        {check.hallucinated ? <IconAlert /> : <IconCheck />}
        {check.hallucinated ? 'Hallucination detected' : 'No hallucination'}
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-y-1.5 text-[12px]">
        <dt className="text-muted-foreground">Claims supported</dt>
        <dd className="font-mono text-right">{check.n_supported}/{check.n_claims}</dd>
        <dt className="text-muted-foreground">Unsupported / contradicted</dt>
        <dd className="font-mono text-right">{check.n_unsupported} / {check.n_contradicted}</dd>
        <dt className="text-muted-foreground">Unsupported claim rate</dt>
        <dd className="font-mono text-right">{(100 * check.unsupported_claim_rate).toFixed(0)}%</dd>
        <dt className="text-muted-foreground">Caused by a harmful doc</dt>
        <dd className="text-right">{check.hd_induced ? 'yes' : 'no'}</dd>
        <dt className="text-muted-foreground">Decision</dt>
        <dd className="text-right">
          <span className={`px-1.5 py-px rounded border text-[11px] ${DECISION_STYLE[check.decision] ?? 'border-border'}`}>
            {check.decision || 'none'}
          </span>
          {check.answer_correct !== null && referenceAnswer && (
            <span className={`ml-1.5 ${check.answer_correct ? 'text-emerald-700 dark:text-emerald-300' : 'text-red-700 dark:text-red-300'}`}>
              {check.answer_correct ? 'matches expert' : 'differs'}
            </span>
          )}
        </dd>
      </dl>

      <div className="mt-3 flex flex-wrap gap-3">
        {check.n_claims > 0 && (
          <button type="button" onClick={() => setOpen((v) => !v)} className="inline-flex items-center gap-1 text-xs text-primary hover:underline cursor-pointer">
            <IconChevron size={12} className={`transition-transform ${open ? 'rotate-90' : ''}`} />
            {open ? 'Hide' : 'Show'} {check.n_claims} claims{bad ? ` (${bad} flagged)` : ''}
          </button>
        )}
        {answer && (
          <button type="button" onClick={() => setShowAnswer((v) => !v)} className="inline-flex items-center gap-1 text-xs text-primary hover:underline cursor-pointer">
            <IconChevron size={12} className={`transition-transform ${showAnswer ? 'rotate-90' : ''}`} />
            {showAnswer ? 'Hide' : 'Show'} this answer
          </button>
        )}
      </div>

      {showAnswer && answer && (
        <p className="mt-2 text-[13px] leading-relaxed p-3 rounded-xl bg-muted/60">{splitDecision(answer).body}</p>
      )}

      {open && (
        <ul className="mt-2 space-y-1.5">
          {check.claims.map((c, i) => (
            <li key={i} className="text-[12px] leading-snug flex gap-2">
              <span className={`shrink-0 h-fit px-1.5 py-px rounded border text-[10px] font-medium ${STATUS_STYLE[c.status]}`}>
                {c.status.toLowerCase()}
              </span>
              <span title={c.reasoning}>
                {c.claim}
                {c.induced_by && <span className="text-red-700 dark:text-red-300"> · from a harmful document</span>}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default function HallucinationCard({ check, error, controlAnswer }: {
  check?: HallucinationCheck | null
  error?: string | null
  controlAnswer?: string | null
}) {
  if (!check) {
    return error ? (
      <div className="rounded-2xl border border-amber-200 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-900 p-4 text-[13px] text-amber-900 dark:text-amber-200">
        Hallucination check unavailable: {error}
      </div>
    ) : null
  }

  return (
    <div className="rounded-2xl border border-border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <h3 className="text-sm font-semibold">Hallucination check</h3>
        <span className="text-[12px] text-muted-foreground">
          Claims judged by {check.judge_model} against{' '}
          {check.reference === 'pubmedqa'
            ? <>the PubMedQA expert evidence{check.reference_answer && <> (expert answer: <b>{check.reference_answer}</b>)</>}</>
            : 'the documents each answer was given (no expert reference for this question)'}
        </span>
      </div>
      <div className="flex flex-col sm:flex-row divide-y sm:divide-y-0 sm:divide-x divide-border">
        <Column title="With the safety filter (SafeMed AI)" check={check.proposed} referenceAnswer={check.reference_answer} />
        <Column title="Without the filter (standard RAG)" check={check.control} answer={controlAnswer} referenceAnswer={check.reference_answer} />
      </div>
    </div>
  )
}
