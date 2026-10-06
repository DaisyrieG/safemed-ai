import { useState } from 'react'
import type { ClinicalResponse, DocumentEvaluationEntry } from '../../lib/api'
import { CATEGORY_STYLE, pct } from '../../lib/display'
import { IconShield } from '../icons'

export interface SessionRun {
  query: string
  controlHallucinated: boolean
  proposedHallucinated: boolean
  controlHdInduced: boolean
  proposedHdInduced: boolean
}

interface Props {
  response: ClinicalResponse | null
  loading: boolean
  session: SessionRun[]
  onOpen: (doc: DocumentEvaluationEntry) => void
}

function DocRow({ doc, tau, onOpen }: { doc: DocumentEvaluationEntry; tau: number | null; onOpen: () => void }) {
  const style = CATEGORY_STYLE[doc.category] ?? CATEGORY_STYLE.MD
  return (
    <button
      type="button"
      onClick={onOpen}
      className="w-full text-left px-3 py-2.5 rounded-xl hover:bg-muted transition-colors cursor-pointer"
    >
      <p className="text-[13px] leading-snug line-clamp-2">{doc.title}</p>
      <div className="mt-1 flex items-center gap-2 text-[11px] text-muted-foreground">
        <span title={style.title} className={`px-1.5 py-px rounded border font-medium ${style.className}`}>
          {style.label}
        </span>
        <span className={`font-mono ${tau !== null && doc.harmful_probability >= tau ? 'text-red-600 dark:text-red-400' : ''}`}>
          P(HD) {doc.harmful_probability.toFixed(2)}
        </span>
        {doc.reinstated && <span className="text-amber-700 dark:text-amber-300">reinstated</span>}
      </div>
    </button>
  )
}

export default function SafetyPanel({ response, loading, session, onOpen }: Props) {
  const [showAll, setShowAll] = useState(false)
  const scan = response?.safety_scan
  const tau = response?.tau_safe ?? null
  const blocked = scan?.evaluation_log.filter((d) => d.is_blocked) ?? []
  const retained = scan?.evaluation_log.filter((d) => !d.is_blocked) ?? []

  const n = session.length
  const ctrl = session.filter((r) => r.controlHallucinated).length
  const prop = session.filter((r) => r.proposedHallucinated).length
  const ctrlHd = session.filter((r) => r.controlHdInduced).length
  const propHd = session.filter((r) => r.proposedHdInduced).length

  return (
    <section className="h-full flex flex-col bg-card border border-border rounded-2xl overflow-hidden">
      <header className="px-4 py-3 border-b border-border flex items-center gap-2">
        <IconShield className="text-primary" />
        <h2 className="text-sm font-semibold">Safety filter</h2>
        <span className="ml-auto text-[11px] text-muted-foreground">Stage 2 · pre-reranking</span>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {loading && <div className="m-4 h-24 rounded-xl bg-muted animate-pulse" />}

        {!loading && !scan && (
          <p className="text-sm text-muted-foreground p-4 leading-relaxed">
            Every retrieved candidate is scored by a fine-tuned cross-encoder. Documents whose harmful probability reaches
            τ are removed before reranking.
          </p>
        )}

        {!loading && scan && (
          <>
            <div className="grid grid-cols-3 gap-px bg-border border-b border-border">
              {[
                ['Screened', scan.total_scanned],
                ['Blocked', scan.blocked_count],
                ['τ', tau !== null ? tau.toFixed(2) : '–'],
              ].map(([label, value]) => (
                <div key={label} className="bg-card px-3 py-3">
                  <div className="text-[11px] text-muted-foreground">{label}</div>
                  <div className={`text-lg font-semibold font-mono ${label === 'Blocked' && scan.blocked_count ? 'text-red-600 dark:text-red-400' : ''}`}>
                    {value}
                  </div>
                </div>
              ))}
            </div>

            <div className="p-2">
              <h3 className="px-3 pt-2 pb-1 text-xs font-medium text-muted-foreground">
                {blocked.length ? 'Blocked before reranking' : 'No document reached τ'}
              </h3>
              {blocked.map((doc, i) => (
                <DocRow key={doc.id ?? i} doc={doc} tau={tau} onOpen={() => onOpen(doc)} />
              ))}

              <button
                type="button"
                onClick={() => setShowAll((v) => !v)}
                className="mx-3 mt-2 mb-1 text-xs text-primary hover:underline cursor-pointer"
              >
                {showAll ? 'Hide' : 'Show'} the {retained.length} retained candidates
              </button>
              {showAll && retained.map((doc, i) => <DocRow key={doc.id ?? i} doc={doc} tau={tau} onOpen={() => onOpen(doc)} />)}
            </div>

            <div className="mx-4 mb-4 mt-1 p-3 rounded-xl bg-muted/60 text-[12px] text-muted-foreground leading-relaxed">
              <span className="font-mono">{scan.total_scanned}</span> retrieved → <span className="font-mono">{scan.blocked_count}</span> blocked →
              top <span className="font-mono">30</span> to the reranker → <span className="font-mono">5</span> to the generator
            </div>
          </>
        )}

        <div className="border-t border-border p-4">
          <h3 className="text-xs font-medium text-muted-foreground mb-2">Hallucination rate · this session</h3>
          {n === 0 ? (
            <p className="text-[12px] text-muted-foreground leading-relaxed">
              Share of demo questions whose answer has at least one unsupported or contradicted claim, with and without the filter.
            </p>
          ) : (
            <div className="space-y-2">
              {[
                ['Without filter', ctrl, ctrlHd],
                ['With filter', prop, propHd],
              ].map(([label, h, hd]) => (
                <div key={label as string}>
                  <div className="flex items-baseline justify-between text-[13px]">
                    <span>{label}</span>
                    <span className="font-mono font-semibold">
                      {pct(h as number, n)} <span className="text-muted-foreground font-normal">({h}/{n})</span>
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-muted overflow-hidden mt-1">
                    <div
                      className={`h-full rounded-full ${label === 'With filter' ? 'bg-primary' : 'bg-slate-400'}`}
                      style={{ width: n ? `${(100 * (h as number)) / n}%` : 0 }}
                    />
                  </div>
                  <div className="text-[11px] text-muted-foreground mt-0.5">HD-induced: {hd}/{n}</div>
                </div>
              ))}
              <p className="text-[11px] text-muted-foreground/80 pt-1">Demo questions only. The thesis results come from the benchmark run.</p>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
