import { useState } from 'react'
import type { Doc } from '../lib/schema'
import LabelChip from './LabelChip'
import GlossaryTooltip from './GlossaryTooltip'

interface Props {
  doc: Doc
  threshold?: number
  showFilterInfo?: boolean
  compact?: boolean
}

export default function DocCard({ doc, threshold, showFilterInfo = false, compact = false }: Props) {
  const [expanded, setExpanded] = useState(false)
  const label = doc.annotatedLabel
  const predicted = !label && doc.filter

  return (
    <div
      className={`rounded border border-border bg-card ${compact ? 'p-2' : 'p-3'} space-y-2 text-sm animate-settle`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-xs text-muted-foreground">{doc.id}</span>
          {label ? (
            <LabelChip label={label} />
          ) : doc.filter ? (
            <LabelChip
              label={doc.filter.pHD > (threshold ?? 0.45) ? 'HD' : doc.filter.pGD > doc.filter.pMD ? 'GD' : 'MD'}
              predicted
            />
          ) : null}
          {doc.rerank && (
            <span className="text-[10px] font-mono text-muted-foreground">
              rank #{doc.rerank.rank} · score {doc.rerank.score.toFixed(3)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="text-[10px] font-mono text-muted-foreground">
            ret.{doc.retrieverRank} · {doc.retrieverScore.toFixed(3)}
          </span>
        </div>
      </div>

      {/* Filter probability bar (treatment only) */}
      {showFilterInfo && doc.filter && (
        <div>
          <div className="text-[10px] text-muted-foreground mb-1 font-mono flex justify-between">
            <span>Filter probability</span>
            {threshold !== undefined && (
              <span>
                threshold{' '}
                <GlossaryTooltip term="FPR">
                  <span className="font-semibold">{threshold.toFixed(2)}</span>
                </GlossaryTooltip>
              </span>
            )}
          </div>
          <div className="h-3 flex rounded overflow-hidden border border-border/50">
            <div
              title={`P(GD)=${doc.filter.pGD.toFixed(2)}`}
              style={{ width: `${doc.filter.pGD * 100}%`, backgroundColor: 'var(--color-gd)' }}
              className="opacity-80"
            />
            <div
              title={`P(HD)=${doc.filter.pHD.toFixed(2)}`}
              style={{ width: `${doc.filter.pHD * 100}%`, backgroundColor: 'var(--color-hd)' }}
              className="opacity-80 relative"
            >
              {threshold !== undefined && (
                <div
                  className="absolute top-0 bottom-0 w-px bg-foreground/80"
                  style={{ left: `${(threshold / doc.filter.pHD) * 100}%` }}
                />
              )}
            </div>
            <div
              title={`P(MD)=${doc.filter.pMD.toFixed(2)}`}
              style={{ width: `${doc.filter.pMD * 100}%`, backgroundColor: 'var(--color-md)' }}
              className="opacity-80"
            />
          </div>
          <div className="flex gap-3 mt-0.5">
            {(['pGD', 'pHD', 'pMD'] as const).map((k) => (
              <span key={k} className="text-[9px] font-mono text-muted-foreground">
                {k.slice(1)}: {doc.filter![k].toFixed(2)}
              </span>
            ))}
          </div>
          {doc.filter.removed && (
            <div className="mt-1 flex items-center gap-1.5">
              <span className="text-[10px] font-mono font-semibold text-[var(--color-hd)]">
                ✕ Removed by filter
              </span>
              {doc.annotatedLabel === 'GD' && (
                <span className="text-[10px] text-amber-600 dark:text-amber-400 font-semibold">
                  ⚠ False positive — annotated GD
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Passage snippet */}
      <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
        {doc.text}
      </p>

      {expanded && (
        <div className="pt-2 border-t border-border/50 space-y-1">
          <p className="text-xs leading-relaxed">{doc.text}</p>
          {doc.filter && (
            <div className="text-[10px] font-mono text-muted-foreground space-y-0.5">
              <div>P(GD): {doc.filter.pGD.toFixed(4)} · P(HD): {doc.filter.pHD.toFixed(4)} · P(MD): {doc.filter.pMD.toFixed(4)}</div>
              <div>Retriever rank: {doc.retrieverRank} · score: {doc.retrieverScore.toFixed(4)}</div>
              {doc.rerank && <div>Reranker rank: {doc.rerank.rank} · score: {doc.rerank.score.toFixed(4)}</div>}
            </div>
          )}
        </div>
      )}

      <button
        onClick={() => setExpanded((v) => !v)}
        className="text-[10px] text-muted-foreground hover:text-foreground transition-colors font-mono"
      >
        {expanded ? '↑ Collapse' : '↓ Expand'}
      </button>
    </div>
  )
}
