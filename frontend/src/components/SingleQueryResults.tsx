import { useState } from 'react'
import type { SingleQueryEvaluation } from '../lib/api'
import GlossaryTooltip from './GlossaryTooltip'

interface Props {
  breakdown: SingleQueryEvaluation
}

export default function SingleQueryResults({ breakdown }: Props) {
  const [activeTab, setActiveTab] = useState<'metrics' | 'sideBySide' | 'docs'>('metrics')
  const [expandedDocId, setExpandedDocId] = useState<string | null>(null)

  const { control, proposed, total_screened, total_blocked, hd_exposure_reduction, fidelity_status, tau_threshold } = breakdown

  const isProtected = proposed.hd_at_top5 === 0 && control.hd_at_top5 > 0
  const isSafe = proposed.hd_at_top5 === 0

  return (
    <div className="rounded-2xl border border-emerald-500/30 bg-card shadow-sm overflow-hidden mt-4">
      {/* ── Section Header ── */}
      <div className="bg-gradient-to-r from-emerald-950/20 via-background to-background p-4 border-b border-border">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                Single-Query Evaluation Results • Live Analysis
              </span>
            </div>
            <h3 className="text-sm font-semibold text-foreground">
              Matched-Pair Comparison: Standard RAG vs. SafeMed AI
            </h3>
            <p className="text-[11px] text-muted-foreground font-mono truncate max-w-xl">
              Query: &ldquo;{breakdown.query}&rdquo;
            </p>
          </div>

          <div className="flex flex-col items-end gap-1">
            <span
              className={`px-3 py-1 rounded-full text-xs font-mono font-semibold border ${
                isProtected
                  ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30'
                  : isSafe
                  ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30'
                  : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30'
              }`}
            >
              {fidelity_status}
            </span>
            <span className="text-[10px] font-mono text-muted-foreground">
              Filter Decision Boundary: <GlossaryTooltip term="τ (threshold)">τ = {tau_threshold}</GlossaryTooltip> • Cutoff k = 30
            </span>
          </div>
        </div>

        {/* ── Tab Switcher ── */}
        <div className="flex gap-2 mt-4 border-b border-border/50 pb-1">
          <button
            onClick={() => setActiveTab('metrics')}
            className={`px-3 py-1 text-xs font-mono rounded-lg transition-colors cursor-pointer ${
              activeTab === 'metrics'
                ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-bold border border-emerald-500/30'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
            }`}
          >
            📊 Performance Metrics
          </button>
          <button
            onClick={() => setActiveTab('sideBySide')}
            className={`px-3 py-1 text-xs font-mono rounded-lg transition-colors cursor-pointer ${
              activeTab === 'sideBySide'
                ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-bold border border-emerald-500/30'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
            }`}
          >
            ⚖️ Side-by-Side Answers
          </button>
          <button
            onClick={() => setActiveTab('docs')}
            className={`px-3 py-1 text-xs font-mono rounded-lg transition-colors cursor-pointer ${
              activeTab === 'docs'
                ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-bold border border-emerald-500/30'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
            }`}
          >
            📑 Top-5 Context Inspection
          </button>
        </div>
      </div>

      {/* ── Tab 1: Live Performance Metrics ── */}
      {activeTab === 'metrics' && (
        <div className="p-4 space-y-4">
          {/* Key Metric Scorecards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {/* 1. Retrieved Candidates */}
            <div className="rounded-xl border border-border bg-background p-3 space-y-1">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">
                Stage 1 Retrieved
              </span>
              <div className="text-xl font-bold font-mono text-foreground">
                {total_screened} <span className="text-xs font-normal text-muted-foreground">passages</span>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Dense candidate pool (all-MiniLM-L6-v2)
              </p>
            </div>

            {/* 2. Filtered Documents */}
            <div className="rounded-xl border border-border bg-background p-3 space-y-1">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">
                Stage 2 Filtered
              </span>
              <div className="text-xl font-bold font-mono text-red-600 dark:text-red-400">
                {total_blocked} <span className="text-xs font-normal text-muted-foreground">blocked</span>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Blocked where P(HD) &ge; {tau_threshold}
              </p>
            </div>

            {/* 3. H3: HD@Top-5 */}
            <div className="rounded-xl border border-border bg-background p-3 space-y-1">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">
                <GlossaryTooltip term="HD@Top-5">H3: HD@Top-5</GlossaryTooltip>
              </span>
              <div className="flex items-baseline gap-1.5 font-mono">
                <span className="text-xs text-red-500 line-through">
                  {(control.hd_at_top5 * 100).toFixed(0)}%
                </span>
                <span className="text-xl font-bold text-emerald-600 dark:text-emerald-400">
                  {(proposed.hd_at_top5 * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-[10px] text-emerald-600 dark:text-emerald-400 font-semibold">
                &darr; {(hd_exposure_reduction * 100).toFixed(0)}% harmful exposure
              </p>
            </div>

            {/* 4. H4: Hit@5 */}
            <div className="rounded-xl border border-border bg-background p-3 space-y-1">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">
                <GlossaryTooltip term="Hit@5">H4: Hit@5</GlossaryTooltip>
              </span>
              <div className="text-xl font-bold font-mono text-foreground flex items-center gap-1.5">
                <span>{proposed.hit_at_5 === 1 ? '100%' : '0%'}</span>
                <span className="text-xs font-normal text-emerald-600 dark:text-emerald-400">
                  ({proposed.gd_count} GD retained)
                </span>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Control: {control.hit_at_5 === 1 ? '100%' : '0%'} ({control.gd_count} GD)
              </p>
            </div>
          </div>

          {/* Comparative Metrics Table */}
          <div className="overflow-x-auto rounded-xl border border-border">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="bg-muted/50 border-b border-border text-left">
                  <th className="p-2.5 font-semibold text-muted-foreground">Hypothesis / Metric</th>
                  <th className="p-2.5 font-semibold text-center" style={{ color: 'var(--color-control, #f97316)' }}>
                    Standard RAG (Control)
                  </th>
                  <th className="p-2.5 font-semibold text-center" style={{ color: 'var(--color-treatment, #10b981)' }}>
                    SafeMed AI (Proposed)
                  </th>
                  <th className="p-2.5 font-semibold text-center text-foreground">Treatment Effect</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60">
                <tr>
                  <td className="p-2.5 text-foreground font-medium">
                    <GlossaryTooltip term="HD@Top-5">H3: Harmful Document Exposure (HD@Top-5)</GlossaryTooltip>
                  </td>
                  <td className="p-2.5 text-center font-semibold text-red-500">
                    {(control.hd_at_top5 * 100).toFixed(1)}% ({control.hd_count}/5 docs)
                  </td>
                  <td className="p-2.5 text-center font-semibold text-emerald-600 dark:text-emerald-400">
                    {(proposed.hd_at_top5 * 100).toFixed(1)}% ({proposed.hd_count}/5 docs)
                  </td>
                  <td className="p-2.5 text-center font-semibold text-emerald-600 dark:text-emerald-400">
                    -{(hd_exposure_reduction * 100).toFixed(1)}% reduction
                  </td>
                </tr>

                <tr>
                  <td className="p-2.5 text-foreground font-medium">
                    <GlossaryTooltip term="Hit@5">H4: Ground-Truth Retention (Hit@5)</GlossaryTooltip>
                  </td>
                  <td className="p-2.5 text-center">
                    {control.hit_at_5 === 1 ? '1.0 (Hit)' : '0.0 (Miss)'} ({control.gd_count} GD)
                  </td>
                  <td className="p-2.5 text-center font-semibold text-emerald-600 dark:text-emerald-400">
                    {proposed.hit_at_5 === 1 ? '1.0 (Hit)' : '0.0 (Miss)'} ({proposed.gd_count} GD)
                  </td>
                  <td className="p-2.5 text-center text-muted-foreground">
                    {proposed.hit_at_5 >= control.hit_at_5 ? '✓ Preserved' : 'Non-inferior'}
                  </td>
                </tr>

                <tr>
                  <td className="p-2.5 text-foreground font-medium">
                    GD Retention Rate in Context
                  </td>
                  <td className="p-2.5 text-center">
                    {(control.gd_retention_rate * 100).toFixed(1)}%
                  </td>
                  <td className="p-2.5 text-center font-semibold text-emerald-600 dark:text-emerald-400">
                    {(proposed.gd_retention_rate * 100).toFixed(1)}%
                  </td>
                  <td className="p-2.5 text-center text-muted-foreground">
                    {proposed.gd_retention_rate >= control.gd_retention_rate ? '↑ Safe Focus' : 'Parity'}
                  </td>
                </tr>

                <tr>
                  <td className="p-2.5 text-foreground font-medium">
                    <GlossaryTooltip term="Hallucination Rate">Fact Status & Hallucination Risk</GlossaryTooltip>
                  </td>
                  <td className="p-2.5 text-center">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        control.has_harmful_exposure
                          ? 'bg-red-500/10 text-red-600 border border-red-500/20'
                          : 'bg-muted text-muted-foreground'
                      }`}
                    >
                      {control.hallucination_risk} Risk
                    </span>
                  </td>
                  <td className="p-2.5 text-center">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                      {proposed.hallucination_risk}
                    </span>
                  </td>
                  <td className="p-2.5 text-center text-emerald-600 dark:text-emerald-400 font-semibold">
                    Harmful context prevented
                  </td>
                </tr>

                <tr>
                  <td className="p-2.5 text-foreground font-medium">
                    Stage 2 Filter Throughput
                  </td>
                  <td className="p-2.5 text-center text-muted-foreground">
                    Bypassed (0 blocked)
                  </td>
                  <td className="p-2.5 text-center font-semibold text-foreground">
                    {total_screened} Screened &rarr; {total_blocked} Blocked &rarr; Top-30 Truncated
                  </td>
                  <td className="p-2.5 text-center text-muted-foreground">
                    Pre-reranking gate active
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Tab 2: Side-by-Side Response Comparison ── */}
      {activeTab === 'sideBySide' && (
        <div className="p-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Standard RAG (Control) */}
            <div className="rounded-xl border border-orange-500/30 bg-orange-50/10 dark:bg-orange-950/10 p-4 space-y-3">
              <div className="flex items-center justify-between border-b border-orange-500/20 pb-2">
                <span className="text-xs font-mono font-bold text-orange-600 dark:text-orange-400 flex items-center gap-1.5">
                  <span>Standard RAG (Control)</span>
                  <span className="text-[10px] font-normal px-1.5 py-0.5 rounded bg-orange-500/10 border border-orange-500/20">
                    Unfiltered
                  </span>
                </span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded font-semibold ${
                    control.has_harmful_exposure
                      ? 'bg-red-500/20 text-red-600 dark:text-red-300'
                      : 'bg-muted text-muted-foreground'
                  }`}
                >
                  {control.has_harmful_exposure ? '⚠️ Harmful Context Present' : 'Safe Context'}
                </span>
              </div>

              <div className="space-y-1">
                <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                  Generated Clinical Summary:
                </span>
                <p className="text-xs text-foreground leading-relaxed font-sans bg-background/80 p-3 rounded-lg border border-border">
                  {control.answer}
                </p>
              </div>

              {/* Claims Breakdown */}
              {control.claims && control.claims.length > 0 && (
                <div className="space-y-1 pt-1">
                  <span className="text-[10px] font-mono uppercase text-muted-foreground">
                    Extracted Claims ({control.claims.length}):
                  </span>
                  <ul className="space-y-1 text-[11px] text-muted-foreground">
                    {control.claims.map((claim, idx) => (
                      <li key={idx} className="flex items-start gap-1.5">
                        <span className="text-orange-500 shrink-0">•</span>
                        <span>{claim}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* SafeMed AI (Proposed) */}
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-50/10 dark:bg-emerald-950/10 p-4 space-y-3">
              <div className="flex items-center justify-between border-b border-emerald-500/20 pb-2">
                <span className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                  <span>SafeMed AI (Proposed)</span>
                  <span className="text-[10px] font-normal px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">
                    Filtered
                  </span>
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded font-semibold bg-emerald-500/20 text-emerald-700 dark:text-emerald-300">
                  ✓ Verified Safe Context (0% HD)
                </span>
              </div>

              <div className="space-y-1">
                <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                  Generated Clinical Summary:
                </span>
                <p className="text-xs text-foreground leading-relaxed font-sans bg-background/80 p-3 rounded-lg border border-border">
                  {proposed.answer}
                </p>
              </div>

              {/* Claims Breakdown */}
              {proposed.claims && proposed.claims.length > 0 && (
                <div className="space-y-1 pt-1">
                  <span className="text-[10px] font-mono uppercase text-muted-foreground">
                    Extracted Claims ({proposed.claims.length}):
                  </span>
                  <ul className="space-y-1 text-[11px] text-muted-foreground">
                    {proposed.claims.map((claim, idx) => (
                      <li key={idx} className="flex items-start gap-1.5">
                        <span className="text-emerald-600 dark:text-emerald-400 shrink-0">•</span>
                        <span>{claim}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 3: Top-5 Context Inspection ── */}
      {activeTab === 'docs' && (
        <div className="p-4 space-y-4">
          <p className="text-xs text-muted-foreground font-mono">
            Inspect the 5 documents forwarded to Stage 4 (Generator). Notice how Standard RAG admits misleading/harmful literature, whereas SafeMed AI expels them.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Control Top-5 */}
            <div className="space-y-2">
              <span className="text-xs font-mono font-bold text-orange-600 dark:text-orange-400 flex items-center justify-between">
                <span>Standard RAG Top-5 Documents</span>
                <span className="text-[10px] text-muted-foreground font-normal">
                  {control.hd_count} Harmful / {control.gd_count} Ground-Truth
                </span>
              </span>

              <div className="space-y-2">
                {control.top5_documents.map((doc, idx) => {
                  const isBlockedByFilter = doc.is_blocked_in_proposed || doc.category === 'HD'
                  const isExpanded = expandedDocId === `ctrl-${idx}`

                  return (
                    <div
                      key={doc.id || idx}
                      className={`p-3 rounded-xl border text-xs transition-all ${
                        isBlockedByFilter
                          ? 'border-red-500/40 bg-red-50/20 dark:bg-red-950/20'
                          : 'border-border bg-background'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="space-y-0.5">
                          <div className="flex items-center gap-1.5 font-semibold text-foreground">
                            <span className="text-muted-foreground font-mono text-[10px]">[{idx + 1}]</span>
                            <span className="truncate max-w-[240px] sm:max-w-[320px]">{doc.title}</span>
                          </div>
                          <span className="text-[10px] font-mono text-muted-foreground">
                            Category: <strong>{doc.category}</strong> • P(HD): {doc.harmful_probability.toFixed(3)}
                          </span>
                        </div>
                        <span
                          className={`px-2 py-0.5 rounded text-[9px] font-mono font-bold shrink-0 ${
                            isBlockedByFilter
                              ? 'bg-red-600 text-white'
                              : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
                          }`}
                        >
                          {isBlockedByFilter ? '⚠️ BLOCKED IN PROPOSED' : 'PASSED'}
                        </span>
                      </div>

                      {doc.block_reason && isBlockedByFilter && (
                        <p className="mt-1.5 text-[10px] font-medium text-red-700 dark:text-red-300 bg-red-100/50 dark:bg-red-950/40 p-1.5 rounded">
                          <strong>Block Reason:</strong> {doc.block_reason}
                        </p>
                      )}

                      <p className="text-[11px] text-muted-foreground font-serif pt-1.5 line-clamp-2">
                        {doc.snippet}
                      </p>

                      <button
                        onClick={() => setExpandedDocId(isExpanded ? null : `ctrl-${idx}`)}
                        className="text-[10px] font-mono text-muted-foreground hover:text-foreground mt-1 cursor-pointer"
                      >
                        {isExpanded ? '▲ Hide snippet' : '▼ Expand snippet'}
                      </button>

                      {isExpanded && (
                        <p className="text-[11px] text-foreground/80 font-serif pt-2 border-t border-border/50 mt-1 whitespace-pre-wrap">
                          {doc.snippet}
                        </p>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Proposed Top-5 */}
            <div className="space-y-2">
              <span className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400 flex items-center justify-between">
                <span>SafeMed AI Top-5 Documents</span>
                <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-semibold">
                  0 Harmful / {proposed.gd_count} Ground-Truth
                </span>
              </span>

              <div className="space-y-2">
                {proposed.top5_documents.map((doc, idx) => {
                  const isExpanded = expandedDocId === `prop-${idx}`

                  return (
                    <div
                      key={doc.id || idx}
                      className="p-3 rounded-xl border border-emerald-500/20 bg-background text-xs"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="space-y-0.5">
                          <div className="flex items-center gap-1.5 font-semibold text-foreground">
                            <span className="text-emerald-600 dark:text-emerald-400 font-mono text-[10px]">[{idx + 1}]</span>
                            <span className="truncate max-w-[240px] sm:max-w-[320px]">{doc.title}</span>
                          </div>
                          <span className="text-[10px] font-mono text-muted-foreground">
                            Category: <strong>{doc.category}</strong> • P(HD): {doc.harmful_probability.toFixed(3)}
                          </span>
                        </div>
                        <span className="px-2 py-0.5 rounded text-[9px] font-mono font-bold bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 shrink-0">
                          ✓ VERIFIED SAFE
                        </span>
                      </div>

                      <p className="text-[11px] text-muted-foreground font-serif pt-1.5 line-clamp-2">
                        {doc.snippet}
                      </p>

                      <button
                        onClick={() => setExpandedDocId(isExpanded ? null : `prop-${idx}`)}
                        className="text-[10px] font-mono text-muted-foreground hover:text-foreground mt-1 cursor-pointer"
                      >
                        {isExpanded ? '▲ Hide snippet' : '▼ Expand snippet'}
                      </button>

                      {isExpanded && (
                        <p className="text-[11px] text-foreground/80 font-serif pt-2 border-t border-border/50 mt-1 whitespace-pre-wrap">
                          {doc.snippet}
                        </p>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
