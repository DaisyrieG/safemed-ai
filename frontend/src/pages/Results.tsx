import { useState, useEffect } from 'react'
import GlossaryTooltip from '../components/GlossaryTooltip'
import { fetchEvaluationResults } from '../lib/api'
import type { EvaluationResults, WilcoxonTestResult } from '../lib/api'

// ---------------------------------------------------------------------------
// Safe Formatters (guaranteed never to crash on null / undefined)
// ---------------------------------------------------------------------------

function pBadge(significant?: boolean) {
  return significant
    ? 'text-emerald-700 dark:text-emerald-300 border-emerald-500/30 bg-emerald-50/60 dark:bg-emerald-950/40'
    : 'text-muted-foreground border-border bg-muted'
}

function fmt(n?: number | null, digits = 4) {
  if (n === undefined || n === null || isNaN(n)) return '—'
  return n.toFixed(digits)
}

function pct(n?: number | null) {
  if (n === undefined || n === null || isNaN(n)) return '—'
  return `${(n * 100).toFixed(1)}%`
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Compact metric row: label | control value | arrow | treatment value */
function RateRow({
  label,
  ctrl,
  treat,
  format = pct,
  lowerBetter,
}: {
  label: React.ReactNode
  ctrl?: number | null
  treat?: number | null
  format?: (n?: number | null) => string
  lowerBetter: boolean
}) {
  const c = ctrl ?? 0
  const t = treat ?? 0
  const improvement = lowerBetter ? t < c : t > c
  const arrow = t === c ? '=' : t < c ? '↓' : '↑'
  const arrowColor =
    t === c
      ? 'text-muted-foreground'
      : improvement
      ? 'text-emerald-600 dark:text-emerald-400 font-bold'
      : 'text-red-600 dark:text-red-400 font-bold'

  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-4 items-center py-1.5 border-b border-border/40 last:border-0 text-xs font-mono">
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums" style={{ color: 'var(--color-control, #d97706)' }}>
        {format(ctrl)}
      </span>
      <span className={`tabular-nums ${arrowColor}`}>
        {arrow}
      </span>
      <span className="tabular-nums font-semibold" style={{ color: 'var(--color-treatment, #059669)' }}>
        {format(treat)}
      </span>
    </div>
  )
}

/** Wilcoxon Signed-Rank Test Statistics Block */
function WilcoxonStatsDisplay({
  result,
  alpha = 0.05,
}: {
  result?: WilcoxonTestResult | null
  alpha?: number
}) {
  if (!result) {
    return (
      <div className="text-xs font-mono text-muted-foreground border-t border-border/40 pt-2 mt-2 italic">
        Inferential test data pending full test execution.
      </div>
    )
  }

  const isSig = Boolean(result.significant || (result.p_value !== undefined && result.p_value < alpha))

  return (
    <div className="space-y-1.5 text-xs font-mono border-t border-border/40 pt-2.5 mt-2">
      <div className="flex justify-between items-center">
        <span className="text-muted-foreground">Inferential Test</span>
        <span className="text-right text-[11px] font-semibold text-foreground">
          {result.test_type || 'Wilcoxon Signed-Rank Test'}
        </span>
      </div>

      <div className="flex justify-between items-center">
        <span className="text-muted-foreground">Test Statistic (W)</span>
        <span className="tabular-nums font-semibold">{fmt(result.statistic, 2)}</span>
      </div>

      <div className="flex justify-between items-center">
        <span className="text-muted-foreground">p-value</span>
        <span className={`tabular-nums font-bold ${isSig ? 'text-emerald-600 dark:text-emerald-400' : 'text-muted-foreground'}`}>
          {fmt(result.p_value, 6)}
          {result.p_value_holm !== undefined && (
            <span className="text-[10px] text-muted-foreground font-normal ml-1">
              (Holm: {fmt(result.p_value_holm, 4)})
            </span>
          )}
        </span>
      </div>

      {result.mean_diff !== undefined && (
        <div className="flex justify-between items-center">
          <span className="text-muted-foreground">Mean Shift (D̄)</span>
          <span className="tabular-nums">{fmt(result.mean_diff)}</span>
        </div>
      )}

      {result.median_diff !== undefined && (
        <div className="flex justify-between items-center">
          <span className="text-muted-foreground">Median Shift (D̃)</span>
          <span className="tabular-nums">{fmt(result.median_diff)}</span>
        </div>
      )}

      {result.rank_biserial !== undefined && (
        <div className="flex justify-between items-center">
          <GlossaryTooltip term="Effect Size">
            <span className="text-muted-foreground">Rank-Biserial (r_rb)</span>
          </GlossaryTooltip>
          <span className="tabular-nums font-medium">{fmt(result.rank_biserial)}</span>
        </div>
      )}

      {result.ci_95 && Array.isArray(result.ci_95) && result.ci_95.length === 2 && (
        <div className="flex justify-between items-center">
          <span className="text-muted-foreground">95% Bootstrap CI</span>
          <span className="tabular-nums">
            [{fmt(result.ci_95[0])}, {fmt(result.ci_95[1])}]
          </span>
        </div>
      )}

      <div className={`text-[10px] mt-2 px-2.5 py-1.5 rounded-lg border font-medium ${pBadge(isSig)}`}>
        {isSig
          ? `✓ Reject H₀ at α = ${alpha} — Statistically significant effect.`
          : `Fail to reject H₀ at α = ${alpha} — No significant difference detected.`}
      </div>
    </div>
  )
}

/** Hypothesis Card Wrapper */
function HCard({
  id,
  title,
  subtitle,
  children,
  significant,
}: {
  id: string
  title: string
  subtitle: string
  children: React.ReactNode
  significant?: boolean
}) {
  return (
    <section className="rounded-2xl border border-border bg-card p-5 space-y-3.5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400 font-bold uppercase tracking-widest mb-0.5">
            {id}
          </div>
          <h2 className="font-serif text-base font-semibold leading-tight text-foreground">
            {title}
          </h2>
          <div className="text-[11px] text-muted-foreground font-mono mt-0.5">
            {subtitle}
          </div>
        </div>
        <span
          className={`shrink-0 text-[10px] font-mono font-semibold px-2.5 py-1 rounded-full border ${
            significant
              ? 'text-emerald-700 dark:text-emerald-300 border-emerald-500/40 bg-emerald-500/10'
              : 'text-muted-foreground border-border bg-muted'
          }`}
        >
          {significant ? 'Reject H₀' : 'Fail to reject H₀'}
        </span>
      </div>

      {/* Column Headers */}
      <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-4 text-[10px] font-mono text-muted-foreground uppercase tracking-wider pb-1 border-b border-border/40">
        <span>Metric</span>
        <span style={{ color: 'var(--color-control, #d97706)' }}>Control (Vanilla)</span>
        <span></span>
        <span style={{ color: 'var(--color-treatment, #059669)' }}>SafeMed AI</span>
      </div>

      {children}
    </section>
  )
}

/** Skeleton Loader */
function SkeletonCard() {
  return (
    <div className="rounded-2xl border border-border bg-card p-5 space-y-3 animate-pulse shadow-sm">
      <div className="h-3 bg-muted rounded w-1/4" />
      <div className="h-5 bg-muted rounded w-2/3" />
      <div className="space-y-2 mt-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-3 bg-muted rounded" />
        ))}
      </div>
    </div>
  )
}

/** Detailed Per-Query Accordion Row */
function QueryDetailRow({ qd }: { qd: EvaluationResults['query_details'][number] }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden shadow-2xs">
      <button
        className="w-full flex items-center justify-between px-4 py-3.5 text-left hover:bg-muted/40 transition-colors cursor-pointer"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <div className="flex items-center gap-2 pr-2">
          <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-muted text-muted-foreground font-semibold">
            {qd.query_id}
          </span>
          <span className="text-xs sm:text-sm font-medium text-foreground line-clamp-1">
            {qd.query}
          </span>
        </div>
        <span className="font-mono text-xs text-muted-foreground ml-3 shrink-0 font-bold">
          {open ? '▲' : '▼'}
        </span>
      </button>

      {open && (
        <div className="border-t border-border px-4 py-4 space-y-4 bg-muted/10">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Control Column */}
            <div className="space-y-2.5 p-3.5 rounded-xl border border-amber-500/20 bg-card">
              <div className="font-mono text-[11px] font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                Standard RAG (Control)
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="rounded-lg border border-border bg-background p-2 text-center">
                  <div className="text-[10px] text-muted-foreground">HD@Top-5</div>
                  <div className="font-bold tabular-nums text-red-600 dark:text-red-400">
                    {pct(qd.control?.her)}
                  </div>
                </div>
                <div className="rounded-lg border border-border bg-background p-2 text-center">
                  <div className="text-[10px] text-muted-foreground">GD Retention</div>
                  <div className="font-bold tabular-nums">{pct(qd.control?.gtrr)}</div>
                </div>
              </div>
              <div className="text-xs text-muted-foreground rounded-lg bg-muted/40 p-2.5 font-serif leading-relaxed line-clamp-4">
                &ldquo;{qd.control?.answer}&rdquo;
              </div>
            </div>

            {/* Treatment Column */}
            <div className="space-y-2.5 p-3.5 rounded-xl border border-emerald-500/30 bg-card">
              <div className="font-mono text-[11px] font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                SafeMed AI (Treatment)
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="rounded-lg border border-border bg-background p-2 text-center">
                  <div className="text-[10px] text-muted-foreground">HD@Top-5</div>
                  <div className="font-bold tabular-nums text-emerald-600 dark:text-emerald-400">
                    {pct(qd.treatment?.her)}
                  </div>
                </div>
                <div className="rounded-lg border border-border bg-background p-2 text-center">
                  <div className="text-[10px] text-muted-foreground">GD Retention</div>
                  <div className="font-bold tabular-nums">{pct(qd.treatment?.gtrr)}</div>
                </div>
              </div>
              <div className="text-xs text-foreground/90 rounded-lg bg-emerald-50/20 dark:bg-emerald-950/20 border border-emerald-500/20 p-2.5 font-serif leading-relaxed line-clamp-4">
                &ldquo;{qd.treatment?.answer}&rdquo;
              </div>
            </div>
          </div>

          <div className="rounded-lg bg-muted/40 border border-border/60 px-3.5 py-2.5 text-xs text-muted-foreground">
            <strong className="text-foreground font-semibold">Reference Ground-Truth: </strong>
            <span>{qd.reference_answer}</span>
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Results Page
// ---------------------------------------------------------------------------

export default function Results() {
  const [data, setData] = useState<EvaluationResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    fetchEvaluationResults()
      .then((d) => {
        setData(d)
        setError(null)
      })
      .catch((e: Error) => {
        setError(e.message || 'Failed to load evaluation results.')
      })
      .finally(() => setLoading(false))
  }, [])

  // Safely extract hypothesis test objects
  const stats = data?.Inferential_Statistics || {}
  const testH1 = stats.H1_HER_Reduction || stats.HD_at_Top5
  const testH2 = stats.H2_GTRR_Retention || stats.Hit_at_5
  const testH3 = stats.H3_Pre_vs_Final_HER
  const testH4 = stats.H4_Hallucination_Rate || stats.Hallucination_Rate

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* ── Page Header ── */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <h1 className="font-serif text-3xl font-bold tracking-tight text-foreground">
            Thesis Benchmark Evaluation (PubMedQA)
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground font-mono mt-1">
            Standardized Held-Out Test Split • Matched-Pairs Comparative Analysis
          </p>
        </div>

        <div className="flex items-center gap-2">
          {!loading && !error && data && (
            <span className="inline-flex items-center gap-1.5 text-xs font-mono font-semibold px-3 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              Pilot Test Split • {data.num_queries} Queries Evaluated
            </span>
          )}

          {!loading && error && (
            <span className="inline-flex items-center gap-1.5 text-xs font-mono font-semibold px-3 py-1 rounded-full border border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400">
              ⚠️ Evaluation Cache Missing
            </span>
          )}
        </div>
      </div>

      {/* ── Error Banner ── */}
      {error && (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-5 space-y-2">
          <div className="font-mono text-sm font-bold text-amber-800 dark:text-amber-300">
            ⚠️ Unable to load evaluation results file
          </div>
          <p className="text-xs text-muted-foreground">
            {error}
          </p>
          <div className="font-mono text-xs text-foreground bg-background/80 p-3 rounded-xl border border-border mt-2 space-y-1">
            <p className="text-muted-foreground">To generate evaluation data, run in your terminal:</p>
            <code className="text-emerald-600 dark:text-emerald-400 block font-bold">
              python experiments/evaluation/run_evaluation.py --max-queries 10
            </code>
          </div>
        </div>
      )}

      {/* ── Loading Skeleton ── */}
      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {[1, 2, 3, 4].map((i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      )}

      {/* ── Main Results Content ── */}
      {!loading && !error && data && (
        <>
          {/* Architecture Callout Banner */}
          <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-5 space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span className="font-mono text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                Methodology: 4-Stage Pre-Reranking Filter Architecture
              </span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Standard RAG fails because cross-encoder rerankers elevate topically similar yet contradictory documents.
              SafeMed AI intercepts candidates at <strong>Stage 2 (Pre-Reranking)</strong> via a fine-tuned Cross-Encoder,
              dropping harmful literature before Stage 3 selects the final Top-5 context.
              All inferential statistics are evaluated using <strong>Wilcoxon Signed-Rank Tests</strong> with <strong>Pratt's zero-method</strong>.
            </p>
          </div>

          {/* ── 4 Primary Thesis Hypothesis Cards ── */}
          <div className="space-y-3">
            <h2 className="font-serif text-xl font-bold text-foreground">
              Primary Hypothesis Tests (H1 – H4)
            </h2>
            <p className="text-xs font-mono text-muted-foreground">
              Evaluated at α = 0.05 (two-tailed) on within-query matched pairs.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* ── H1: Harmful Exposure Reduction ── */}
              <HCard
                id="H1 (SOP 1)"
                title="Harmful Document Exposure Reduction"
                subtitle="Paired Wilcoxon test on Control vs. SafeMed AI HD@Top-5"
                significant={testH1?.significant}
              >
                <RateRow
                  label="Mean HD@Top-5 Exposure"
                  ctrl={data.HD_Exposure_Top5_Control}
                  treat={data.HD_Exposure_Top5_Treatment}
                  lowerBetter
                />
                {data.HER_Reduction !== undefined && (
                  <div className="flex justify-between text-xs font-mono py-1 border-b border-border/40">
                    <span className="text-muted-foreground">Net HER Reduction</span>
                    <span className="text-emerald-600 dark:text-emerald-400 font-bold tabular-nums">
                      {pct(data.HER_Reduction)}
                    </span>
                  </div>
                )}
                <WilcoxonStatsDisplay result={testH1} />
              </HCard>

              {/* ── H2: Ground-Truth Retention ── */}
              <HCard
                id="H2 (SOP 2)"
                title="Ground-Truth Retention Rate (Hit@5)"
                subtitle="One-sample Wilcoxon test against non-inferiority margin"
                significant={testH2?.significant}
              >
                <RateRow
                  label="GD Retention Rate (Hit@5)"
                  ctrl={data.GD_Retention_Top5_Control}
                  treat={data.GD_Retention_Top5_Treatment}
                  lowerBetter={false}
                />
                <WilcoxonStatsDisplay result={testH2} />
              </HCard>

              {/* ── H3: Pre-Filter vs Final Exposure ── */}
              <HCard
                id="H3 (SOP 3)"
                title="Pre-Filter vs. Final Context Exposure"
                subtitle="Paired Wilcoxon test: Retrieved pool density vs. Final Top-5"
                significant={testH3?.significant}
              >
                <RateRow
                  label="Initial vs. Final HD"
                  ctrl={data.Mean_Pre_Filter_Harmful_Density ?? data.HD_Exposure_Top5_Control}
                  treat={data.HD_Exposure_Top5_Treatment}
                  lowerBetter
                />
                <WilcoxonStatsDisplay result={testH3} />
              </HCard>

              {/* ── H4: Hallucination Suppression ── */}
              <HCard
                id="H4 (SOP 4)"
                title="Hallucination & Veracity Assurance"
                subtitle="Wilcoxon test on fact-level ungrounded claim occurrence"
                significant={testH4?.significant}
              >
                <RateRow
                  label="Hallucination Rate (HR)"
                  ctrl={data.Fact_Level_Metrics?.control?.hr}
                  treat={data.Fact_Level_Metrics?.treatment?.hr}
                  lowerBetter
                />
                <RateRow
                  label="Harmful-Induced Hallucination (HDIHR)"
                  ctrl={data.Fact_Level_Metrics?.control?.hdihr}
                  treat={data.Fact_Level_Metrics?.treatment?.hdihr}
                  lowerBetter
                />
                <WilcoxonStatsDisplay result={testH4} />
              </HCard>
            </div>
          </div>

          {/* ── Descriptive Statistics Summary Table ── */}
          <section className="space-y-3">
            <h2 className="font-serif text-lg font-bold text-foreground">
              Descriptive Statistics Summary
            </h2>
            <div className="overflow-x-auto rounded-2xl border border-border bg-card shadow-xs">
              <table className="text-xs font-mono border-collapse w-full min-w-[540px]">
                <thead>
                  <tr className="bg-muted/50 border-b border-border text-muted-foreground uppercase text-[10px]">
                    <th className="text-left p-3 font-semibold">Evaluation Metric</th>
                    <th className="p-3 font-semibold text-center" style={{ color: 'var(--color-control, #d97706)' }}>
                      Control (Standard RAG)
                    </th>
                    <th className="p-3 font-semibold text-center" style={{ color: 'var(--color-treatment, #059669)' }}>
                      Proposed (SafeMed AI)
                    </th>
                    <th className="p-3 font-semibold text-center">Net Difference (Δ)</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    {
                      name: 'HD@Top-5 (Harmful Exposure)',
                      c: data.HD_Exposure_Top5_Control,
                      t: data.HD_Exposure_Top5_Treatment,
                      low: true,
                    },
                    {
                      name: 'Hit@5 (Ground-Truth Retention)',
                      c: data.GD_Retention_Top5_Control,
                      t: data.GD_Retention_Top5_Treatment,
                      low: false,
                    },
                    {
                      name: 'Hallucination Rate (HR)',
                      c: data.Fact_Level_Metrics?.control?.hr,
                      t: data.Fact_Level_Metrics?.treatment?.hr,
                      low: true,
                    },
                    {
                      name: 'Harmful-Induced Hallucination (HDIHR)',
                      c: data.Fact_Level_Metrics?.control?.hdihr,
                      t: data.Fact_Level_Metrics?.treatment?.hdihr,
                      low: true,
                    },
                    {
                      name: 'Answer Precision (Factual Accuracy)',
                      c: data.Fact_Level_Metrics?.control?.precision,
                      t: data.Fact_Level_Metrics?.treatment?.precision,
                      low: false,
                    },
                    {
                      name: 'Fact Supporting F1',
                      c: data.Fact_Level_Metrics?.control?.f1,
                      t: data.Fact_Level_Metrics?.treatment?.f1,
                      low: false,
                    },
                  ].map(({ name, c, t, low }) => {
                    const ctrlVal = c ?? 0
                    const treatVal = t ?? 0
                    const delta = treatVal - ctrlVal
                    const isGood = low ? delta < 0 : delta > 0
                    const deltaColor =
                      delta === 0
                        ? 'text-muted-foreground'
                        : isGood
                        ? 'text-emerald-600 dark:text-emerald-400 font-bold'
                        : 'text-red-600 dark:text-red-400 font-bold'

                    return (
                      <tr key={name} className="border-b border-border/40 last:border-0 hover:bg-muted/30 transition-colors">
                        <td className="p-3 font-medium text-foreground">{name}</td>
                        <td className="p-3 text-center tabular-nums">{pct(c)}</td>
                        <td className="p-3 text-center tabular-nums font-semibold">{pct(t)}</td>
                        <td className={`p-3 text-center tabular-nums ${deltaColor}`}>
                          {delta > 0 ? '+' : ''}{pct(delta)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {/* ── Stage 2 Safety Filter Performance Metrics ── */}
          {data.Filter_Metrics && (
            <section className="rounded-2xl border border-border bg-card p-5 space-y-3 shadow-xs">
              <h2 className="font-serif text-lg font-bold text-foreground">
                Stage 2 Safety Filter Classification Performance
              </h2>
              <p className="text-xs text-muted-foreground font-mono">
                Evaluates cross-encoder capability in isolating harmful documents (Positive Class = Harmful Document).
              </p>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
                {[
                  { label: 'TPR / Recall (Sensitivity)', val: data.Filter_Metrics.TPR_Recall },
                  { label: 'FPR (Fallout Rate)', val: data.Filter_Metrics.FPR },
                  { label: 'Precision', val: data.Filter_Metrics.Precision },
                  { label: 'F1 Score', val: data.Filter_Metrics.F1_score },
                ].map(({ label, val }) => (
                  <div key={label} className="p-3 rounded-xl border border-border bg-muted/20 text-center">
                    <span className="text-[10px] font-mono text-muted-foreground uppercase block mb-1">
                      {label}
                    </span>
                    <span className="text-base font-mono font-bold text-emerald-600 dark:text-emerald-400 tabular-nums">
                      {fmt(val, 4)}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* ── Per-Query Detailed Inspection ── */}
          {data.query_details && data.query_details.length > 0 && (
            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-serif text-lg font-bold text-foreground">
                  Benchmark Test Query Traces (Held-Out Split)
                </h2>
                <span className="text-xs font-mono text-muted-foreground">
                  {data.query_details.length} Benchmark Queries
                </span>
              </div>
              <p className="text-xs text-muted-foreground font-mono">
                Click any benchmark query below to inspect matched-pair document exposures, generated consensus, and ground-truth references.
              </p>

              <div className="space-y-2">
                {data.query_details.map((qd) => (
                  <QueryDetailRow key={qd.query_id} qd={qd} />
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  )
}
