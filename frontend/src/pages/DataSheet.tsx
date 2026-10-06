import React, { useState, useMemo } from 'react'
import DemoTag from '../components/DemoTag'
import LabelChip from '../components/LabelChip'
import { buildEvaluationSummary } from '../lib/fixtures'
import type { Label } from '../lib/schema'

const summary = buildEvaluationSummary()
const rows = summary.perQueryRows

export default function DataSheet() {
  const [configFilter, setConfigFilter] = useState<'all' | 'control' | 'treatment'>('all')
  const [hdFilter, setHdFilter] = useState(false)
  const [hitFilter, setHitFilter] = useState(false)
  const [search, setSearch] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)

  const filtered = useMemo(() => {
    return rows.filter((r) => {
      if (configFilter !== 'all' && r.config !== configFilter) return false
      if (hdFilter && r.her <= 0) return false
      if (hitFilter && r.gtrr >= 1) return false
      if (search && !r.queryId.toLowerCase().includes(search.toLowerCase()) && !r.answer.toLowerCase().includes(search.toLowerCase())) return false
      return true
    })
  }, [configFilter, hdFilter, hitFilter, search])

  const downloadCSV = () => {
    const header = 'queryId,config,docIds,hd_at_top5,hit_at_5,factPrecision,factRecall,factF1,factHallucinationRate,answer'
    const csvRows = filtered.map((r) =>
      [
        r.queryId, r.config,
        r.docIds.join(';'),
        r.her, r.gtrr,
        r.factPrecision, r.factRecall, r.factF1, r.factHallucinationRate,
        `"${r.answer.replace(/"/g, '""')}"`
      ].join(',')
    )
    const blob = new Blob([header + '\n' + csvRows.join('\n')], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'ILLUSTRATIVE-data-collection-sheet.csv'
    a.click()
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center gap-3">
        <h1 className="font-serif text-2xl font-semibold">Data Sheet</h1>
        <DemoTag />
      </div>

      <p className="text-sm text-muted-foreground">
        One row per query per configuration. Mirrors the thesis Data Collection Sheet. All data is illustrative demo data.
      </p>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-0.5 rounded border border-border overflow-hidden">
          {(['all', 'control', 'treatment'] as const).map((c) => (
            <button
              key={c}
              onClick={() => setConfigFilter(c)}
              className={`px-3 py-1.5 text-xs font-mono transition-colors ${configFilter === c ? 'bg-foreground text-background font-semibold' : 'text-muted-foreground hover:bg-muted'}`}
            >
              {c === 'all' ? 'All' : c === 'control' ? 'Standard RAG' : 'Proposed'}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1.5 text-xs font-mono text-muted-foreground cursor-pointer">
          <input type="checkbox" checked={hdFilter} onChange={(e) => setHdFilter(e.target.checked)} className="accent-foreground" />
          HD@Top-5 &gt; 0
        </label>
        <label className="flex items-center gap-1.5 text-xs font-mono text-muted-foreground cursor-pointer">
          <input type="checkbox" checked={hitFilter} onChange={(e) => setHitFilter(e.target.checked)} className="accent-foreground" />
          Hit@5 &lt; 1
        </label>
        <input
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search query ID…"
          className="rounded border border-border bg-background text-xs font-mono px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-ring"
        />
        <button
          onClick={downloadCSV}
          className="ml-auto text-xs font-mono px-3 py-1.5 rounded border border-border hover:bg-muted transition-colors"
        >
          Download CSV (ILLUSTRATIVE)
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded border border-border">
        <table className="text-xs font-mono border-collapse w-full min-w-[900px]">
          <thead>
            <tr className="bg-muted sticky top-0">
              <th className="text-left p-2 border border-border">Query ID</th>
              <th className="p-2 border border-border">Config</th>
              <th className="p-2 border border-border">Top-5 docs</th>
              <th className="p-2 border border-border">HD@Top-5</th>
              <th className="p-2 border border-border">Hit@5</th>
              <th className="p-2 border border-border">Fact Prec.</th>
              <th className="p-2 border border-border">Fact Rec.</th>
              <th className="p-2 border border-border">Fact F1</th>
              <th className="p-2 border border-border">Hall. Rate</th>
              <th className="p-2 border border-border">Answer snippet</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row, i) => {
              const key = `${row.queryId}-${row.config}`
              const isExpanded = expanded === key
              return (
                <React.Fragment key={key}>
                  <tr
                    className={`hover:bg-muted/40 cursor-pointer ${i % 2 === 0 ? '' : 'bg-muted/20'}`}
                    onClick={() => setExpanded(isExpanded ? null : key)}
                  >
                    <td className="p-2 border border-border">{row.queryId}</td>
                    <td className="p-2 border border-border">
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
                        style={{
                          color: row.config === 'control' ? 'var(--color-control)' : 'var(--color-treatment)',
                          background: row.config === 'control' ? 'var(--color-control-bg)' : 'var(--color-treatment-bg)',
                        }}
                      >
                        {row.config === 'control' ? 'Control' : 'Treatment'}
                      </span>
                    </td>
                    <td className="p-2 border border-border">
                      <div className="flex gap-0.5 flex-wrap">
                        {row.docIds.map((id) => (
                          <span key={id} className="text-[9px] bg-muted rounded px-1">{id.replace('DEMO-', '')}</span>
                        ))}
                      </div>
                    </td>
                    <td className="p-2 border border-border text-center tabular-nums">{row.her.toFixed(2)}</td>
                    <td className="p-2 border border-border text-center tabular-nums">{row.gtrr.toFixed(2)}</td>
                    <td className="p-2 border border-border text-center tabular-nums">{row.factPrecision.toFixed(3)}</td>
                    <td className="p-2 border border-border text-center tabular-nums">{row.factRecall.toFixed(3)}</td>
                    <td className="p-2 border border-border text-center tabular-nums">{row.factF1.toFixed(3)}</td>
                    <td className="p-2 border border-border text-center tabular-nums">{row.factHallucinationRate.toFixed(3)}</td>
                    <td className="p-2 border border-border max-w-[200px] truncate text-muted-foreground">{row.answer.slice(0, 60)}…</td>
                  </tr>
                  {isExpanded && (
                    <tr>
                      <td colSpan={10} className="p-3 border border-border bg-muted/20">
                        <div className="space-y-2">
                          <div className="font-semibold">Answer:</div>
                          <p className="text-xs leading-relaxed text-muted-foreground">{row.answer}</p>
                          <div className="font-semibold">Claims:</div>
                          <div className="space-y-1">
                            {row.claims.map((c, ci) => (
                              <div key={ci} className="flex items-start gap-2 text-xs">
                                <span className={`shrink-0 ${c.verdict === 'supported' ? 'text-[var(--color-gd)]' : 'text-[var(--color-hd)]'}`}>
                                  {c.verdict === 'supported' ? '✓' : '✗'}
                                </span>
                                <span>{c.text}</span>
                              </div>
                            ))}
                          </div>
                          <div className="flex gap-2">
                            {row.labels.map((l, li) => l ? <LabelChip key={li} label={l as Label} size="sm" /> : <span key={li} className="text-[9px] text-muted-foreground">?</span>)}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="text-[10px] font-mono text-muted-foreground">
        {filtered.length} of {rows.length} rows shown
      </div>
    </div>
  )
}
