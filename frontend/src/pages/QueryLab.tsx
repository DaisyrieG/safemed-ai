import { useState, useEffect } from 'react'
import {
  executeClinicalQuery,
  fetchSampleCases,
  fetchHealth,
  type ClinicalResponse,
  type SampleCase,
  type SystemHealth,
} from '../lib/api'
import SingleQueryResults from '../components/SingleQueryResults'
import DocumentDetailModal, { type DocumentDetail } from '../components/DocumentDetailModal'

export default function QueryLab() {
  const [inputQuery, setInputQuery] = useState('')
  const [activeQuery, setActiveQuery] = useState('')
  const [sampleCases, setSampleCases] = useState<SampleCase[]>([])
  const [backendHealth, setBackendHealth] = useState<SystemHealth | null>(null)
  const [backendError, setBackendError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeStage, setActiveStage] = useState<number>(0)
  const [activeTab, setActiveTab] = useState<'sources' | 'safety' | 'claims' | 'evaluation'>('sources')
  const [currentResponse, setCurrentResponse] = useState<ClinicalResponse | null>(null)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [selectedDoc, setSelectedDoc] = useState<DocumentDetail | null>(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  useEffect(() => {
    if (!loading) return
    setElapsedSeconds(0)
    const started = Date.now()
    const tick = setInterval(() => setElapsedSeconds(Math.floor((Date.now() - started) / 1000)), 1000)
    return () => clearInterval(tick)
  }, [loading])

  // Default fallback sample cases in case backend is loading
  const defaultSampleCases: SampleCase[] = [
    {
      case_id: 'pqa-24318956',
      title: 'Case 1: Digoxin & Prostate Cancer Risk',
      query: 'Is digoxin use for cardiovascular disease associated with risk of prostate cancer?',
      description: 'PubMedQA test question, PMID 24318956. Expert answer: yes.',
    },
    {
      case_id: 'pqa-24666444',
      title: 'Case 2: The "July Effect" in Cancer Surgery',
      query: 'Is there any evidence of a "July effect" in patients undergoing major cancer surgery?',
      description: 'PubMedQA test question, PMID 24666444. Expert answer: no.',
    },
    {
      case_id: 'pqa-25371231',
      title: 'Case 3: Vitamin D & Osteochondritis Dissecans',
      query: 'Is vitamin D insufficiency or deficiency related to the development of osteochondritis dissecans?',
      description: 'PubMedQA test question, PMID 25371231. Expert answer: maybe.',
    },
  ]

  // Check backend health and load sample cases on mount
  useEffect(() => {
    let isMounted = true
    const checkConnection = async () => {
      try {
        const health = await fetchHealth()
        if (isMounted) {
          setBackendHealth(health)
          setBackendError(null)
        }
      } catch (err) {
        if (isMounted) {
          setBackendHealth(null)
          setBackendError('Backend offline. Run: python src/api/app.py in the backend folder.')
        }
      }

      try {
        const cases = await fetchSampleCases()
        if (isMounted && cases && cases.length > 0) {
          setSampleCases(cases)
        } else if (isMounted) {
          setSampleCases(defaultSampleCases)
        }
      } catch (err) {
        if (isMounted) {
          setSampleCases(defaultSampleCases)
        }
      }
    }

    checkConnection()
    const interval = setInterval(checkConnection, 8000)
    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [])

  const handleSearch = async (textToSearch?: string, caseId?: string) => {
    const queryText = (textToSearch !== undefined ? textToSearch : inputQuery).trim()
    if (!queryText || loading) return

    setInputQuery(queryText)
    setActiveQuery(queryText)
    setLoading(true)
    setQueryError(null)
    setActiveStage(1)

    // Progressive stage visualization timers
    const timer1 = setTimeout(() => setActiveStage(2), 500)
    const timer2 = setTimeout(() => setActiveStage(3), 1100)
    const timer3 = setTimeout(() => setActiveStage(4), 1700)

    try {
      // Execute live query via backend API - dynamic, not hardcoded
      const result = await executeClinicalQuery(queryText, caseId)

      clearTimeout(timer1)
      clearTimeout(timer2)
      clearTimeout(timer3)
      setActiveStage(0)
      setCurrentResponse(result)
      setActiveTab('sources')
    } catch (err: any) {
      clearTimeout(timer1)
      clearTimeout(timer2)
      clearTimeout(timer3)
      setActiveStage(0)
      setQueryError(
        err.message ||
          'Could not reach SafeMed AI backend. Please verify python src/api/app.py is running on port 8000.'
      )
    } finally {
      setLoading(false)
    }
  }

  const openSource = (idx: number, focusAnswerSentence?: number) => {
    if (!currentResponse) return
    const src = currentResponse.trusted_sources[idx]
    if (!src) return
    setSelectedDoc({
      ...src,
      kind: src.verification_status?.includes('reinstated') ? 'reinstated' : 'trusted',
      rank: idx + 1,
      sentences: currentResponse.attribution?.sources[idx]?.sentences,
      focusAnswerSentence,
    })
  }

  const renderSentenceWithCitations = (text: string, sentenceIdx: number) =>
    text.split(/(\[\d+(?:\s*[,-]\s*\d+)*\])/g).map((part, i) => {
      const nums = /^\[(.+)\]$/.exec(part)
      if (!nums) return <span key={i}>{part}</span>
      const ids = nums[1].split(/\s*,\s*/).flatMap((p) => {
        const [lo, hi] = p.split('-').map(Number)
        return hi ? Array.from({ length: hi - lo + 1 }, (_, k) => lo + k) : [lo]
      })
      return (
        <span key={i} className="whitespace-nowrap">
          {ids.map((n) => (
            <button
              key={n}
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                openSource(n - 1, sentenceIdx)
              }}
              title={currentResponse?.trusted_sources[n - 1]?.title ?? `Source ${n}`}
              className="mx-0.5 px-1 rounded bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 font-mono text-[11px] font-bold hover:bg-emerald-500/30 cursor-pointer align-baseline"
            >
              [{n}]
            </button>
          ))}
        </span>
      )
    })

  const handleClear = () => {
    setInputQuery('')
    setActiveQuery('')
    setCurrentResponse(null)
    setQueryError(null)
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* ── Page Header / Hero ── */}
      <div className="text-center space-y-3 max-w-3xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-mono font-medium">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>Stage 2 Pre-Reranking Safety Filter Active</span>
        </div>

        <h1 className="text-3xl sm:text-4xl font-serif font-bold text-foreground tracking-tight">
          Clinical Evidence Search
        </h1>
        <p className="text-sm sm:text-base text-muted-foreground leading-relaxed">
          Search peer-reviewed PubMed literature through a 4-Stage Modular RAG Pipeline with automated Pre-Reranking Safety Filtering to eliminate contradictory and misleading evidence.
        </p>

        {/* Backend Connectivity Status */}
        <div className="flex items-center justify-center gap-2 pt-1">
          {backendHealth ? (
            <span className="inline-flex items-center gap-1.5 text-xs font-mono text-emerald-600 dark:text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Backend Connected (Port 8000) • MiniLM + Cross-Encoder
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-xs font-mono text-amber-600 dark:text-amber-400">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-ping" />
              Connecting to backend service (http://127.0.0.1:8000)...
            </span>
          )}
        </div>
      </div>

      {/* Backend Alert if offline */}
      {backendError && (
        <div className="max-w-3xl mx-auto p-3.5 rounded-xl border border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-200 text-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <span>⚠️ {backendError}</span>
          <code className="px-2 py-0.5 rounded bg-background/80 font-mono text-[11px] text-amber-700 dark:text-amber-300 shrink-0">
            cd backend && python src/api/app.py
          </code>
        </div>
      )}

      {/* ── Direct Clinical Search Input Bar ── */}
      <div className="max-w-4xl mx-auto space-y-4">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            handleSearch()
          }}
          className="relative flex items-center shadow-lg rounded-2xl border border-border bg-card focus-within:border-emerald-500 focus-within:ring-2 focus-within:ring-emerald-500/20 transition-all p-2 gap-2"
        >
          <div className="pl-3 text-muted-foreground text-lg">
            🔍
          </div>

          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            placeholder="Enter clinical question (e.g., Does daily aspirin reduce colorectal cancer risk in Lynch syndrome?)..."
            disabled={loading}
            className="w-full bg-transparent border-none text-foreground placeholder:text-muted-foreground/60 text-sm sm:text-base focus:outline-hidden py-2 px-1"
          />

          {inputQuery && !loading && (
            <button
              type="button"
              onClick={handleClear}
              className="text-muted-foreground hover:text-foreground p-1.5 rounded-lg hover:bg-muted text-xs font-mono transition-colors"
              title="Clear input"
            >
              ✕
            </button>
          )}

          <button
            type="submit"
            disabled={loading || !inputQuery.trim()}
            className="px-5 py-2.5 rounded-xl font-mono text-xs sm:text-sm font-semibold bg-emerald-600 hover:bg-emerald-700 text-white shadow-xs disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shrink-0 cursor-pointer"
          >
            {loading ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Searching... {elapsedSeconds}s</span>
              </>
            ) : (
              <>
                <span>Search Evidence</span>
                <span>→</span>
              </>
            )}
          </button>
        </form>

        {/* ── Clickable Sample Clinical Question Chips (No Need to Type) ── */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono uppercase tracking-wider font-semibold text-muted-foreground">
              Sample Clinical Cases (Click to Run Live Pipeline):
            </span>
            {currentResponse && (
              <button
                onClick={handleClear}
                className="text-[11px] font-mono text-muted-foreground hover:text-foreground underline cursor-pointer"
              >
                Reset Search
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
            {(sampleCases.length > 0 ? sampleCases : defaultSampleCases).map((c) => {
              const isSelected = activeQuery === c.query
              return (
                <button
                  key={c.case_id}
                  onClick={() => handleSearch(c.query, c.case_id)}
                  disabled={loading}
                  className={`p-3 rounded-xl border text-left transition-all group flex flex-col justify-between gap-2 cursor-pointer ${
                    isSelected
                      ? 'border-emerald-500 bg-emerald-500/10 shadow-xs ring-1 ring-emerald-500/30'
                      : 'border-border bg-card hover:border-emerald-500/40 hover:bg-card/90 shadow-2xs'
                  }`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono font-bold text-emerald-600 dark:text-emerald-400">
                        {c.title.split(':')[0]}
                      </span>
                      <span className="text-[11px] text-muted-foreground group-hover:text-emerald-500 transition-transform group-hover:translate-x-1">
                        →
                      </span>
                    </div>
                    <p className="text-xs font-medium text-foreground line-clamp-2">
                      {c.query}
                    </p>
                  </div>
                  <span className="text-[10px] text-muted-foreground line-clamp-1 italic">
                    {c.description}
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* ── Query Execution Error Display ── */}
      {queryError && (
        <div className="max-w-4xl mx-auto p-4 rounded-xl border border-red-500/30 bg-red-50/50 dark:bg-red-950/20 text-red-900 dark:text-red-200 text-sm space-y-1">
          <strong className="font-semibold">Query Execution Failed:</strong>
          <p className="text-xs font-mono">{queryError}</p>
        </div>
      )}

      {/* ── Active 4-Stage Execution Progress Stepper (While Loading) ── */}
      {loading && (
        <div className="max-w-4xl mx-auto p-6 rounded-2xl border border-emerald-500/30 bg-card shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping" />
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                Executing 4-Stage SafeMed AI Protocol
              </span>
            </div>
            <span className="text-xs font-mono text-muted-foreground truncate max-w-xs">
              &ldquo;{activeQuery}&rdquo;
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs font-mono">
            <div
              className={`p-3 rounded-xl border text-center transition-all ${
                activeStage >= 1
                  ? 'border-emerald-500 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold'
                  : 'border-border text-muted-foreground/40'
              }`}
            >
              <div className="text-[10px] opacity-70">Stage 1</div>
              <div>Retrieval (Top-50)</div>
            </div>

            <div
              className={`p-3 rounded-xl border text-center transition-all ${
                activeStage >= 2
                  ? 'border-emerald-500 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold'
                  : 'border-border text-muted-foreground/40'
              }`}
            >
              <div className="text-[10px] opacity-70">Stage 2</div>
              <div>Safety Filter Scan</div>
            </div>

            <div
              className={`p-3 rounded-xl border text-center transition-all ${
                activeStage >= 3
                  ? 'border-emerald-500 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold'
                  : 'border-border text-muted-foreground/40'
              }`}
            >
              <div className="text-[10px] opacity-70">Stage 3</div>
              <div>Relevance Rerank (Top-5)</div>
            </div>

            <div
              className={`p-3 rounded-xl border text-center transition-all ${
                activeStage >= 4
                  ? 'border-emerald-500 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold'
                  : 'border-border text-muted-foreground/40'
              }`}
            >
              <div className="text-[10px] opacity-70">Stage 4</div>
              <div>Consensus Synthesis</div>
            </div>
          </div>
        </div>
      )}

      {/* ── Search Results: Direct Clinical Evidence Synthesis ── */}
      {currentResponse && !loading && (
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Main Clinical Consensus Summary Card */}
          <div className="p-6 rounded-2xl border border-emerald-500/30 bg-card shadow-sm space-y-4">
            {/* Top Status Row */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
              <div className="flex items-center gap-2.5">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 border border-emerald-500/30">
                  ✓ {currentResponse.consensus_status}
                </span>
                <span className="text-xs text-muted-foreground font-mono">
                  Verified Clinical Consensus
                </span>
              </div>

              {/* Quick Pipeline Stats */}
              <div className="flex items-center gap-2 text-[11px] font-mono text-muted-foreground">
                <span className="px-2 py-0.5 rounded bg-muted">
                  50 Passages Screened
                </span>
                <span className="px-2 py-0.5 rounded bg-red-500/10 text-red-600 dark:text-red-400 font-semibold">
                  {currentResponse.safety_scan.blocked_count} Blocked Harmful
                </span>
                <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-semibold">
                  Top-5 Safe Context
                </span>
              </div>
            </div>

            {/* Queried Question */}
            <div>
              <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider block mb-1">
                Clinical Question:
              </span>
              <h2 className="text-base sm:text-lg font-serif font-semibold text-foreground">
                {currentResponse.query}
              </h2>
            </div>

            {/* Synthesized Narrative Summary */}
            <div className="bg-muted/30 p-4 rounded-xl border border-border/60">
              <span className="text-[10px] font-mono uppercase tracking-wider font-semibold text-emerald-600 dark:text-emerald-400 block mb-1.5">
                Evidence Synthesis & Findings:
              </span>
              {currentResponse.attribution && currentResponse.attribution.answer_sentences.length > 0 ? (
                <>
                  <p className="text-sm sm:text-base text-foreground leading-relaxed font-sans">
                    {currentResponse.attribution.answer_sentences.map((sent) => {
                      const best = sent.supported_by[0]
                      return (
                        <span
                          key={sent.index}
                          onClick={() => best && openSource(best.source - 1, sent.index)}
                          title={
                            best
                              ? `Best supported by source [${best.source}] (similarity ${best.score}). Click to see the passage.`
                              : 'No closely matching passage found in the top-5 sources.'
                          }
                          className={best ? 'cursor-pointer rounded hover:bg-amber-200/50 dark:hover:bg-amber-500/20' : ''}
                        >
                          {renderSentenceWithCitations(sent.text, sent.index)}{' '}
                        </span>
                      )
                    })}
                  </p>
                  <p className="text-[10px] font-mono text-muted-foreground mt-2">
                    Click a [n] citation or a sentence to open the source with the supporting passage highlighted. Matching:{' '}
                    {currentResponse.attribution.method === 'embedding' ? 'MiniLM sentence similarity' : 'word overlap'}.
                    {currentResponse.llm_model && <> Generated by {currentResponse.llm_model}.</>}
                  </p>
                </>
              ) : (
                <p className="text-sm sm:text-base text-foreground leading-relaxed font-sans">
                  {currentResponse.clinical_summary}
                </p>
              )}
            </div>
          </div>

          {/* ── Detailed Evidence & Safety Audit Tabs ── */}
          <div className="rounded-2xl border border-border bg-card shadow-xs overflow-hidden">
            {/* Tab Navigation Header */}
            <div className="flex flex-wrap items-center gap-2 border-b border-border bg-muted/20 p-2.5">
              <button
                onClick={() => setActiveTab('sources')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-medium transition-colors cursor-pointer ${
                  activeTab === 'sources'
                    ? 'bg-emerald-600 text-white font-bold shadow-xs'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                }`}
              >
                📚 Top-5 Evidence Sources ({currentResponse.trusted_sources.length})
              </button>

              <button
                onClick={() => setActiveTab('safety')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-medium transition-colors cursor-pointer flex items-center gap-1.5 ${
                  activeTab === 'safety'
                    ? 'bg-emerald-600 text-white font-bold shadow-xs'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                }`}
              >
                <span>🛡️ Pre-Reranking Safety Log</span>
                <span className="px-1.5 py-0.2 rounded text-[10px] bg-red-500/20 text-red-600 dark:text-red-300 font-bold">
                  {currentResponse.safety_scan.blocked_count} blocked
                </span>
              </button>

              <button
                onClick={() => setActiveTab('claims')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-medium transition-colors cursor-pointer ${
                  activeTab === 'claims'
                    ? 'bg-emerald-600 text-white font-bold shadow-xs'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                }`}
              >
                🔬 Atomic Claims ({currentResponse.atomic_claims?.length || 0})
              </button>

              {currentResponse.evaluation_breakdown && (
                <button
                  onClick={() => setActiveTab('evaluation')}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-medium transition-colors cursor-pointer ${
                    activeTab === 'evaluation'
                      ? 'bg-emerald-600 text-white font-bold shadow-xs'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  }`}
                >
                  📊 Thesis Evaluation Breakdown
                </button>
              )}
            </div>

            {/* Tab 1: Top-5 Evidence Context Sources */}
            {activeTab === 'sources' && (
              <div className="p-5 space-y-3">
                <p className="text-xs text-muted-foreground font-mono">
                  These 5 verified passages were selected by the Stage 3 Cross-Encoder Reranker after the Stage 2 Safety Filter purged all harmful/contradictory candidates. Click any document to read the full abstract:
                </p>

                <div className="grid gap-3">
                  {currentResponse.trusted_sources.map((src, idx) => (
                    <button
                      type="button"
                      key={src.id || idx}
                      onClick={() => openSource(idx)}
                      title="Click to read the full abstract"
                      className="w-full text-left cursor-pointer p-4 rounded-xl border border-emerald-500/20 bg-emerald-50/10 dark:bg-emerald-950/10 hover:border-emerald-500/60 hover:bg-emerald-50/20 dark:hover:bg-emerald-950/30 transition-colors flex flex-col sm:flex-row items-start justify-between gap-3 text-xs"
                    >
                      <div className="space-y-1 flex-1">
                        <div className="flex items-center gap-2 font-semibold text-foreground">
                          <span className="text-emerald-600 dark:text-emerald-400 font-mono font-bold">
                            [{idx + 1}]
                          </span>
                          <span>{src.title}</span>
                        </div>
                        <p className="text-[11px] text-muted-foreground font-mono">
                          {src.source} • Cross-Encoder Relevance: {src.relevance_score}
                        </p>
                        {(() => {
                          const used = (currentResponse.attribution?.sources[idx]?.sentences ?? [])
                            .flatMap((s) => s.answer_sentences)
                            .filter((v, i, arr) => arr.indexOf(v) === i)
                            .sort((x, y) => x - y)
                          return currentResponse.attribution ? (
                            <p className="text-[10px] font-mono text-amber-700 dark:text-amber-300">
                              {used.length > 0
                                ? `Used in answer sentence${used.length > 1 ? 's' : ''} ${used.map((i) => i + 1).join(', ')}`
                                : 'Not visibly used in the answer'}
                            </p>
                          ) : null
                        })()}
                        <p className="text-xs text-foreground/80 font-serif pt-1 leading-relaxed bg-background/50 p-2.5 rounded-lg border border-border/40">
                          &ldquo;{src.snippet}&rdquo;
                        </p>
                      </div>
                      <span className="px-2.5 py-1 rounded text-[10px] font-mono bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 shrink-0 font-bold border border-emerald-500/20">
                        {src.verification_status?.includes('reinstated') ? '↺ REINSTATED' : '✓ VERIFIED SAFE'}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Tab 2: Pre-Reranking Safety Filter Audit Log */}
            {activeTab === 'safety' && (
              <div className="p-5 space-y-4">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-muted-foreground">
                    Stage 2 Filter Decision: Total Screened: {currentResponse.safety_scan.total_scanned} | Blocked: {currentResponse.safety_scan.blocked_count} | Retained: {currentResponse.safety_scan.safe_count}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-semibold">
                    Cutoff Threshold τ = 0.50
                  </span>
                </div>

                <div className="space-y-2.5 max-h-96 overflow-y-auto pr-1">
                  {currentResponse.safety_scan.evaluation_log.map((doc, idx) => (
                    <button
                      type="button"
                      key={doc.id || idx}
                      onClick={() =>
                        setSelectedDoc({
                          ...doc,
                          kind: doc.reinstated ? 'reinstated' : doc.is_blocked ? 'blocked' : 'retained',
                        })
                      }
                      title="Click to read the full abstract"
                      className={`block w-full text-left cursor-pointer p-3.5 rounded-xl border text-xs transition-colors ${
                        doc.is_blocked
                          ? 'border-red-500/40 bg-red-50/40 dark:bg-red-950/20 hover:border-red-500/80'
                          : 'border-border/60 bg-card/60 hover:border-emerald-500/50'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className="font-semibold text-foreground truncate">
                          {doc.title}
                        </span>
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold shrink-0 ${
                            doc.is_blocked
                              ? 'bg-red-600 text-white'
                              : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
                          }`}
                        >
                          {doc.reinstated ? '↺ REINSTATED' : doc.is_blocked ? '✕ BLOCKED HARMFUL' : '✓ RETAINED'}
                        </span>
                      </div>
                      <p className="text-[11px] text-muted-foreground font-mono mb-1">
                        {doc.source} • Document Category: {doc.category} • P(HD): {doc.harmful_probability.toFixed(3)}
                      </p>
                      {doc.is_blocked && doc.block_reason && (
                        <p className="text-red-700 dark:text-red-300 font-medium text-[11px] mt-1.5 bg-red-100/60 dark:bg-red-950/50 p-2 rounded-lg border border-red-500/20">
                          <strong>Pre-Rerank Safety Interception:</strong> {doc.block_reason}
                        </p>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Tab 3: Atomic Fact-Level Claims */}
            {activeTab === 'claims' && (
              <div className="p-5 space-y-3">
                <p className="text-xs text-muted-foreground font-mono">
                  Extracted atomic claims verified against the retrieved evidence pool:
                </p>
                <div className="grid gap-2">
                  {currentResponse.atomic_claims?.map((claim, cIdx) => (
                    <div
                      key={cIdx}
                      className="p-3 rounded-xl border border-border bg-muted/20 text-xs text-foreground flex items-start gap-2.5"
                    >
                      <span className="w-5 h-5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center font-bold text-[11px] shrink-0 font-mono">
                        {cIdx + 1}
                      </span>
                      <span className="leading-relaxed font-sans">{claim}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Tab 4: Thesis Single Query Evaluation */}
            {activeTab === 'evaluation' && currentResponse.evaluation_breakdown && (
              <div className="p-4">
                <SingleQueryResults breakdown={currentResponse.evaluation_breakdown} />
              </div>
            )}
          </div>
        </div>
      )}

      <DocumentDetailModal
        doc={selectedDoc}
        answerSentences={currentResponse?.attribution?.answer_sentences}
        onClose={() => setSelectedDoc(null)}
      />
    </div>
  )
}
