import { useEffect, useRef, useState } from 'react'
import {
  executeClinicalQuery,
  fetchSampleCases,
  runHallucinationCheck,
  type ClinicalResponse,
  type DocumentEvaluationEntry,
  type SampleCase,
} from '../lib/api'
import { DECISION_STYLE, displaySource, splitDecision } from '../lib/display'
import DocumentDetailModal, { type DocumentDetail } from '../components/DocumentDetailModal'
import SourcesPanel from '../components/workspace/SourcesPanel'
import SafetyPanel, { type SessionRun } from '../components/workspace/SafetyPanel'
import HallucinationCard from '../components/workspace/HallucinationCard'
import { IconSend, IconSparkle } from '../components/icons'

const STEPS = [
  'Retrieving 50 candidates (MiniLM bi-encoder)',
  'Screening every candidate with the safety filter',
  'Reranking the survivors to the top 5',
  'Writing the answer, with and without the filter',
]

export default function Workspace() {
  const [input, setInput] = useState('')
  const [question, setQuestion] = useState('')
  const [cases, setCases] = useState<SampleCase[]>([])
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [response, setResponse] = useState<ClinicalResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<DocumentDetail | null>(null)
  const [session, setSession] = useState<SessionRun[]>([])
  const [checking, setChecking] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const autoRan = useRef(false)

  useEffect(() => {
    fetchSampleCases()
      .then((list) => {
        setCases(list)
        const wanted = new URLSearchParams(window.location.search).get('case')
        const c = list.find((x) => x.case_id === wanted)
        if (c && !autoRan.current) {
          autoRan.current = true
          ask(c.query, c.case_id)
        }
      })
      .catch(() => setCases([]))
  }, [])

  useEffect(() => {
    if (!loading) return
    const started = Date.now()
    setElapsed(0)
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000)
    return () => clearInterval(id)
  }, [loading])

  const ask = async (text?: string, caseId?: string) => {
    const q = (text ?? input).trim()
    if (!q || loading || checking) return
    setQuestion(q)
    setInput('')
    setLoading(true)
    setError(null)
    setResponse(null)
    const id = caseId ?? cases.find((c) => c.query === q)?.case_id
    let res: ClinicalResponse
    try {
      res = await executeClinicalQuery(q, id, false)
      setResponse(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not reach the SafeMed AI backend on port 8000.')
      return
    } finally {
      setLoading(false)
    }
    setChecking(true)
    try {
      const hc = await runHallucinationCheck(res, id)
      setResponse((r) => (r === res ? { ...res, hallucination_check: hc, hallucination_error: null } : r))
      setSession((s) => [
        ...s,
        {
          query: q,
          controlHallucinated: hc.control.hallucinated,
          proposedHallucinated: hc.proposed.hallucinated,
          controlHdInduced: hc.control.hd_induced,
          proposedHdInduced: hc.proposed.hd_induced,
        },
      ])
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Judge unavailable.'
      setResponse((r) => (r === res ? { ...res, hallucination_error: msg } : r))
    } finally {
      setChecking(false)
    }
  }

  const openSource = (idx: number, focusAnswerSentence?: number) => {
    const src = response?.trusted_sources[idx]
    if (!src) return
    setSelected({
      ...src,
      source: displaySource(src.source),
      kind: src.verification_status?.includes('reinstated') ? 'reinstated' : 'trusted',
      rank: idx + 1,
      sentences: response?.attribution?.sources[idx]?.sentences,
      focusAnswerSentence,
    })
  }

  const openScanned = (doc: DocumentEvaluationEntry) =>
    setSelected({
      ...doc,
      source: displaySource(doc.source),
      kind: doc.reinstated ? 'reinstated' : doc.is_blocked ? 'blocked' : 'retained',
    })

  const renderCitations = (text: string, sentenceIdx: number) =>
    text.split(/(\[\d+(?:\s*[,-]\s*\d+)*\])/g).map((part, i) => {
      const m = /^\[(.+)\]$/.exec(part)
      if (!m) return <span key={i}>{part}</span>
      const ids = m[1].split(/\s*,\s*/).flatMap((p) => {
        const [lo, hi] = p.split('-').map(Number)
        return hi ? Array.from({ length: hi - lo + 1 }, (_, k) => lo + k) : [lo]
      })
      return (
        <span key={i} className="whitespace-nowrap">
          {ids.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => openSource(n - 1, sentenceIdx)}
              title={response?.trusted_sources[n - 1]?.title ?? `Source ${n}`}
              className="mx-0.5 inline-flex items-center justify-center min-w-5 h-5 px-1 rounded-md bg-primary/10 text-primary text-[11px] font-semibold align-[2px] hover:bg-primary/20 cursor-pointer"
            >
              {n}
            </button>
          ))}
        </span>
      )
    })

  const decision = response?.decision ?? (response ? splitDecision(response.clinical_summary).decision : null)
  const sentences = response?.attribution?.answer_sentences ?? []

  return (
    <div className="h-full p-3 sm:p-4 grid gap-3 sm:gap-4 grid-cols-1 lg:grid-cols-[minmax(240px,300px)_minmax(0,1fr)_minmax(280px,340px)] overflow-y-auto lg:overflow-hidden">
      <div className="lg:min-h-0 order-2 lg:order-1">
        <SourcesPanel response={response} loading={loading} onOpen={(i) => openSource(i)} />
      </div>

      <section className="order-1 lg:order-2 lg:min-h-0 flex flex-col bg-card border border-border rounded-2xl overflow-hidden">
        <div className="flex-1 min-h-0 overflow-y-auto">
          {!question && (
            <div className="max-w-2xl mx-auto px-6 pt-16 pb-8 text-center">
              <div className="mx-auto w-11 h-11 rounded-2xl bg-primary/10 text-primary flex items-center justify-center">
                <IconSparkle size={22} />
              </div>
              <h1 className="mt-4 text-2xl font-semibold tracking-tight">Ask a biomedical question</h1>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                SafeMed AI retrieves PubMed abstracts, removes documents that look relevant but point to the wrong answer,
                and writes a cited answer. Each answer is then checked for hallucination, with and without the filter.
              </p>
              <p className="mt-3 text-[12px] text-muted-foreground/80 leading-relaxed">
                Demo corpus: 1,000 expert-labelled PubMedQA abstracts plus labelled synthetic test documents. The thesis
                experiment runs separately on the 211,269-abstract pqa_artificial corpus.
              </p>
              <div className="mt-8 grid sm:grid-cols-2 gap-2 text-left">
                {cases.map((c) => (
                  <button
                    key={c.case_id}
                    type="button"
                    onClick={() => ask(c.query, c.case_id)}
                    className="p-3.5 rounded-xl border border-border text-left hover:border-primary/50 hover:bg-muted/50 transition-colors cursor-pointer"
                  >
                    <div className="text-[13px] font-medium">{c.title}</div>
                    <div className="mt-1 text-[12px] text-muted-foreground line-clamp-2">{c.query}</div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {question && (
            <div className="max-w-3xl mx-auto px-5 sm:px-8 py-6 space-y-5">
              <div className="flex justify-end">
                <div className="max-w-[85%] px-4 py-2.5 rounded-2xl rounded-br-md bg-primary text-primary-foreground text-[14px] leading-relaxed">
                  {question}
                </div>
              </div>

              {loading && (
                <div className="rounded-2xl border border-border p-5">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <span className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                    Working on it · <span className="font-mono text-muted-foreground">{elapsed}s</span>
                  </div>
                  <ul className="mt-3 space-y-1.5 text-[13px] text-muted-foreground">
                    {STEPS.map((s) => (
                      <li key={s} className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-border" />
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {error && (
                <div className="rounded-2xl border border-red-200 bg-red-50 dark:bg-red-950/30 dark:border-red-900 p-4 text-[13px] text-red-800 dark:text-red-200">
                  {error}
                </div>
              )}

              {response && (
                <>
                  <article className="space-y-3">
                    <div className="flex items-center gap-2">
                      <span className="w-7 h-7 rounded-full bg-primary/10 text-primary flex items-center justify-center">
                        <IconSparkle size={15} />
                      </span>
                      {decision && (
                        <span className={`px-2.5 py-0.5 rounded-full border text-[12px] font-semibold capitalize ${DECISION_STYLE[decision] ?? ''}`}>
                          Decision: {decision}
                        </span>
                      )}
                    </div>
                    <p className="text-[15.5px] leading-[1.75]">
                      {sentences.length > 0
                        ? sentences.map((s) => {
                            const text = splitDecision(s.text).body
                            const best = s.supported_by[0]
                            return text ? (
                              <span
                                key={s.index}
                                onClick={() => best && openSource(best.source - 1, s.index)}
                                className={best ? 'cursor-pointer rounded hover:bg-amber-100/70 dark:hover:bg-amber-500/15' : ''}
                              >
                                {renderCitations(text, s.index)}{' '}
                              </span>
                            ) : null
                          })
                        : splitDecision(response.clinical_summary).body}
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      Written by {response.llm_model} from the 5 sources on the left. Click a number or a sentence to see the
                      supporting passage. Research prototype; not for clinical decisions.
                    </p>
                  </article>

                  {checking && !response.hallucination_check ? (
                    <div className="rounded-2xl border border-border p-4 flex items-center gap-2 text-[13px] text-muted-foreground">
                      <span className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                      Checking both answers for hallucination (claims judged against the evidence)…
                    </div>
                  ) : (
                    <HallucinationCard
                      check={response.hallucination_check}
                      error={response.hallucination_error}
                      controlAnswer={response.control_answer}
                    />
                  )}
                </>
              )}
            </div>
          )}
        </div>

        <div className="shrink-0 border-t border-border p-3 sm:p-4">
          {question && cases.length > 0 && (
            <div className="max-w-3xl mx-auto mb-2 flex gap-1.5 overflow-x-auto pb-1">
              {cases.map((c) => (
                <button
                  key={c.case_id}
                  type="button"
                  disabled={loading || checking}
                  onClick={() => ask(c.query, c.case_id)}
                  className="shrink-0 px-3 py-1 rounded-full border border-border text-[12px] text-muted-foreground hover:text-foreground hover:border-primary/50 disabled:opacity-50 cursor-pointer"
                >
                  {c.title}
                </button>
              ))}
            </div>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault()
              ask()
            }}
            className="max-w-3xl mx-auto flex items-end gap-2 rounded-2xl border border-border bg-background px-3 py-2 focus-within:border-primary/60 focus-within:ring-4 focus-within:ring-primary/10 transition"
          >
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              disabled={loading}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  ask()
                }
              }}
              placeholder="Ask a biomedical question…"
              className="flex-1 resize-none bg-transparent py-1.5 text-[14px] placeholder:text-muted-foreground/70 focus:outline-none max-h-32"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              aria-label="Ask"
              className="w-9 h-9 shrink-0 rounded-xl bg-primary text-primary-foreground flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              <IconSend />
            </button>
          </form>
        </div>
      </section>

      <div className="order-3 lg:min-h-0">
        <SafetyPanel response={response} loading={loading} session={session} onOpen={openScanned} />
      </div>

      <DocumentDetailModal
        doc={selected}
        tau={response?.tau_safe ?? undefined}
        answerSentences={response?.attribution?.answer_sentences}
        onClose={() => setSelected(null)}
      />
    </div>
  )
}
