import type { ClinicalResponse } from '../../lib/api'
import { CATEGORY_STYLE, displaySource } from '../../lib/display'
import { IconBook } from '../icons'

interface Props {
  response: ClinicalResponse | null
  loading: boolean
  onOpen: (idx: number) => void
}

export default function SourcesPanel({ response, loading, onOpen }: Props) {
  const categories = new Map(
    (response?.evaluation_breakdown?.proposed.top5_documents ?? []).map((d) => [String(d.id), d.category]),
  )

  return (
    <section className="h-full flex flex-col bg-card border border-border rounded-2xl overflow-hidden">
      <header className="px-4 py-3 border-b border-border flex items-center gap-2">
        <IconBook className="text-muted-foreground" />
        <h2 className="text-sm font-semibold">Sources</h2>
        {response && <span className="text-xs text-muted-foreground">{response.trusted_sources.length}</span>}
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        {loading && (
          <div className="space-y-2 p-2">
            {[0, 1, 2, 3, 4].map((i) => (
              <div key={i} className="h-16 rounded-xl bg-muted animate-pulse" />
            ))}
          </div>
        )}

        {!loading && !response && (
          <p className="text-sm text-muted-foreground p-3 leading-relaxed">
            The five documents passed to the generator will appear here, after the safety filter and the reranker.
          </p>
        )}

        {!loading &&
          response?.trusted_sources.map((src, idx) => {
            const category = categories.get(String(src.id)) ?? 'MD'
            const style = CATEGORY_STYLE[category] ?? CATEGORY_STYLE.MD
            const used = (response.attribution?.sources[idx]?.sentences ?? []).some((s) => s.answer_sentences.length > 0)
            const reinstated = src.verification_status?.includes('reinstated')
            return (
              <button
                key={src.id ?? idx}
                type="button"
                onClick={() => onOpen(idx)}
                className="w-full text-left p-3 rounded-xl hover:bg-muted transition-colors cursor-pointer group"
              >
                <div className="flex items-start gap-2.5">
                  <span className="mt-0.5 w-5 h-5 shrink-0 rounded-md bg-primary/10 text-primary text-[11px] font-semibold flex items-center justify-center">
                    {idx + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-[13px] font-medium leading-snug line-clamp-2 group-hover:text-primary">{src.title}</p>
                    <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
                      <span title={style.title} className={`px-1.5 py-px rounded border font-medium ${style.className}`}>
                        {style.label}
                      </span>
                      {reinstated && (
                        <span className="px-1.5 py-px rounded border border-amber-200 bg-amber-50 text-amber-800 dark:bg-amber-950/50 dark:text-amber-200 dark:border-amber-900">
                          Reinstated
                        </span>
                      )}
                      <span className="truncate">{displaySource(src.source)}</span>
                    </div>
                    {response.attribution && (
                      <p className={`mt-1 text-[11px] ${used ? 'text-primary' : 'text-muted-foreground/70'}`}>
                        {used ? 'Cited in the answer' : 'Not cited in the answer'}
                      </p>
                    )}
                  </div>
                </div>
              </button>
            )
          })}
      </div>
    </section>
  )
}
