import { useState } from 'react'
import GlossaryTooltip from './GlossaryTooltip'

const STAGES = [
  {
    id: 'retriever',
    label: '① Retriever',
    model: 'all-MiniLM-L6-v2',
    role: 'Bi-encoder that scores and retrieves the top-50 candidate passages from the PubMedQA corpus.',
    both: true,
  },
  {
    id: 'filter',
    label: '② Filter',
    model: 'Fine-tuned cross-encoder',
    role: 'Jointly encodes query and each candidate; outputs P(GD)/P(HD)/P(MD). Removes candidates above the tuned harmful-probability threshold and truncates to top-30.',
    both: false,
    treatmentOnly: true,
  },
  {
    id: 'reranker',
    label: '③ Reranker',
    model: 'ms-marco-MiniLM-L-6-v2',
    role: 'Cross-encoder that orders remaining candidates by query–document relevance. Selects top-5.',
    both: true,
  },
  {
    id: 'generator',
    label: '④ Generator',
    model: 'GPT-4o-mini',
    role: 'Generates an answer from the top-5 documents only.',
    both: true,
  },
]

interface Props {
  compact?: boolean
}

export default function PipelineSchematic({ compact = false }: Props) {
  const [hovered, setHovered] = useState<string | null>(null)

  return (
    <div className="rounded border border-border bg-card p-4">
      <div className="text-xs font-mono text-muted-foreground mb-4 flex items-center gap-3">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: 'var(--color-control)' }} />
          Standard RAG (control)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: 'var(--color-treatment)' }} />
          Proposed (treatment)
        </span>
      </div>

      {/* Two-column paths */}
      <div className="grid grid-cols-2 gap-4">
        {/* Control */}
        <div>
          <div className="text-[10px] font-mono font-semibold mb-2 uppercase tracking-wider" style={{ color: 'var(--color-control)' }}>
            Standard RAG
          </div>
          <div className="space-y-1.5">
            {STAGES.map((stage, i) => (
              <div key={stage.id}>
                <button
                  className={`w-full text-left rounded border px-3 py-2 transition-all text-xs font-mono ${
                    stage.treatmentOnly
                      ? 'border-dashed border-border/40 text-muted-foreground/40 bg-muted/30 cursor-default'
                      : hovered === stage.id
                      ? 'border-border bg-muted shadow-sm'
                      : 'border-border/60 bg-card hover:bg-muted'
                  }`}
                  onMouseEnter={() => !stage.treatmentOnly && setHovered(stage.id)}
                  onMouseLeave={() => setHovered(null)}
                  disabled={!!stage.treatmentOnly}
                  aria-label={stage.treatmentOnly ? 'Skipped — no filter in Standard RAG' : stage.label}
                >
                  {stage.treatmentOnly ? (
                    <span className="opacity-40">② Filter — skipped</span>
                  ) : (
                    <span>
                      <GlossaryTooltip term={stage.model}>
                        {stage.label}
                      </GlossaryTooltip>
                    </span>
                  )}
                </button>
                {i < STAGES.length - 1 && !stage.treatmentOnly && (
                  <div className="flex justify-center my-0.5">
                    <span className="text-border text-xs">↓</span>
                  </div>
                )}
                {stage.treatmentOnly && (
                  <div className="flex justify-center my-0.5">
                    <span className="text-border text-xs">↓</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Treatment */}
        <div>
          <div className="text-[10px] font-mono font-semibold mb-2 uppercase tracking-wider" style={{ color: 'var(--color-treatment)' }}>
            Proposed
          </div>
          <div className="space-y-1.5">
            {STAGES.map((stage, i) => (
              <div key={stage.id}>
                <button
                  className={`w-full text-left rounded border px-3 py-2 transition-all text-xs font-mono ${
                    hovered === stage.id
                      ? 'border-border bg-muted shadow-sm'
                      : 'border-border/60 bg-card hover:bg-muted'
                  }`}
                  style={stage.id === 'filter' ? { borderColor: 'var(--color-treatment)', background: 'var(--color-treatment-bg)' } : {}}
                  onMouseEnter={() => setHovered(stage.id)}
                  onMouseLeave={() => setHovered(null)}
                  aria-label={stage.label}
                >
                  <GlossaryTooltip term={stage.model}>
                    {stage.label}
                  </GlossaryTooltip>
                </button>
                {i < STAGES.length - 1 && (
                  <div className="flex justify-center my-0.5">
                    <span className="text-border text-xs">↓</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tooltip panel */}
      {hovered && (() => {
        const stage = STAGES.find((s) => s.id === hovered)!
        return (
          <div className="mt-4 p-3 rounded border border-border bg-background text-xs leading-relaxed">
            <div className="font-mono font-semibold text-foreground mb-1">{stage.label}</div>
            <div className="font-mono text-[10px] text-muted-foreground mb-1">{stage.model}</div>
            <div className="text-muted-foreground">{stage.role}</div>
          </div>
        )
      })()}
    </div>
  )
}
