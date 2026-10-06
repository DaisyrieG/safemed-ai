import GlossaryTooltip from './GlossaryTooltip'

interface ConfigValue {
  label: string
  value: number | null
  config: 'control' | 'treatment'
}

interface Props {
  hdTop5?: { control: number | null; treatment: number | null }
  hit5?: { control: number | null; treatment: number | null }
  // Backwards compatibility aliases
  her?: { control: number | null; treatment: number | null }
  gtrr?: { control: number | null; treatment: number | null }
  showDelta?: boolean
}

function MetricBar({
  metric,
  values,
  lowerBetter,
}: {
  metric: string
  values: ConfigValue[]
  lowerBetter: boolean
}) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="font-mono text-xs font-semibold text-foreground">
          <GlossaryTooltip term={metric}>{metric}</GlossaryTooltip>
        </span>
        <span className="text-[10px] text-muted-foreground">{lowerBetter ? 'lower is better' : 'higher is better'}</span>
      </div>
      <div className="space-y-1.5">
        {values.map((v) => (
          <div key={v.config} className="flex items-center gap-2">
            <span
              className="text-[10px] font-mono w-24 shrink-0"
              style={{ color: v.config === 'control' ? 'var(--color-control)' : 'var(--color-treatment)' }}
            >
              {v.label}
            </span>
            <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
              {v.value !== null ? (
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${v.value * 100}%`,
                    backgroundColor:
                      v.config === 'control' ? 'var(--color-control)' : 'var(--color-treatment)',
                  }}
                />
              ) : (
                <div className="h-full bg-border/50 rounded-full" />
              )}
            </div>
            <span className="font-mono text-xs w-10 text-right tabular-nums text-foreground">
              {v.value !== null ? v.value.toFixed(2) : '—'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function PairedMetric({ hdTop5, hit5, her, gtrr, showDelta = false }: Props) {
  const hdData = hdTop5 || her || { control: null, treatment: null }
  const hitData = hit5 || gtrr || { control: null, treatment: null }

  const hdDelta =
    hdData.treatment !== null && hdData.control !== null ? hdData.treatment - hdData.control : null
  const hitDelta =
    hitData.treatment !== null && hitData.control !== null ? hitData.treatment - hitData.control : null

  return (
    <div className="rounded border border-border bg-card p-4 space-y-4">
      <MetricBar
        metric="HD@Top-5"
        lowerBetter={true}
        values={[
          { label: 'Standard RAG', value: hdData.control, config: 'control' },
          { label: 'Proposed', value: hdData.treatment, config: 'treatment' },
        ]}
      />
      <div className="border-t border-border/50" />
      <MetricBar
        metric="Hit@5"
        lowerBetter={false}
        values={[
          { label: 'Standard RAG', value: hitData.control, config: 'control' },
          { label: 'Proposed', value: hitData.treatment, config: 'treatment' },
        ]}
      />
      {showDelta && (hdDelta !== null || hitDelta !== null) && (
        <div className="flex gap-3 pt-1">
          {hdDelta !== null && (
            <span
              className={`text-xs font-mono px-2 py-0.5 rounded border ${
                hdDelta < 0
                  ? 'text-[var(--color-gd)] border-[var(--color-gd)]/30 bg-[var(--color-gd-bg)]'
                  : hdDelta > 0
                  ? 'text-[var(--color-hd)] border-[var(--color-hd)]/30 bg-[var(--color-hd-bg)]'
                  : 'text-muted-foreground border-border bg-muted'
              }`}
            >
              HD@Top-5 {hdDelta >= 0 ? '+' : ''}{hdDelta.toFixed(2)}
            </span>
          )}
          {hitDelta !== null && (
            <span
              className={`text-xs font-mono px-2 py-0.5 rounded border ${
                hitDelta > 0
                  ? 'text-[var(--color-gd)] border-[var(--color-gd)]/30 bg-[var(--color-gd-bg)]'
                  : hitDelta < 0
                  ? 'text-[var(--color-hd)] border-[var(--color-hd)]/30 bg-[var(--color-hd-bg)]'
                  : 'text-muted-foreground border-border bg-muted'
              }`}
            >
              Hit@5 {hitDelta >= 0 ? '+' : ''}{hitDelta.toFixed(2)}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
