import type { Label } from '../lib/schema'

interface Props {
  label: Label
  predicted?: boolean
  size?: 'sm' | 'md'
}

const CONFIG: Record<Label, { icon: string; text: string; style: string }> = {
  GD: {
    icon: '✓',
    text: 'GD',
    style: 'bg-[var(--color-gd-bg)] text-[var(--color-gd)] border-[var(--color-gd)]/30',
  },
  HD: {
    icon: '⚠',
    text: 'HD',
    style: 'bg-[var(--color-hd-bg)] text-[var(--color-hd)] border-[var(--color-hd)]/30',
  },
  MD: {
    icon: '–',
    text: 'MD',
    style: 'bg-[var(--color-md-bg)] text-[var(--color-md)] border-[var(--color-md)]/30',
  },
}

const FULL_NAMES: Record<Label, string> = {
  GD: 'Ground-Truth Document',
  HD: 'Harmful Document',
  MD: 'Mediocre Document',
}

export default function LabelChip({ label, predicted = false, size = 'md' }: Props) {
  const { icon, text, style } = CONFIG[label]
  const sizeClass = size === 'sm' ? 'text-[10px] px-1.5 py-0.5 gap-0.5' : 'text-xs px-2 py-0.5 gap-1'

  return (
    <span
      title={`${FULL_NAMES[label]}${predicted ? ' (predicted)' : ' (annotated)'}`}
      className={`inline-flex items-center border rounded font-mono font-medium leading-none ${sizeClass} ${style} ${predicted ? 'hatched' : ''}`}
    >
      <span aria-hidden="true">{icon}</span>
      <span>{text}</span>
      {predicted && <span className="opacity-60 text-[9px]">~</span>}
    </span>
  )
}
