import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement> & { size?: number }

function base({ size = 16, ...rest }: IconProps) {
  return {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.8,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    'aria-hidden': true,
    ...rest,
  }
}

export const IconShield = (p: IconProps) => (
  <svg {...base(p)}><path d="M12 3l7 3v5c0 4.5-3 8.5-7 10-4-1.5-7-5.5-7-10V6l7-3z" /><path d="M9 12l2 2 4-4" /></svg>
)
export const IconBook = (p: IconProps) => (
  <svg {...base(p)}><path d="M4 5a2 2 0 012-2h12v16H6a2 2 0 00-2 2V5z" /><path d="M4 19a2 2 0 012-2h12" /></svg>
)
export const IconSend = (p: IconProps) => (
  <svg {...base(p)}><path d="M5 12h14" /><path d="M13 6l6 6-6 6" /></svg>
)
export const IconCheck = (p: IconProps) => (
  <svg {...base(p)}><path d="M5 12l4 4 10-10" /></svg>
)
export const IconX = (p: IconProps) => (
  <svg {...base(p)}><path d="M6 6l12 12M18 6L6 18" /></svg>
)
export const IconAlert = (p: IconProps) => (
  <svg {...base(p)}><path d="M12 4l9 16H3l9-16z" /><path d="M12 10v4M12 17h.01" /></svg>
)
export const IconSparkle = (p: IconProps) => (
  <svg {...base(p)}><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z" /></svg>
)
export const IconChevron = (p: IconProps) => (
  <svg {...base(p)}><path d="M9 6l6 6-6 6" /></svg>
)
export const IconMoon = (p: IconProps) => (
  <svg {...base(p)}><path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z" /></svg>
)
export const IconSun = (p: IconProps) => (
  <svg {...base(p)}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
)
