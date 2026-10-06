import { Outlet } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { fetchHealth, type SystemHealth } from '../lib/api'
import { IconMoon, IconShield, IconSun } from './icons'

function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  const toggle = () => {
    setDark((v) => {
      document.documentElement.classList.toggle('dark', !v)
      return !v
    })
  }
  return (
    <button
      onClick={toggle}
      className="w-8 h-8 inline-flex items-center justify-center rounded-full text-muted-foreground hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
      aria-label="Toggle light/dark theme"
    >
      {dark ? <IconSun /> : <IconMoon />}
    </button>
  )
}

function StatusPill() {
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [offline, setOffline] = useState(false)

  useEffect(() => {
    let alive = true
    const check = () =>
      fetchHealth()
        .then((h) => alive && (setHealth(h), setOffline(false)))
        .catch(() => alive && (setHealth(null), setOffline(true)))
    check()
    const id = setInterval(check, 10000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  if (offline) {
    return (
      <span className="inline-flex items-center gap-2 text-xs text-amber-700 dark:text-amber-300" title="cd backend && python src/api/app.py">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-500" /> Backend offline
      </span>
    )
  }
  if (!health) {
    return (
      <span className="inline-flex items-center gap-2 text-xs text-muted-foreground">
        <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-pulse" /> Connecting…
      </span>
    )
  }
  return (
    <span
      className="hidden sm:inline-flex items-center gap-2 text-xs text-muted-foreground"
      title={`Generator: ${health.stage_4_generator} · Judge: ${health.stage_5_judge ?? 'n/a'}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${health.stage_4_ready ? 'bg-emerald-500' : 'bg-amber-500'}`} />
      {health.stage_4_generator}
      {health.stage_5_judge && (
        <>
          <span className="text-border">|</span>
          <span className={`w-1.5 h-1.5 rounded-full ${health.stage_5_ready ? 'bg-emerald-500' : 'bg-amber-500'}`} />
          judge {health.stage_5_judge}
        </>
      )}
    </span>
  )
}

export default function Layout() {
  return (
    <div className="h-screen flex flex-col bg-background text-foreground font-sans">
      <header className="shrink-0 h-14 border-b border-border bg-card">
        <div className="h-full px-4 sm:px-6 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center">
              <IconShield size={18} />
            </div>
            <div className="leading-tight">
              <div className="font-semibold text-[15px] tracking-tight">SafeMed AI</div>
              <div className="text-[11px] text-muted-foreground hidden sm:block">Biomedical RAG with a pre-reranking safety filter</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <StatusPill />
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main className="flex-1 min-h-0">
        <Outlet />
      </main>
    </div>
  )
}
