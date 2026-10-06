import { NavLink, Outlet } from 'react-router-dom'
import { useState } from 'react'

const APP_NAME = 'SafeMed AI'

const NAV_LINKS = [
  { to: '/', label: 'Clinical Search', exact: true },
  { to: '/results', label: 'Benchmark Evaluation (H1–H4)' },
  { to: '/sheet', label: 'Literature Corpus' },
]

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
      className="text-xs font-mono px-2.5 py-1 rounded-lg border border-border hover:bg-muted transition-colors cursor-pointer"
      aria-label="Toggle light/dark theme"
    >
      {dark ? '☀ Light' : '◐ Dark'}
    </button>
  )
}

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground font-sans">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-border bg-background/95 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between gap-4">
          {/* Brand */}
          <NavLink to="/" className="flex items-center gap-2.5 shrink-0 group">
            <div className="w-7 h-7 rounded-lg bg-emerald-600 flex items-center justify-center text-white font-bold text-sm shadow-sm group-hover:bg-emerald-700 transition-colors">
              ✚
            </div>
            <div>
              <span className="font-bold text-base tracking-tight text-foreground">{APP_NAME}</span>
              <span className="hidden sm:inline-block ml-2 text-[10px] font-mono text-muted-foreground uppercase tracking-wider font-semibold">
                Clinical Search
              </span>
            </div>
          </NavLink>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-1" aria-label="Main navigation">
            {NAV_LINKS.map(({ to, label, exact }) => (
              <NavLink
                key={to}
                to={to}
                end={exact}
                className={({ isActive }) =>
                  `text-xs font-mono px-3 py-1.5 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-muted text-foreground font-semibold shadow-xs'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>

          {/* Right Controls */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-[11px] font-mono font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Safety Filter Active
            </div>
            <ThemeToggle />
            <button
              className="md:hidden text-xs font-mono px-2 py-1 border border-border rounded"
              onClick={() => setMobileOpen((v) => !v)}
              aria-label="Toggle mobile menu"
            >
              ☰
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileOpen && (
          <nav className="md:hidden border-t border-border bg-background px-4 py-3 flex flex-col gap-1" aria-label="Mobile navigation">
            {NAV_LINKS.map(({ to, label, exact }) => (
              <NavLink
                key={to}
                to={to}
                end={exact}
                onClick={() => setMobileOpen(false)}
                className={({ isActive }) =>
                  `text-xs font-mono px-3 py-2 rounded-lg ${
                    isActive ? 'bg-muted font-semibold text-foreground' : 'text-muted-foreground'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1" id="main-content">
        <Outlet />
      </main>
    </div>
  )
}
