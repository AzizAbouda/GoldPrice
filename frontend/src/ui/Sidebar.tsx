import { NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import GoldCoin from './GoldCoin'

const navItems = [
  { to: '/', label: 'Dashboard', icon: 'dashboard' },
  { to: '/predictions', label: 'Predictions', icon: 'chart' },
  { to: '/calculations', label: 'Calculations', icon: 'calc' },
  { to: '/about', label: 'About', icon: 'info' },
]

type SidebarProps = {
  variant: 'desktop' | 'mobile'
  open?: boolean
  onClose?: () => void
  collapsed?: boolean
  onHoverChange?: (collapsed: boolean) => void
}

function Sidebar({ variant, open = false, onClose, collapsed = false, onHoverChange }: SidebarProps) {
  if (variant === 'desktop') {
    return (
      <aside className="hidden md:block">
        <motion.div
          onMouseEnter={() => onHoverChange?.(false)}
          onMouseLeave={() => onHoverChange?.(true)}
          animate={{ width: collapsed ? 72 : 260 }}
          transition={{ duration: 0.25, ease: 'easeInOut' }}
          className="fixed left-6 top-6 bottom-6 rounded-2xl border border-amber-500/20 bg-slate-900/60 backdrop-blur shadow-[0_10px_40px_-15px_rgba(0,0,0,0.6)] p-4 ring-1 ring-amber-400/10"
        >
          {!collapsed && <div className="text-sm font-semibold text-amber-300/80 px-2"></div>}
          <nav className="mt-2 space-y-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => {
                  const base = `flex items-center gap-3 rounded-md transition-colors ${isActive ? 'bg-amber-500/15 text-amber-200 ring-1 ring-amber-400/20' : 'text-white/80 hover:text-amber-200 hover:bg-amber-500/10'}`
                  return collapsed ? `${base} h-10 w-10 justify-center` : `${base} px-3 py-2`
                }}
              >
                {item.icon === 'dashboard' && <GoldCoin className="w-4 h-4" />}
                {item.icon === 'chart' && (
                  <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor" aria-hidden>
                    <path d="M4 5.75a.75.75 0 0 1 .75-.75h.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-.5A.75.75 0 0 1 4 18.25V5.75Zm5 4a.75.75 0 0 1 .75-.75h.5a.75.75 0 0 1 .75.75v8.5a.75.75 0 0 1-.75.75h-.5A.75.75 0 0 1 9 18.25v-8.5Zm5-3a.75.75 0 0 1 .75-.75h.5a.75.75 0 0 1 .75.75v11.5a.75.75 0 0 1-.75.75h-.5a.75.75 0 0 1-.75-.75V6.75Zm5 6a.75.75 0 0 1 .75-.75h.5a.75.75 0 0 1 .75.75v5.5a.75.75 0 0 1-.75.75h-.5a.75.75 0 0 1-.75-.75v-5.5Z"/>
                  </svg>
                )}
                {item.icon === 'calc' && (
                  <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor" aria-hidden>
                    <path d="M6 3.75h12A2.25 2.25 0 0 1 20.25 6v12A2.25 2.25 0 0 1 18 20.25H6A2.25 2.25 0 0 1 3.75 18V6A2.25 2.25 0 0 1 6 3.75Zm1.5 2.5h9v3h-9v-3Zm0 6h2v2h-2v-2Zm0 3.5h2v2h-2v-2Zm3.5-3.5h2v2h-2v-2Zm0 3.5h2v2h-2v-2Zm3.5-3.5h2v2h-2v-2Zm0 3.5h2v2h-2v-2Z"/>
                  </svg>
                )}
                {item.icon === 'info' && (
                  <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor" aria-hidden>
                    <path d="M12 2.25a9.75 9.75 0 1 0 0 19.5 9.75 9.75 0 0 0 0-19.5Zm.75 5.5a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0Zm-2 3a.75.75 0 0 1 .75-.75h1a.75.75 0 0 1 .75.75v6.25h1a.75.75 0 0 1 0 1.5h-4a.75.75 0 0 1 0-1.5h1v-5.5h-.5a.75.75 0 0 1-.75-.75Z"/>
                  </svg>
                )}
                {!collapsed && (
                  <motion.span whileHover={{ x: 2 }} className="inline-block">
                    {item.label}
                  </motion.span>
                )}
              </NavLink>
            ))}
          </nav>
        </motion.div>
      </aside>
    )
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-black/40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            initial={{ x: -280, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -280, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed inset-y-0 left-0 z-50 w-[80%] max-w-[300px] p-4 border-r border-amber-500/20 bg-slate-900/90 backdrop-blur ring-1 ring-amber-400/20"
          >
            <button className="absolute right-3 top-3 w-9 h-9 rounded-md bg-white/10 hover:bg-white/15 border border-white/10" aria-label="Close menu" onClick={onClose}>
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 mx-auto my-auto"><path fillRule="evenodd" d="M5.47 5.47a.75.75 0 0 1 1.06 0L12 10.94l5.47-5.47a.75.75 0 1 1 1.06 1.06L13.06 12l5.47 5.47a.75.75 0 1 1-1.06 1.06L12 13.06l-5.47 5.47a.75.75 0 0 1-1.06-1.06L10.94 12 5.47 6.53a.75.75 0 0 1 0-1.06Z" clipRule="evenodd"/></svg>
            </button>
            <div className="mt-10 text-sm font-semibold text-amber-300/80 px-2">Navigation</div>
            <nav className="mt-2 space-y-1" onClick={onClose}>
              {navItems.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${
                      isActive ? 'bg-amber-500/15 text-amber-200 ring-1 ring-amber-400/20' : 'text-white/80 hover:text-amber-200 hover:bg-amber-500/10'
                    }`
                  }
                >
                  {item.icon === 'dashboard' && <GoldCoin className="w-4 h-4" />}
                  {item.icon === 'chart' && (
                    <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor" aria-hidden>
                      <path d="M4 5.75a.75.75 0 0 1 .75-.75h.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-.5A.75.75 0 0 1 4 18.25V5.75Zm5 4a.75.75 0 0 1 .75-.75h.5a.75.75 0 0 1 .75.75v8.5a.75.75 0 0 1-.75.75h-.5A.75.75 0 0 1 9 18.25v-8.5Zm5-3a.75.75 0 0 1 .75-.75h.5a.75.75 0 0 1 .75.75v11.5a.75.75 0 0 1-.75.75h-.5a.75.75 0 0 1-.75-.75V6.75Zm5 6a.75.75 0 0 1 .75-.75h.5a.75.75 0 0 1 .75.75v5.5a.75.75 0 0 1-.75.75h-.5a.75.75 0 0 1-.75-.75v-5.5Z"/>
                    </svg>
                  )}
                  {item.icon === 'calc' && (
                    <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor" aria-hidden>
                      <path d="M6 3.75h12A2.25 2.25 0 0 1 20.25 6v12A2.25 2.25 0 0 1 18 20.25H6A2.25 2.25 0 0 1 3.75 18V6A2.25 2.25 0 0 1 6 3.75Zm1.5 2.5h9v3h-9v-3Zm0 6h2v2h-2v-2Zm0 3.5h2v2h-2v-2Zm3.5-3.5h2v2h-2v-2Zm0 3.5h2v2h-2v-2Zm3.5-3.5h2v2h-2v-2Zm0 3.5h2v2h-2v-2Z"/>
                    </svg>
                  )}
                  {item.icon === 'info' && (
                    <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor" aria-hidden>
                      <path d="M12 2.25a9.75 9.75 0 1 0 0 19.5 9.75 9.75 0 0 0 0-19.5Zm.75 5.5a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0Zm-2 3a.75.75 0 0 1 .75-.75h1a.75.75 0 0 1 .75.75v6.25h1a.75.75 0 0 1 0 1.5h-4a.75.75 0 0 1 0-1.5h1v-5.5h-.5a.75.75 0 0 1-.75-.75Z"/>
                    </svg>
                  )}
                  <span>{item.label}</span>
                </NavLink>
              ))}
            </nav>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}

export default Sidebar


