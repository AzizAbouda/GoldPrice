import { motion } from 'framer-motion'

type HeaderProps = { onMenuClick?: () => void }

function Header({ onMenuClick }: HeaderProps) {
  return (
    <motion.header
      initial={{ y: -16, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="sticky top-0 z-10 backdrop-blur supports-[backdrop-filter]:bg-slate-900/70 border-b border-white/5"
    >
      <div className="px-4 md:px-6 lg:px-8 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button className="md:hidden inline-flex items-center justify-center w-9 h-9 rounded-md bg-amber-500/15 hover:bg-amber-500/25 border border-amber-400/20" onClick={onMenuClick} aria-label="Open menu">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M3.75 6.75a.75.75 0 0 1 .75-.75h15a.75.75 0 0 1 0 1.5h-15a.75.75 0 0 1-.75-.75Zm0 10.5a.75.75 0 0 1 .75-.75h15a.75.75 0 0 1 0 1.5h-15a.75.75 0 0 1-.75-.75Zm0-5.25a.75.75 0 0 1 .75-.75h15a.75.75 0 0 1 0 1.5h-15a.75.75 0 0 1-.75-.75Z" clipRule="evenodd"/></svg>
          </button>
          <h1 className="text-lg md:text-xl font-semibold tracking-tight bg-gradient-to-r from-amber-300 via-amber-400 to-yellow-300 bg-clip-text text-transparent">Gold Price Estimator</h1>
        </div>
        <div className="text-xs text-amber-200/80">AI-powered forecasts</div>
      </div>
    </motion.header>
  )
}

export default Header


