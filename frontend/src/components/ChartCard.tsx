import { PropsWithChildren } from 'react'
import { motion } from 'framer-motion'

type ChartCardProps = PropsWithChildren<{
  title: string
  subtitle?: string
  footer?: string
}>

function ChartCard({ title, subtitle, footer, children }: ChartCardProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="rounded-xl border border-amber-500/20 bg-gradient-to-br from-slate-900/60 to-slate-900/30 backdrop-blur shadow-[0_10px_40px_-15px_rgba(0,0,0,0.6)] hover:shadow-[0_20px_48px_-12px_rgba(0,0,0,0.7)] transition-shadow ring-1 ring-amber-400/10"
    >
      <div className="p-4 border-b border-white/10">
        <h3 className="text-base md:text-lg font-semibold text-amber-200">{title}</h3>
        {subtitle && <p className="text-xs text-white/70 mt-0.5">{subtitle}</p>}
      </div>
      <div className="p-4">
        {children}
      </div>
      {footer && (
        <div className="px-4 py-3 text-xs text-white/60 border-t border-white/10">{footer}</div>
      )}
    </motion.section>
  )
}

export default ChartCard


