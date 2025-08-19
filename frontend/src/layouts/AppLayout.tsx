import { Outlet } from 'react-router-dom'
import { useState } from 'react'
import { motion } from 'framer-motion'
import Header from '../ui/Header.tsx'
import Sidebar from '../ui/Sidebar.tsx'

function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(true)
  return (
    <div className="min-h-dvh bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white relative overflow-hidden">
      {/* Animated gold glow */}
      <motion.div
        aria-hidden
        className="pointer-events-none absolute -top-40 -left-40 h-96 w-96 rounded-full"
        style={{ background: 'radial-gradient(closest-side, rgba(245,158,11,0.15), rgba(245,158,11,0))' }}
        animate={{ x: [0, 120, -40, 0], y: [0, 40, -20, 0] }}
        transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
      />

      <Sidebar variant="desktop" collapsed={collapsed} onHoverChange={(v) => setCollapsed(v)} />
      <Sidebar variant="mobile" open={mobileOpen} onClose={() => setMobileOpen(false)} />
      <div className={`${collapsed ? 'md:pl-[96px]' : 'md:pl-[300px]'} transition-[padding] duration-300 ease-in-out flex flex-col min-h-dvh`}>
        <Header onMenuClick={() => setMobileOpen(true)} />
        <main className="flex-1 p-4 md:p-6 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default AppLayout


