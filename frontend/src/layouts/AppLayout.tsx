import { Outlet } from 'react-router-dom'
import { useState, useCallback, memo } from 'react'
import Header from '../components/ui/Header.tsx'
import Sidebar from '../components/ui/Sidebar.tsx'

// Reduced number of shapes and simplified animation
const BackgroundShapes = memo(() => (
  <>
    {[...Array(3)].map((_, index) => (
      <div
        key={`shape-${index}`}
        className="absolute pointer-events-none left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2
          border border-amber-500/10 transform-gpu will-change-transform"
        style={{
          width: `${120 + index * 60}px`,
          height: `${120 + index * 60}px`,
          borderRadius: index % 2 === 0 ? '20%' : '50%',
          animation: `floatSlow ${20 + index * 5}s infinite linear`
        }}
      />
    ))}
  </>
))

// Single optimized glowing orb instead of multiple
const GlowingOrb = memo(() => (
  <div
    className="absolute rounded-full blur-md transform-gpu will-change-transform opacity-60"
    style={{
      background: 'radial-gradient(circle, rgba(245,158,11,0.2) 0%, transparent 70%)',
      width: '200px',
      height: '200px',
      animation: 'floatSlow 25s infinite linear'
    }}
  />
))

function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(true)

  const handleMobileClose = useCallback(() => setMobileOpen(false), [])
  const handleMenuClick = useCallback(() => setMobileOpen(true), [])
  const handleCollapse = useCallback((v: boolean) => setCollapsed(v), [])

  return (
    <div className="min-h-dvh bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white relative overflow-hidden">
      <BackgroundShapes />
      <GlowingOrb />

      <div className="relative z-10 flex">
        <Sidebar 
          open={mobileOpen}
          onClose={handleMobileClose}
          collapsed={collapsed}
          onHoverChange={handleCollapse}
        />
        
        <div 
          className={`
            flex-1
            ${collapsed ? 'md:ml-[96px]' : 'md:ml-[260px]'} 
            transition-[margin] duration-300 ease-out 
            flex flex-col min-h-dvh
          `}
        >
          <Header onMenuClick={handleMenuClick} />
          <main className="flex-1 p-4 md:p-6 lg:p-8 relative">
            <div
              className="
                relative z-10 bg-slate-900/50 
                rounded-xl p-4 shadow-xl border border-slate-800/50
                transition-colors duration-200
              "
            >
              <Outlet />
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}

export default memo(AppLayout)


