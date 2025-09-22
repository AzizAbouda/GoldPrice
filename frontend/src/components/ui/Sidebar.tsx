import { NavLink } from 'react-router-dom'
import { memo, useCallback } from 'react'
import GoldCoin from './GoldCoin'

const navItems = [
  { to: '/', label: 'Dashboard', icon: 'dashboard' },
  { to: '/predictions', label: 'Predictions', icon: 'chart' },
  { to: '/calculations', label: 'Calculations', icon: 'calc' },
  { to: '/about', label: 'About', icon: 'info' },
]

const icons = {
  chart: (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
      <path d="M4 5.75v12.5h2V5.75H4zm5 4v8.5h2v-8.5H9zm5-3v11.5h2V6.75h-2zm5 6v5.5h2v-5.5h-2z"/>
    </svg>
  ),
  calc: (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
      <path d="M6 4h12a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2zm1.5 2.5v3h9v-3h-9z"/>
    </svg>
  ),
  info: (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2.25a9.75 9.75 0 100 19.5 9.75 9.75 0 000-19.5zM12 8a1 1 0 110-2 1 1 0 010 2zm-1 3v6h2v-6h-2z"/>
    </svg>
  ),
}

// Memoized NavItem
const NavItem = memo(({ to, label, icon, collapsed, onClick }: {
  to: string
  label: string
  icon: string
  collapsed?: boolean
  onClick?: () => void
}) => {
  const baseClass = `flex items-center gap-3 rounded-md transition-colors duration-150
    ${collapsed ? 'w-10 h-10 justify-center' : 'px-3 py-2'}`

  return (
    <NavLink
      to={to}
      onClick={onClick}
      className={({ isActive }) =>
        `${baseClass} ${
          isActive 
            ? 'bg-amber-500/15 text-amber-200 ring-1 ring-amber-400/20' 
            : 'text-white/80 hover:text-amber-200 hover:bg-amber-500/10'
        }`
      }
    >
      {icon === 'dashboard' ? <GoldCoin className="w-4 h-4" /> : icons[icon as keyof typeof icons]}
      {!collapsed && <span className="whitespace-nowrap">{label}</span>}
    </NavLink>
  )
})

// Main Sidebar
type SidebarProps = {
  open?: boolean
  onClose?: () => void
  collapsed?: boolean
  onHoverChange?: (collapsed: boolean) => void
}

const Sidebar = memo(function Sidebar({ 
  open = false, 
  onClose,
  collapsed = false, 
  onHoverChange 
}: SidebarProps) {
  const handleMouseEnter = useCallback(() => onHoverChange?.(false), [onHoverChange])
  const handleMouseLeave = useCallback(() => onHoverChange?.(true), [onHoverChange])

  return (
    <>
      {/* Desktop Sidebar */}
      <aside className="hidden md:block fixed top-0 left-0 h-full z-20">
        <div
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          className={`
            m-6 h-[calc(100%-3rem)]
            rounded-2xl border border-amber-500/20 
            bg-slate-900/90 backdrop-blur-sm 
            shadow-lg p-4
            transition-[width] duration-200 ease-out
            transform-gpu
            ${collapsed ? 'w-[72px]' : 'w-[260px]'}
          `}
        >
          <nav className="mt-2 space-y-1">
            {navItems.map((item) => (
              <NavItem key={item.to} {...item} collapsed={collapsed} />
            ))}
          </nav>
        </div>
      </aside>

      {/* Mobile Sidebar */}
      <div 
        className={`
          md:hidden fixed inset-0 z-50 
          transition-opacity duration-200 ease-out
          ${open ? 'opacity-100 visible' : 'opacity-0 invisible'}
        `}
      >
        <div 
          className="absolute inset-0 bg-black/40" 
          onClick={onClose}
        />
        <aside 
          className={`
            fixed inset-y-0 left-0
            w-[80%] max-w-[300px] 
            p-4 border-r border-amber-500/20 
            bg-slate-900/90 backdrop-blur-sm
            shadow-lg
            transition-transform duration-200 ease-out
            transform-gpu
            ${open ? 'translate-x-0' : '-translate-x-full'}
          `}
        >
          <nav className="mt-10 space-y-1">
            {navItems.map((item) => (
              <NavItem key={item.to} {...item} onClick={onClose} />
            ))}
          </nav>
        </aside>
      </div>
    </>
  )
})

export default Sidebar
