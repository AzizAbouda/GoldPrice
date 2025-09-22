type GoldCoinProps = {
  className?: string
}

function GoldCoin({ className }: GoldCoinProps) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden>
      <defs>
        <radialGradient id="gc_grad" cx="50%" cy="35%" r="60%">
          <stop offset="0%" stopColor="#ffe8a3" />
          <stop offset="45%" stopColor="#f5b942" />
          <stop offset="100%" stopColor="#b97a07" />
        </radialGradient>
        <linearGradient id="gc_shine" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#fff" stopOpacity="0" />
          <stop offset="40%" stopColor="#fff" stopOpacity=".35" />
          <stop offset="60%" stopColor="#fff" stopOpacity=".1" />
          <stop offset="100%" stopColor="#fff" stopOpacity="0" />
        </linearGradient>
      </defs>
      <circle cx="12" cy="12" r="10" fill="url(#gc_grad)" />
      <circle cx="12" cy="12" r="8.2" fill="none" stroke="#fff" strokeOpacity=".18" strokeWidth="1.2" />
      {/* shimmer */}
      <g style={{ mixBlendMode: 'screen' }}>
        <rect x="-18" y="-2" width="12" height="28" fill="url(#gc_shine)" className="gold-shine" />
      </g>
    </svg>
  )
}

export default GoldCoin


