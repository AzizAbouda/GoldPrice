type KPIStatProps = {
  label: string
  value: string | number
  delta?: string
}

function KPIStat({ label, value, delta }: KPIStatProps) {
  return (
    <div className="rounded-xl border border-amber-500/20 bg-gradient-to-br from-slate-900/60 to-slate-900/30 p-4 ring-1 ring-amber-400/10">
      <div className="text-xs text-amber-200/80">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-amber-100">{value}</div>
      {delta && <div className="mt-1 text-xs text-emerald-400">{delta}</div>}
    </div>
  )
}

export default KPIStat


