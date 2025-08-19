import { useEffect, useState } from 'react'
import ChartCard from '../components/ChartCard'
import { fetchMetrics } from '../services/api'

type Metrics = {
  rmse: number
  mape: number
  lastTrainDate?: string
  model?: string
}

function Calculations() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)

  useEffect(() => {
    fetchMetrics().then(setMetrics).catch(() => setMetrics(null))
  }, [])

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <ChartCard title="Model Performance" subtitle="Evaluation on holdout set">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-white/70">RMSE</div>
            <div className="text-3xl font-semibold">{metrics ? metrics.rmse.toFixed(2) : '—'}</div>
          </div>
          <div>
            <div className="text-sm text-white/70">MAPE</div>
            <div className="text-3xl font-semibold">{metrics ? `${metrics.mape.toFixed(2)}%` : '—'}</div>
          </div>
          <div>
            <div className="text-sm text-white/70">Last Train</div>
            <div className="text-lg">{metrics?.lastTrainDate ?? '—'}</div>
          </div>
          <div>
            <div className="text-sm text-white/70">Model</div>
            <div className="text-lg">{metrics?.model ?? '—'}</div>
          </div>
        </div>
      </ChartCard>

      <ChartCard title="Notes" subtitle="How predictions are computed">
        <ul className="list-disc pl-6 space-y-2 text-sm text-white/80">
          <li>Uses a lag-based Linear Regression with 7-day window.</li>
          <li>Short-term forecast generated iteratively with rolling window.</li>
          <li>Confidence intervals derived from residual variance heuristic.</li>
        </ul>
      </ChartCard>
    </div>
  )
}

export default Calculations


