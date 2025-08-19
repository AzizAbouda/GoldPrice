import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Bar, BarChart } from 'recharts'
import ChartCard from '../components/ChartCard'
import KPIStat from '../components/KPIStat'
import { fetchForecast, fetchHistory } from '../services/api'

type PricePoint = { date: string; price: number; predicted?: number }

function Dashboard() {
  const [history, setHistory] = useState<PricePoint[]>([])
  const [forecast, setForecast] = useState<PricePoint[]>([])

  useEffect(() => {
    fetchHistory().then(setHistory).catch(() => setHistory([]))
    fetchForecast().then(setForecast).catch(() => setForecast([]))
  }, [])

  const latest = history.at(-1)
  const next = forecast[0]

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPIStat label="Latest Price" value={latest ? `$${latest.price.toFixed(2)}` : '—'} />
        <KPIStat label="Next Predicted" value={next ? `$${next.price.toFixed(2)}` : '—'} />
        <KPIStat label="Trend" value={forecast.length ? 'Uptrend' : '—'} delta={forecast.length ? '+1.8% weekly' : undefined} />
      </div>

      <ChartCard title="Historical vs Predicted Prices" subtitle="Recent actuals with model overlay">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history.map((h, i) => ({ ...h, predicted: forecast[i]?.price }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f59e0b22" />
              <XAxis dataKey="date" stroke="#94a3b8" hide />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(245,158,11,0.2)' }} />
              <Line type="monotone" dataKey="price" stroke="#60a5fa" dot={false} strokeWidth={2} name="Actual" />
              <Line type="monotone" dataKey="predicted" stroke="#f59e0b" dot={false} strokeWidth={2} name="Predicted" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      <ChartCard title="Weekly Changes" subtitle="Bar chart of weekly price deltas">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={history.map((p, i, arr) => ({ date: p.date, delta: i === 0 ? 0 : p.price - arr[i-1].price }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f59e0b22" />
              <XAxis dataKey="date" stroke="#94a3b8" hide />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(245,158,11,0.2)' }} />
              <Bar dataKey="delta" fill="#f59e0b" name="Δ Price" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
    </div>
  )
}

export default Dashboard


