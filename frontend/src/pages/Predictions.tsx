import { useEffect, useState } from 'react'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area } from 'recharts'
import ChartCard from '../components/ChartCard'
import { fetchForecast } from '../services/api'

type ForecastPoint = { date: string; price: number; lower?: number; upper?: number }

function Predictions() {
  const [forecast, setForecast] = useState<ForecastPoint[]>([])

  useEffect(() => {
    fetchForecast().then(setForecast).catch(() => setForecast([]))
  }, [])

  return (
    <div className="space-y-6">
      <ChartCard title="Forecast with Confidence Interval" subtitle="Shaded band shows uncertainty">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={forecast}>
              <defs>
                <linearGradient id="colorCI" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.35}/>
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f59e0b22" />
              <XAxis dataKey="date" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(245,158,11,0.2)' }} />
              <Area type="monotone" dataKey="upper" stroke="none" fill="url(#colorCI)" />
              <Area type="monotone" dataKey="lower" stroke="none" fill="url(#colorCI)" />
              <Line type="monotone" dataKey="price" stroke="#f59e0b" dot={false} strokeWidth={2} name="Predicted" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      <ChartCard title="Short-term Trend" subtitle="Next 14 days">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={forecast.slice(0, 14)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
              <XAxis dataKey="date" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(255,255,255,0.1)' }} />
              <Line type="monotone" dataKey="price" stroke="#60a5fa" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
    </div>
  )
}

export default Predictions


