import axios from 'axios'
import type { HistoryPoint, ForecastPoint, Metrics } from '../types/api'
import { USE_MOCK } from '../config'

function daysAgo(n: number): string {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().slice(0, 10)
}

function genHistory(len = 120): HistoryPoint[] {
  const base = 2400
  let price = base
  const arr: HistoryPoint[] = []
  for (let i = len - 1; i >= 0; i--) {
    price += (Math.random() - 0.5) * 10
    arr.push({ date: daysAgo(i), price: Math.round(price * 100) / 100 })
  }
  return arr
}

function genForecast(startPrice: number, horizon = 56): ForecastPoint[] {
  const arr: ForecastPoint[] = []
  let price = startPrice
  for (let i = 1; i <= horizon; i++) {
    const drift = 0.8
    price += drift + (Math.random() - 0.5) * 6
    const ci = 18 + Math.sqrt(i)
    const d = new Date()
    d.setDate(d.getDate() + i)
    arr.push({
      date: d.toISOString().slice(0, 10),
      price: Math.round(price * 100) / 100,
      lower: Math.round((price - ci) * 100) / 100,
      upper: Math.round((price + ci) * 100) / 100,
    })
  }
  return arr
}

export async function fetchHistory(): Promise<HistoryPoint[]> {
  if (USE_MOCK) return genHistory(150)
  const res = await axios.get<HistoryPoint[]>('/api/history')
  return res.data
}

export async function fetchForecast(): Promise<ForecastPoint[]> {
  if (USE_MOCK) {
    const hist = genHistory(150)
    return genForecast(hist[hist.length - 1].price)
  }
  const res = await axios.get<ForecastPoint[]>('/api/forecast')
  return res.data
}

export async function fetchMetrics(): Promise<Metrics> {
  if (USE_MOCK) {
    return {
      rmse: 12.8,
      mape: 1.9,
      lastTrainDate: daysAgo(1),
      model: 'LinearRegression-7lag (mock)',
    }
  }
  const res = await axios.get<Metrics>('/api/metrics')
  return res.data
}


