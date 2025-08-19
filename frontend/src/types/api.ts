export type HistoryPoint = { date: string; price: number }
export type ForecastPoint = { date: string; price: number; lower?: number; upper?: number }
export type Metrics = { rmse: number; mape: number; lastTrainDate?: string; model?: string }


