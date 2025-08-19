from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'gold_price_dataset.csv'

app = FastAPI(title="Gold Price Estimator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PricePoint(BaseModel):
    date: str
    price: float


class ForecastPoint(PricePoint):
    lower: Optional[float] = None
    upper: Optional[float] = None


class Metrics(BaseModel):
    rmse: float
    mape: float
    lastTrainDate: Optional[str] = None
    model: Optional[str] = None


def load_csv() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.dropna(subset=['gold_price'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


@app.get("/api/history", response_model=List[PricePoint])
def history():
    df = load_csv()
    # Return last 180 points for a manageable chart
    tail = df.tail(180)
    return [PricePoint(date=d.strftime('%Y-%m-%d'), price=float(p)) for d, p in zip(tail['date'], tail['gold_price'])]


@app.get("/api/forecast", response_model=List[ForecastPoint])
def forecast():
    df = load_csv()
    # naive forecast: moving average + simple CI band using std of residuals
    series = df['gold_price'].astype(float).values
    window = 7
    ma = pd.Series(series).rolling(window).mean().iloc[-1]
    std = float(pd.Series(series).diff().dropna().std() or 0.0)

    horizon = 56  # 8 weeks
    base = ma if not np.isnan(ma) else float(series[-1])
    # gentle trend: small drift using last 30d slope
    last_n = min(30, len(series) - 1)
    if last_n > 1:
        x = np.arange(last_n)
        y = series[-last_n:]
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    results: List[ForecastPoint] = []
    for t in range(1, horizon + 1):
        pred = base + slope * (t / window)
        ci = 1.96 * std * np.sqrt(t / window)
        results.append(
            ForecastPoint(
                date=(df['date'].iloc[-1] + pd.Timedelta(days=t)).strftime('%Y-%m-%d'),
                price=float(pred),
                lower=float(pred - ci),
                upper=float(pred + ci),
            )
        )
    return results


@app.get("/api/metrics", response_model=Metrics)
def metrics():
    df = load_csv()
    # minimal placeholder metrics; plug in from your model if available
    rmse = float(pd.Series(df['gold_price']).diff().dropna().abs().mean() or 0.0)
    mape = float((rmse / (df['gold_price'].mean() or 1.0)) * 100)
    return Metrics(
        rmse=rmse,
        mape=mape,
        lastTrainDate=df['date'].max().strftime('%Y-%m-%d'),
        model="LinearRegression-7lag (placeholder)",
    )


