import yfinance as yf
import pandas as pd
from fredapi import Fred

# Your FRED API key
FRED_API_KEY = '8da9a6fd7d447afa5cd8c8edf68e1ebb'

def download_data():
    symbols = {
        'gold_price': 'GC=F',
        'usd_index': 'DX-Y.NYB',
        'vix': '^VIX',
        'oil_price': 'CL=F',
        'interest_rate': '^TNX'
    }

    data_frames = []
    for name, symbol in symbols.items():
        print(f"Downloading {name} data...")
        df = yf.download(symbol, start='2000-01-01', progress=False)[['Close']]
        df = df.rename(columns={'Close': name})
        df = df.reset_index()  # reset index to get Date as column
        df = df.rename(columns={'Date': 'date'}, inplace=True)  # rename Date column to 'date'
        data_frames.append(df)

# Merge all Yahoo Finance dataframes on 'date' using outer join
        combined_df = None
        for df in data_frames:
            if combined_df is None:
                combined_df = df
        else:
            combined_df = combined_df.merge(df, on='date', how='outer')


    # Flatten any MultiIndex columns (fix for merge error)
    if isinstance(combined_df.columns, pd.MultiIndex):
        combined_df.columns = [
            '_'.join(col).strip() if col[1] else col[0] for col in combined_df.columns
        ]
    else:
        combined_df.columns = combined_df.columns.get_level_values(0)

    combined_df = combined_df.sort_values('date').reset_index(drop=True)

    # Download inflation data (monthly) from FRED
    fred = Fred(api_key=FRED_API_KEY)
    print("Downloading inflation (CPI) data from FRED...")
    inflation = fred.get_series('CPIAUCSL')
    inflation = inflation.to_frame(name='inflation')
    inflation.index = pd.to_datetime(inflation.index)

    # Resample monthly CPI to daily by forward filling, then reset index
    inflation_daily = inflation.resample('D').ffill().reset_index()
    inflation_daily = inflation_daily.rename(columns={'index': 'date'})

    print("After reset_index, combined_df columns:", combined_df.columns)
    print("inflation_daily columns:", inflation_daily.columns)

    # Merge inflation with combined data on 'date'
    combined_df = combined_df.merge(inflation_daily, on='date', how='left')

    # Set 'date' as index for time series analysis
    combined_df = combined_df.set_index('date')

    # Fill missing values forward then backward
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.fillna(method='bfill', inplace=True)

    # Rename columns to match model expectations
    rename_map = {
        'gold_price_GC=F': 'gold_price',
        'usd_index_DX-Y.NYB': 'usd_index',
        'vix_^VIX': 'vix',
        'oil_price_CL=F': 'oil_price',
        'interest_rate_^TNX': 'interest_rate'
    }
    combined_df.rename(columns=rename_map, inplace=True)

    # Save to CSV
    combined_df.to_csv('gold_price_dataset.csv', index=False)
    print("Data saved to gold_price_dataset.csv")

if __name__ == '__main__':
    download_data()
