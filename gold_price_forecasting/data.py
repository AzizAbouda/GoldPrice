import yfinance as yf
import pandas as pd

def download_gold_price():
    symbol = 'GC=F'  # Gold Futures
    print("Downloading gold price data...")
    
    # Download gold price
    df = yf.download(symbol, start='2000-01-01', progress=False)[['Close']]
    df = df.rename(columns={'Close': 'gold_price'})
    df = df.reset_index()  # Reset index to get Date as a column
    df = df.rename(columns={'Date': 'date'})  # Rename Date column to 'date'
    
    # Sort by date just to be safe
    df = df.sort_values('date').reset_index(drop=True)

    # Save to CSV
    df.to_csv('gold_price_dataset.csv', index=False)
    print("Gold price data saved to gold_price_dataset.csv")

if __name__ == '__main__':
    download_gold_price()
