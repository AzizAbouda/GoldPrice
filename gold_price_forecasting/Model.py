# gold_price_linear_regression_annotated.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import math

# -----------------------------
# Metrics
# -----------------------------
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# -----------------------------
# Step 1: Load data
# -----------------------------
DATA_PATH = "gold_price_dataset.csv"
DATE_COL = "date"
PRICE_COL = "gold_price"

df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df.set_index(DATE_COL)

# Convert price column to numeric and remove non-numeric rows
df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors='coerce')
df = df.dropna(subset=[PRICE_COL])

print("Data loaded and cleaned:", df.shape)

# -----------------------------
# Step 2: Create lag features
# -----------------------------
LAGS = 7  # use past 7 days
def create_lag_features(df, col, lags=LAGS):
    X, y = [], []
    prices = df[col].values
    for i in range(lags, len(prices)):
        X.append(prices[i-lags:i])
        y.append(prices[i])
    return np.array(X), np.array(y)

X, y = create_lag_features(df, PRICE_COL, LAGS)

# -----------------------------
# Step 3: Train/Test split
# -----------------------------
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = df.index[LAGS + split_idx:]  # corresponding dates for test set

# -----------------------------
# Step 4: Train Linear Regression
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Linear Regression RMSE:", rmse(y_test, preds))
print("Linear Regression MAPE:", mape(y_test, preds))

# -----------------------------
# Step 5: Plot predictions with annotations
# -----------------------------
plot_points = 200
y_true_plot = y_test[-plot_points:]
preds_plot = preds[-plot_points:]
dates_plot = dates_test[-plot_points:]

plt.figure(figsize=(14,6))
plt.plot(dates_plot, y_true_plot, label="Actual Price", color='blue')
plt.plot(dates_plot, preds_plot, label="Predicted Price", color='orange')

# Annotate max and min actual prices
max_idx = y_true_plot.argmax()
min_idx = y_true_plot.argmin()
plt.annotate(f'Max: {y_true_plot[max_idx]:.2f}', 
             xy=(dates_plot[max_idx], y_true_plot[max_idx]), 
             xytext=(dates_plot[max_idx], y_true_plot[max_idx]+20),
             arrowprops=dict(facecolor='green', arrowstyle='->'),
             fontsize=10, color='green')

plt.annotate(f'Min: {y_true_plot[min_idx]:.2f}', 
             xy=(dates_plot[min_idx], y_true_plot[min_idx]), 
             xytext=(dates_plot[min_idx], y_true_plot[min_idx]-30),
             arrowprops=dict(facecolor='red', arrowstyle='->'),
             fontsize=10, color='red')

plt.title("Gold Price Forecast vs Actual (Last 200 Days)")
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# -----------------------------
# Step 6: Predict next 2 months (approx. 8 weeks)
# -----------------------------
future_weeks = 8
days_per_week = 7
future_days = future_weeks * days_per_week

# Start with the last LAGS days from the dataset
last_lags = df[PRICE_COL].values[-LAGS:].tolist()
future_preds = []

for _ in range(future_days):
    x_input = np.array(last_lags[-LAGS:]).reshape(1, -1)
    next_pred = model.predict(x_input)[0]
    future_preds.append(next_pred)
    last_lags.append(next_pred)

# Generate weekly x-axis labels
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days)
weekly_labels = future_dates[::days_per_week]

# Aggregate predictions to weekly (average)
weekly_preds = [np.mean(future_preds[i*days_per_week:(i+1)*days_per_week]) for i in range(future_weeks)]

# -----------------------------
# Step 7: Plot future 2-month forecast
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(weekly_labels, weekly_preds, marker='o', color='purple', label='Predicted Price (Weekly Avg)')
plt.title("Gold Price Forecast Next 2 Months (Weekly)")
plt.xlabel("Week")
plt.ylabel("Gold Price (USD)")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.show()
