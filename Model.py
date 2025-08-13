# gold_forecast_hybrid.py
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Utility metrics
# -----------------------------
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # avoid division by zero
    denom = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

# -----------------------------
# Step 0: configuration
# -----------------------------
DATA_PATH = "gold_price_dataset.csv"  # replace with your CSV
DATE_COL = "date"                           # replace as necessary
PRICE_COL = "gold_price"                    # replace as necessary
EXOG_COLS = ["usd_index", "vix", "oil_price", "interest_rate", "inflation"]  # adapt

# forecasting horizon: how many steps ahead to predict (days)
HORIZON = 7

# how many lags to use for features (in days)
LAGS = 14

# LSTM config
SEQ_LEN = 28  # how many past days fed into LSTM
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 6

# -----------------------------
# Step 1: Load and preprocess
# -----------------------------
def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, PRICE_COL] + EXOG_COLS]
    df = df.set_index(DATE_COL)

    # simple missing value handling: linear interpolation for continuous
    df = df.interpolate(method="time").ffill().bfill()
    return df

df = load_and_preprocess(DATA_PATH)
print("Data loaded:", df.shape)

# -----------------------------
# Step 2: Decompose with STL
# -----------------------------
stl = STL(df[PRICE_COL], period=365, robust=True)  # if daily data; adjust period if weekly
res = stl.fit()
df["trend"] = res.trend
df["seasonal"] = res.seasonal
df["residual"] = res.resid

# optional: plot decomposition (helpful)
def plot_decomposition(df):
    plt.figure(figsize=(12,8))
    plt.subplot(411); plt.plot(df[PRICE_COL]); plt.title("Original")
    plt.subplot(412); plt.plot(df["trend"]); plt.title("Trend")
    plt.subplot(413); plt.plot(df["seasonal"]); plt.title("Seasonal")
    plt.subplot(414); plt.plot(df["residual"]); plt.title("Residual")
    plt.tight_layout()
    plt.show()

# plot_decomposition(df)

# -----------------------------
# Step 3: Create lag features (for XGBoost baseline + static features for LSTM)
# -----------------------------
def create_lag_features(df, cols, lags):
    df_ = df.copy()
    for col in cols:
        for lag in range(1, lags+1):
            df_[f"{col}_lag{lag}"] = df_[col].shift(lag)
    return df_

feature_cols_base = EXOG_COLS + ["trend", "seasonal"]  # features for baseline might include trend/seasonal
df_feats = create_lag_features(df, EXOG_COLS + ["residual"], LAGS)  # create lags for exogs and residuals
df_feats = df_feats.dropna().copy()

# Target: we will forecast the price HORIZON days ahead. For residual modeling target is future residual.
df_feats[f"residual_t_plus_{HORIZON}"] = df_feats["residual"].shift(-HORIZON)
df_feats[f"price_t_plus_{HORIZON}"] = df_feats[PRICE_COL].shift(-HORIZON)
df_feats = df_feats.dropna().copy()

print("Features prepared:", df_feats.shape)

# -----------------------------
# Step 4: Train/Test split (walk-forward / time-based)
# -----------------------------
# use last X% as test or fixed cutoff
split_date = df_feats.index[int(len(df_feats)*0.8)]
train_df = df_feats[df_feats.index <= split_date]
test_df  = df_feats[df_feats.index > split_date]
print("Train:", train_df.shape, "Test:", test_df.shape)

# -----------------------------
# Step 5: Baseline - XGBoost on lag features to predict price directly
# -----------------------------
xgb_features = [c for c in train_df.columns if ("lag" in c) or (c in feature_cols_base)]
# remove any target-like columns
xgb_features = [c for c in xgb_features if f"t_plus" not in c]

X_train_xgb = train_df[xgb_features]
y_train_xgb = train_df[f"price_t_plus_{HORIZON}"]
X_test_xgb  = test_df[xgb_features]
y_test_xgb  = test_df[f"price_t_plus_{HORIZON}"]

xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train_xgb, y_train_xgb,
              eval_set=[(X_test_xgb, y_test_xgb)], early_stopping_rounds=30, verbose=False)

pred_xgb = xgb_model.predict(X_test_xgb)
print("XGBoost RMSE:", rmse(y_test_xgb, pred_xgb), "MAPE:", mape(y_test_xgb, pred_xgb))

# -----------------------------
# Step 6: LSTM to predict residuals (sequence modeling)
# -----------------------------
# We'll build sequences of length SEQ_LEN using exogenous lagged features and residual lags.
# Prepare scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# choose input features for LSTM: exogenous columns (current and lags) + residual lags
lstm_input_cols = []
# Add current exogenous values and their lags up to LAGS
for c in EXOG_COLS:
    lstm_input_cols.append(c)
    for lag in range(1, LAGS+1):
        lstm_input_cols.append(f"{c}_lag{lag}")

# residual lags
for lag in range(1, LAGS+1):
    lstm_input_cols.append(f"residual_lag{lag}")

# ensure columns exist in df_feats
missing = [c for c in lstm_input_cols if c not in df_feats.columns]
if missing:
    raise ValueError("Missing columns for LSTM features: " + str(missing))

# Build sequence datasets
def build_sequences(df_source, seq_len, input_cols, target_col):
    Xs, ys, idxs = [], [], []
    arr = df_source[input_cols].values
    y_arr = df_source[target_col].values
    for i in range(len(df_source) - seq_len - HORIZON + 1):
        Xs.append(arr[i:i+seq_len])
        ys.append(y_arr[i+seq_len-1])  # target aligned to last timestep of the input window
        idxs.append(df_source.index[i+seq_len-1])
    return np.array(Xs), np.array(ys), np.array(idxs)

# Build on the full df_feats so scaler fits on train only
# first split sequences by date
train_seq_source = df_feats[df_feats.index <= split_date]
test_seq_source  = df_feats[df_feats.index > split_date]

X_train_seq, y_train_seq, idx_train = build_sequences(train_seq_source, SEQ_LEN, lstm_input_cols, f"residual_t_plus_{HORIZON}")
X_test_seq,  y_test_seq,  idx_test  = build_sequences(test_seq_source, SEQ_LEN, lstm_input_cols, f"residual_t_plus_{HORIZON}")

print("LSTM seq shapes:", X_train_seq.shape, y_train_seq.shape, X_test_seq.shape, y_test_seq.shape)

# flatten to fit scaler (fit scaler on training data only)
nsamples, nsteps, nfeats = X_train_seq.shape
X_train_flat = X_train_seq.reshape((nsamples * nsteps, nfeats))
scaler_X.fit(X_train_flat)
# transform train/test
X_train_scaled = scaler_X.transform(X_train_flat).reshape((nsamples, nsteps, nfeats))
X_test_flat = X_test_seq.reshape((X_test_seq.shape[0] * X_test_seq.shape[1], X_test_seq.shape[2]))
X_test_scaled = scaler_X.transform(X_test_flat).reshape((X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2]))

# scale y
y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1,1)).ravel()
y_test_scaled  = scaler_y.transform(y_test_seq.reshape(-1,1)).ravel()

# build LSTM model
def build_lstm(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model

lstm_model = build_lstm((SEQ_LEN, nfeats))
es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
history = lstm_model.fit(X_train_scaled, y_train_scaled,
                        validation_data=(X_test_scaled, y_test_scaled),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)

# predict residuals (remember to inverse scale)
pred_residuals_scaled = lstm_model.predict(X_test_scaled).ravel()
pred_residuals = scaler_y.inverse_transform(pred_residuals_scaled.reshape(-1,1)).ravel()

# For comparison, direct residual baseline can be e.g., persistence (use last residual)
# compute actual price predictions: price_pred = trend_at_t+H + seasonal_at_t+H + predicted_residual
# get corresponding trend & seasonal for the forecast horizon point:
# idx_test corresponds to the last timestamp in input window; the target time is idx + HORIZON days
# We'll align indices purely by using df.loc[idx + HORIZON days] if exact date exists. For simplicity, assume daily continuous index.

price_preds_lstm = []
price_trues = []
for i, idx in enumerate(idx_test):
    target_date = idx + pd.Timedelta(days=HORIZON)
    if target_date not in df.index:
        # if missing, skip (this happens near the end); alternatively you can reindex with fill
        price_preds_lstm.append(np.nan)
        price_trues.append(np.nan)
        continue
    trend_t = df.loc[target_date, "trend"]
    seasonal_t = df.loc[target_date, "seasonal"]
    pred_resid = pred_residuals[i]
    pred_price = trend_t + seasonal_t + pred_resid
    true_price = df.loc[target_date, PRICE_COL]
    price_preds_lstm.append(pred_price)
    price_trues.append(true_price)

# filter NaNs
valid_mask = ~np.isnan(price_preds_lstm)
price_preds_lstm = np.array(price_preds_lstm)[valid_mask]
price_trues = np.array(price_trues)[valid_mask]

print("LSTM-based final forecast RMSE:", rmse(price_trues, price_preds_lstm), "MAPE:", mape(price_trues, price_preds_lstm))

# -----------------------------
# Step 7: Compare with XGBoost baseline predictions aligned by date
# -----------------------------
# We earlier predicted on test_df rows: pred_xgb corresponds to test_df index order.
# But the XGBoost target was price_t_plus_HORIZON so alignment is direct.
xgb_pred_series = pd.Series(pred_xgb, index=test_df.index)
xgb_true_series = test_df[f"price_t_plus_{HORIZON}"]

# compute overall XGBoost metrics (already printed), but compare with LSTM on same dates
# For fair comparison, limit to dates where LSTM produced predictions (target_date existed)
# gather xgb preds for same target dates used in LSTM evaluation:
lstm_target_dates = [idx + pd.Timedelta(days=HORIZON) for idx in idx_test[valid_mask]]
# For each target_date, find the corresponding row in test_df that had that target_date as price_t_plus_HORIZON
xgb_preds_for_lstm_dates = []
xgb_trues_for_lstm_dates = []
for td in lstm_target_dates:
    # find row in test_df where index + HORIZON == td
    # equivalently, the row index should be td - HORIZON
    row_idx = td - pd.Timedelta(days=HORIZON)
    if row_idx in test_df.index:
        xgb_preds_for_lstm_dates.append(xgb_pred_series.loc[row_idx])
        xgb_trues_for_lstm_dates.append(xgb_true_series.loc[row_idx])

if xgb_preds_for_lstm_dates:
    print("XGBoost on same dates RMSE:", rmse(xgb_trues_for_lstm_dates, xgb_preds_for_lstm_dates),
          "MAPE:", mape(xgb_trues_for_lstm_dates, xgb_preds_for_lstm_dates))

# -----------------------------
# Step 8: Plot example of true vs preds (last N points)
# -----------------------------
def plot_preds(true, pred, model_name="model"):
    plt.figure(figsize=(10,4))
    plt.plot(true[-120:], label="true")
    plt.plot(pred[-120:], label=f"pred_{model_name}")
    plt.legend()
    plt.title(f"True vs Predicted (last 120 points) - {model_name}")
    plt.show()

plot_preds(price_trues, price_preds_lstm, "LSTM_hybrid")
# for XGBoost:
# we need series aligned:
xgb_aligned_preds = xgb_preds_for_lstm_dates
plot_preds(xgb_trues_for_lstm_dates, xgb_aligned_preds, "XGBoost")

# -----------------------------
# Step 9: Save models (optional)
# -----------------------------
xgb_model.save_model("xgb_price_model.json")
lstm_model.save("lstm_residuals_model.h5")
print("Models saved.")

# -----------------------------
# Notes and next steps
# -----------------------------
print("""
Notes:
- Adjust STL period depending on data frequency and seasonality (365 for daily yearly seasonality).
- You can improve LSTM by adding exogenous features (we included them), attention layers, or using a Temporal Fusion Transformer (TFT) implementation.
- For production: implement walk-forward retraining (rolling window), strong feature selection, and hyperparameter tuning (Optuna).
- Consider ensembling XGBoost and LSTM predictions (weighted average) for improved robustness.
""")
