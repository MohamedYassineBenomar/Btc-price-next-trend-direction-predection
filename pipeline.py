"""BTC price-direction forecast pipeline.

Steps:
  1. Fetch all-time BTC-USD daily history from Yahoo Finance, resample to
     15-day average closing price (bi-monthly buckets).
  2. Hold out the last 24 buckets (~360 days, ≈ 1 year) as a blind test set.
  3. Train Prophet on data strictly before the test window.
  4. Predict the test window, score MAE / RMSE / MAPE / directional accuracy.
  5. Re-train on the full history and project 12 buckets (~180 days) forward.
  6. Export everything the dashboard needs as dashboard/data.json.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet

# Silence the chatty libraries
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
for noisy in ("prophet.plot", "fbprophet"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DASH_DIR = ROOT / "dashboard"
DATA_DIR.mkdir(exist_ok=True)
DASH_DIR.mkdir(exist_ok=True)

TICKER = "BTC-USD"
BUCKET_DAYS = 15
HOLDOUT_BUCKETS = 24      # ≈ 360 days (one year of 15-day buckets)
FUTURE_BUCKETS = 12       # ≈ 180 days forward


def fetch_history() -> pd.DataFrame:
    print(f"[1/5] Fetching {TICKER} all-time daily history → {BUCKET_DAYS}-day average…")
    df = yf.download(
        TICKER,
        start="2014-09-17",
        end=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        sys.exit("ERROR: yfinance returned no data — check network access.")

    # yfinance can return a MultiIndex on columns; flatten if needed.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df = df.dropna().sort_values("ds").reset_index(drop=True)

    # Resample to 15-day average closing price.
    bucketed = (
        df.set_index("ds")["y"]
          .resample(f"{BUCKET_DAYS}D")
          .mean()
          .dropna()
          .reset_index()
    )

    # Drop the most recent bucket if it spans into the future (incomplete).
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    last_bucket = bucketed["ds"].iloc[-1]
    next_bucket = last_bucket + pd.Timedelta(days=BUCKET_DAYS)
    if next_bucket > today:
        bucketed = bucketed.iloc[:-1].reset_index(drop=True)

    bucketed.to_csv(DATA_DIR / "btc_history.csv", index=False)
    print(f"      → {len(bucketed):,} {BUCKET_DAYS}-day rows from {bucketed['ds'].min().date()} to {bucketed['ds'].max().date()}")
    return bucketed


def make_model() -> Prophet:
    # BTC spans 4+ orders of magnitude over 11 years, so we model in log-space
    # and exponentiate the predictions back. With 15-day buckets we don't have
    # daily/weekly seasonality to model — only the yearly cycle is meaningful.
    return Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=True,
        changepoint_prior_scale=0.15,
        changepoint_range=0.95,
        seasonality_prior_scale=5.0,
        interval_width=0.80,
    )


def fit_log(df: pd.DataFrame) -> Prophet:
    """Fit Prophet on log(price). Prices live on a multiplicative scale, so
    fitting in log-space keeps proportional moves consistent and stops the
    trend from being dominated by the most recent absolute-dollar moves.
    """
    work = df.copy()
    work["y"] = np.log(work["y"])
    model = make_model()
    model.fit(work)
    return model


def predict_exp(model: Prophet, future: pd.DataFrame) -> pd.DataFrame:
    forecast = model.predict(future)
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        forecast[col] = np.exp(forecast[col])
    return forecast


def blind_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = df.iloc[:-HOLDOUT_BUCKETS].copy()
    test = df.iloc[-HOLDOUT_BUCKETS:].copy()

    print(f"[2/5] Backtest split: train {len(train)} buckets  ·  test {len(test)} buckets (>{train['ds'].iloc[-1].date()})")
    print(f"[3/5] Training Prophet (log-space) on training set…")
    model = fit_log(train)

    future = model.make_future_dataframe(periods=HOLDOUT_BUCKETS + 4, freq=f"{BUCKET_DAYS}D")
    forecast = predict_exp(model, future)

    merged = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(test, on="ds", how="inner")
    err = merged["y"] - merged["yhat"]
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((err.abs() / merged["y"]).mean() * 100)

    # Directional accuracy: did the model get the bucket-over-bucket up/down right?
    actual_dir = (merged["y"].diff() > 0).astype(int)
    pred_dir = (merged["yhat"].diff() > 0).astype(int)
    direction_match = (actual_dir == pred_dir).iloc[1:]
    dir_acc = float(direction_match.mean() * 100)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": dir_acc,
        "test_start": str(merged["ds"].min().date()),
        "test_end": str(merged["ds"].max().date()),
        "n_test_points": int(len(merged)),
        "train_size": int(len(train)),
    }
    print(
        f"      → MAE ${mae:,.0f}  ·  RMSE ${rmse:,.0f}  ·  MAPE {mape:.2f}%  ·  Dir-acc {dir_acc:.2f}%"
    )
    return merged, metrics


def forward_forecast(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[4/5] Re-training on full history (log-space), projecting {FUTURE_BUCKETS} buckets (~{FUTURE_BUCKETS * BUCKET_DAYS} days) forward…")
    model = fit_log(df)
    future = model.make_future_dataframe(periods=FUTURE_BUCKETS, freq=f"{BUCKET_DAYS}D")
    forecast = predict_exp(model, future)
    last = df["ds"].max()
    forward = forecast[forecast["ds"] > last][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    return forward


def to_record(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = df[cols].copy()
    if "ds" in out.columns:
        out["ds"] = pd.to_datetime(out["ds"]).dt.strftime("%Y-%m-%d")
    for c in cols:
        if c == "ds":
            continue
        out[c] = out[c].astype(float).round(2)
    return out.to_dict(orient="records")


def prior_year_overlay(df: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    """Return the BTC 15-day average for the HOLDOUT_BUCKETS buckets
    immediately before the blind-test window, with `ds` shifted forward by
    HOLDOUT_BUCKETS × BUCKET_DAYS days so the points line up on the same
    x-axis as the test window.
    """
    test_start = pd.Timestamp(backtest["ds"].min())
    shift = pd.Timedelta(days=HOLDOUT_BUCKETS * BUCKET_DAYS)
    prior_start = test_start - shift
    prior_end = test_start - pd.Timedelta(days=BUCKET_DAYS)
    prior = df[(df["ds"] >= prior_start) & (df["ds"] <= prior_end)].copy()
    prior["ds"] = prior["ds"] + shift
    prior = prior.rename(columns={"y": "y_prior"})
    return prior[["ds", "y_prior"]]


def main() -> None:
    df = fetch_history()
    backtest, metrics = blind_backtest(df)
    forward = forward_forecast(df)
    prior = prior_year_overlay(df, backtest)

    print("[5/5] Writing dashboard/data.json…")
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ticker": TICKER,
            "frequency": f"{BUCKET_DAYS}d_avg",
            "bucket_days": BUCKET_DAYS,
            "data_start": str(df["ds"].min().date()),
            "data_end": str(df["ds"].max().date()),
            "n_observations": int(len(df)),
            "current_price": float(df["y"].iloc[-1]),
            "previous_close": float(df["y"].iloc[-2]),
            "all_time_high": float(df["y"].max()),
            "all_time_high_date": str(df.loc[df["y"].idxmax(), "ds"].date()),
            "holdout_buckets": HOLDOUT_BUCKETS,
            "future_buckets": FUTURE_BUCKETS,
        },
        "history": to_record(df, ["ds", "y"]),
        "backtest": {
            "predictions": to_record(backtest, ["ds", "y", "yhat", "yhat_lower", "yhat_upper"]),
            "prior_year": to_record(prior, ["ds", "y_prior"]),
            "metrics": metrics,
        },
        "forecast": to_record(forward, ["ds", "yhat", "yhat_lower", "yhat_upper"]),
    }

    out_path = DASH_DIR / "data.json"
    with out_path.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    print(f"      → {out_path} ({size_kb:.1f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
