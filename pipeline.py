"""BTC price-direction forecast pipeline.

Steps:
  1. Fetch the last ~2 years of BTC-USD hourly closes from Yahoo Finance
     (yfinance caps hourly history at 730 days, so we can't go further).
  2. Hold out the last 30 days (720 hourly points) as a blind test set.
  3. Train Prophet on data strictly before the test window.
  4. Predict the test window, score MAE / RMSE / MAPE / directional accuracy.
  5. Re-train on the full history and project 30 days (720 hours) forward.
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
HOLDOUT_HOURS = 24 * 30    # 30 days = 720 hours
FUTURE_HOURS = 24 * 30     # 30 days forward


def fetch_history() -> pd.DataFrame:
    print(f"[1/5] Fetching {TICKER} hourly history (last 730 days, yfinance limit)…")
    df = yf.download(
        TICKER,
        period="730d",
        interval="1h",
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        sys.exit("ERROR: yfinance returned no data — check network access.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    # yfinance's hourly index column can be named "Datetime" or "index"
    ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df[[ts_col, "Close"]].rename(columns={ts_col: "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df = df.dropna().sort_values("ds").reset_index(drop=True)

    df.to_csv(DATA_DIR / "btc_history.csv", index=False)
    print(f"      → {len(df):,} hourly rows from {df['ds'].min()} to {df['ds'].max()}")
    return df


def make_model() -> Prophet:
    # Hourly data over ~2 years — daily and weekly seasonality become
    # meaningful (intraday cycles, weekend effects). Log-space keeps
    # proportional moves consistent. Yearly seasonality off — we don't have
    # enough years for the yearly Fourier basis to mean anything.
    return Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.1,
        changepoint_range=0.95,
        seasonality_prior_scale=8.0,
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
    train = df.iloc[:-HOLDOUT_HOURS].copy()
    test = df.iloc[-HOLDOUT_HOURS:].copy()

    print(f"[2/5] Backtest split: train {len(train):,} hours  ·  test {len(test):,} hours (>{train['ds'].iloc[-1]})")
    print(f"[3/5] Training Prophet (log-space) on training set…")
    model = fit_log(train)

    future = model.make_future_dataframe(periods=HOLDOUT_HOURS + 24, freq="h")
    forecast = predict_exp(model, future)

    merged = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(test, on="ds", how="inner")
    err = merged["y"] - merged["yhat"]
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((err.abs() / merged["y"]).mean() * 100)

    # Directional accuracy: did the model get the hour-over-hour up/down right?
    actual_dir = (merged["y"].diff() > 0).astype(int)
    pred_dir = (merged["yhat"].diff() > 0).astype(int)
    direction_match = (actual_dir == pred_dir).iloc[1:]
    dir_acc = float(direction_match.mean() * 100)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": dir_acc,
        "test_start": str(merged["ds"].min()),
        "test_end": str(merged["ds"].max()),
        "n_test_points": int(len(merged)),
        "train_size": int(len(train)),
    }
    print(
        f"      → MAE ${mae:,.0f}  ·  RMSE ${rmse:,.0f}  ·  MAPE {mape:.2f}%  ·  Dir-acc {dir_acc:.2f}%"
    )
    return merged, metrics


def forward_forecast(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[4/5] Re-training on full history (log-space), projecting {FUTURE_HOURS} hours (~{FUTURE_HOURS // 24} days) forward…")
    model = fit_log(df)
    future = model.make_future_dataframe(periods=FUTURE_HOURS, freq="h")
    forecast = predict_exp(model, future)
    last = df["ds"].max()
    forward = forecast[forecast["ds"] > last][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    return forward


def to_record(df: pd.DataFrame, cols: list[str], hourly: bool = False) -> list[dict]:
    out = df[cols].copy()
    if "ds" in out.columns:
        fmt = "%Y-%m-%dT%H:00" if hourly else "%Y-%m-%d"
        out["ds"] = pd.to_datetime(out["ds"]).dt.strftime(fmt)
    for c in cols:
        if c == "ds":
            continue
        out[c] = out[c].astype(float).round(2)
    return out.to_dict(orient="records")


def downsample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily mean of an hourly series — used so the all-time display chart
    sends ~730 points instead of ~17k.
    """
    daily = (
        df.set_index("ds")["y"]
          .resample("1D")
          .mean()
          .dropna()
          .reset_index()
    )
    return daily


def prior_year_overlay(df: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    """Return the BTC hourly closes for the same calendar window one year
    before the blind-test window, with `ds` shifted forward by 365 days so
    the points line up on the test window's x-axis. Returns empty if the
    history doesn't reach back a full year before the test start.
    """
    test_start = pd.Timestamp(backtest["ds"].min())
    test_end = pd.Timestamp(backtest["ds"].max())
    shift = pd.Timedelta(days=365)
    prior_start = test_start - shift
    prior_end = test_end - shift
    if prior_start < df["ds"].min():
        return pd.DataFrame(columns=["ds", "y_prior"])
    prior = df[(df["ds"] >= prior_start) & (df["ds"] <= prior_end)].copy()
    prior["ds"] = prior["ds"] + shift
    prior = prior.rename(columns={"y": "y_prior"})
    return prior[["ds", "y_prior"]].reset_index(drop=True)


def main() -> None:
    df = fetch_history()
    backtest, metrics = blind_backtest(df)
    forward = forward_forecast(df)
    prior = prior_year_overlay(df, backtest)

    print("[5/5] Writing dashboard/data.json…")
    daily_history = downsample_daily(df)
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ticker": TICKER,
            "frequency": "hourly",
            "data_start": str(df["ds"].min()),
            "data_end": str(df["ds"].max()),
            "n_observations": int(len(df)),
            "current_price": float(df["y"].iloc[-1]),
            "previous_close": float(df["y"].iloc[-2]),
            "all_time_high": float(df["y"].max()),
            "all_time_high_date": str(df.loc[df["y"].idxmax(), "ds"]),
            "holdout_hours": HOLDOUT_HOURS,
            "future_hours": FUTURE_HOURS,
        },
        # Daily means for the all-time chart (compact). Charts that need
        # hourly resolution get hourly data in their own keys below.
        "history": to_record(daily_history, ["ds", "y"]),
        "backtest": {
            "predictions": to_record(backtest, ["ds", "y", "yhat", "yhat_lower", "yhat_upper"], hourly=True),
            "prior_year": to_record(prior, ["ds", "y_prior"], hourly=True),
            "metrics": metrics,
        },
        "forecast": to_record(forward, ["ds", "yhat", "yhat_lower", "yhat_upper"], hourly=True),
    }

    out_path = DASH_DIR / "data.json"
    with out_path.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    print(f"      → {out_path} ({size_kb:.1f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
