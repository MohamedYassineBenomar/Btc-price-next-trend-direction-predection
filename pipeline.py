"""BTC price-direction forecast pipeline.

Steps:
  1. Fetch all-time BTC-USD daily history from Yahoo Finance, resample to
     monthly average closing price.
  2. Hold out the last 12 months as a blind test set.
  3. Train Prophet on data strictly before the test window.
  4. Predict the test window, score MAE / RMSE / MAPE / directional accuracy.
  5. Re-train on the full history and project 6 months forward.
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
HOLDOUT_MONTHS = 12
FUTURE_MONTHS = 6


def fetch_history() -> pd.DataFrame:
    print(f"[1/5] Fetching {TICKER} all-time daily history → monthly average…")
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

    # Resample to monthly average closing price (month-start labels).
    monthly = (
        df.set_index("ds")["y"]
          .resample("MS")
          .mean()
          .dropna()
          .reset_index()
    )
    # Drop the most recent month if it's still partial — only keep complete months
    # whose close-month boundary has passed. We compare the next month-start to today.
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    last_month_start = monthly["ds"].iloc[-1]
    next_month_start = (last_month_start + pd.offsets.MonthBegin(1))
    if next_month_start > today:
        monthly = monthly.iloc[:-1].reset_index(drop=True)

    monthly.to_csv(DATA_DIR / "btc_history.csv", index=False)
    print(f"      → {len(monthly):,} monthly rows from {monthly['ds'].min().date()} to {monthly['ds'].max().date()}")
    return monthly


def make_model() -> Prophet:
    # BTC spans 4+ orders of magnitude over 11 years, so we model in log-space
    # and exponentiate the predictions back. With monthly data we don't have
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
    train = df.iloc[:-HOLDOUT_MONTHS].copy()
    test = df.iloc[-HOLDOUT_MONTHS:].copy()

    print(f"[2/5] Backtest split: train {len(train)} months  ·  test {len(test)} months (>{train['ds'].iloc[-1].date()})")
    print(f"[3/5] Training Prophet (log-space) on training set…")
    model = fit_log(train)

    future = model.make_future_dataframe(periods=HOLDOUT_MONTHS + 3, freq="MS")
    forecast = predict_exp(model, future)

    merged = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(test, on="ds", how="inner")
    err = merged["y"] - merged["yhat"]
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((err.abs() / merged["y"]).mean() * 100)

    # Directional accuracy: did the model get the month-over-month up/down right?
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
    print(f"[4/5] Re-training on full history (log-space), projecting {FUTURE_MONTHS} months forward…")
    model = fit_log(df)
    future = model.make_future_dataframe(periods=FUTURE_MONTHS, freq="MS")
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
    """Return the BTC monthly average for the 12 months immediately before the
    blind-test window, with `ds` shifted forward by one year so the points
    line up on the same x-axis as the test window. Lets the dashboard plot a
    same-period-last-year overlay against actual + predicted.
    """
    test_start = pd.Timestamp(backtest["ds"].min())
    prior_start = test_start - pd.DateOffset(years=1)
    prior_end = test_start - pd.DateOffset(months=1)
    prior = df[(df["ds"] >= prior_start) & (df["ds"] <= prior_end)].copy()
    prior["ds"] = prior["ds"] + pd.DateOffset(years=1)
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
            "frequency": "monthly_avg",
            "data_start": str(df["ds"].min().date()),
            "data_end": str(df["ds"].max().date()),
            "n_observations": int(len(df)),
            "current_price": float(df["y"].iloc[-1]),
            "previous_close": float(df["y"].iloc[-2]),
            "all_time_high": float(df["y"].max()),
            "all_time_high_date": str(df.loc[df["y"].idxmax(), "ds"].date()),
            "holdout_months": HOLDOUT_MONTHS,
            "future_months": FUTURE_MONTHS,
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
