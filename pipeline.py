"""BTC price-direction forecast pipeline.

Steps:
  1. Fetch all-time BTC/USD hourly closes from Bitstamp (since 2014-09-17).
     Bitstamp's free OHLC endpoint goes back to 2011, so we get the full
     11+ year history at hourly resolution. Binance only has ~9 years
     (BTCUSDT launched 2017-08), and yfinance caps hourly at 730 days.
  2. Hold out the last 30 days (720 hourly bars) as a blind test set.
  3. Train Prophet on data strictly before the test window.
  4. Predict the test window, score MAE / RMSE / MAPE / directional accuracy.
  5. Re-train on the full history and project 30 days forward.
  6. Export everything the dashboard needs as dashboard/data.json.

Streamlit Cloud reads the committed dashboard/data.json directly; Prophet
itself doesn't run there. Refresh data by running this script locally and
pushing — the Bitstamp fetcher is incremental, so subsequent runs only
pull bars newer than the cached CSV.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
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

TICKER = "btcusd"
BITSTAMP_OHLC = f"https://www.bitstamp.net/api/v2/ohlc/{TICKER}/"
HISTORY_START = pd.Timestamp("2014-09-17", tz="UTC")  # match the prior daily-data start

HOLDOUT_HOURS = 24 * 30      # 30-day blind test = 720 hours
FUTURE_HOURS = 24 * 30       # 30-day forward forecast = 720 hours


def _cache_path() -> Path:
    return DATA_DIR / "btc_history.csv"


def _bitstamp_chunk(start_s: int) -> list[dict]:
    """One paginated Bitstamp OHLC call. Returns up to 1000 hourly bars
    starting at start_s (seconds, UTC).
    """
    params = {"step": 3600, "limit": 1000, "start": start_s}
    r = requests.get(BITSTAMP_OHLC, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    return payload.get("data", {}).get("ohlc", [])


def fetch_history() -> pd.DataFrame:
    """Fetch all-time hourly BTC/USD closes from Bitstamp, with on-disk cache.

    On first run, paginates from 2014-09-17 to now (~100 calls, ~100k rows).
    Subsequent runs only fetch bars newer than what's already in the CSV.
    """
    cache = _cache_path()
    existing: pd.DataFrame | None = None
    if cache.exists():
        existing = pd.read_csv(cache)
        existing["ds"] = pd.to_datetime(existing["ds"])
        last = existing["ds"].max()
        start_ts = pd.Timestamp(last, tz="UTC") + pd.Timedelta(hours=1)
    else:
        start_ts = HISTORY_START

    end_ts = pd.Timestamp.now(tz="UTC")
    if end_ts <= start_ts:
        if existing is not None:
            print(f"[1/5] Cache is current ({len(existing):,} rows from {existing['ds'].min()} to {existing['ds'].max()})")
            return existing
        raise RuntimeError("No cache and start time is in the future — clock issue?")

    print(f"[1/5] Fetching BTC/USD hourly from Bitstamp (since {start_ts.date()})…")
    rows: list[tuple] = []
    cursor_s = int(start_ts.timestamp())
    end_s = int(end_ts.timestamp())
    last_seen_s = cursor_s

    while cursor_s < end_s:
        try:
            chunk = _bitstamp_chunk(cursor_s)
        except requests.RequestException as e:
            print(f"      ! Bitstamp error: {e} — sleeping 5s and retrying")
            time.sleep(5)
            continue
        if not chunk:
            break
        for bar in chunk:
            ts = int(bar["timestamp"])
            if ts > end_s:
                break
            rows.append((pd.Timestamp(ts, unit="s"), float(bar["close"])))
        new_last = int(chunk[-1]["timestamp"])
        if new_last <= last_seen_s:
            break  # no progress — bail out so we don't loop forever
        last_seen_s = new_last
        cursor_s = new_last + 3600  # next bar after the last close
        time.sleep(0.05)
        if len(rows) % 10000 == 0 and rows:
            print(f"      … fetched {len(rows):,} new bars so far")

    new = pd.DataFrame(rows, columns=["ds", "y"])
    if existing is not None and not new.empty:
        df = pd.concat([existing, new], ignore_index=True).drop_duplicates("ds", keep="last")
    elif existing is not None:
        df = existing
    else:
        df = new
    df = df.sort_values("ds").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("Bitstamp returned no data and there is no cache to fall back to.")

    df.to_csv(cache, index=False)
    print(f"      → {len(df):,} hourly rows from {df['ds'].min()} to {df['ds'].max()}  ({len(new):,} new this run)")
    return df


def make_model() -> Prophet:
    # Hourly BTCUSDT over ~9 years — log-space keeps proportional moves
    # consistent. All three seasonalities are meaningful at this resolution
    # (24h intraday cycle, weekly weekend effect, yearly Bitcoin macro cycle).
    return Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
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
    if len(df) <= HOLDOUT_HOURS:
        raise RuntimeError(
            f"Only {len(df)} rows but need > {HOLDOUT_HOURS} for the holdout split."
        )

    train = df.iloc[:-HOLDOUT_HOURS].copy()
    test = df.iloc[-HOLDOUT_HOURS:].copy()

    print(f"[2/5] Backtest split: train {len(train):,} hours  ·  test {len(test):,} hours (>{train['ds'].iloc[-1]})")
    print(f"[3/5] Training Prophet (log-space) on training set… this can take a few minutes for hourly data.")
    model = fit_log(train)

    future = model.make_future_dataframe(periods=HOLDOUT_HOURS + 24, freq="h")
    forecast = predict_exp(model, future)

    merged = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(test, on="ds", how="inner")
    err = merged["y"] - merged["yhat"]
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((err.abs() / merged["y"]).mean() * 100)

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


def daily_means(df: pd.DataFrame) -> pd.DataFrame:
    """Daily mean of the hourly series — used for the all-time long-view
    chart so the dashboard ships ~3.3k points instead of ~78k."""
    daily = (
        df.set_index("ds")["y"]
          .resample("1D")
          .mean()
          .dropna()
          .reset_index()
    )
    return daily


def prior_year_overlay(df: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    """Same calendar window 1 year before the blind-test window, shifted
    forward by 365d so the line aligns with the test x-axis. Empty if
    history doesn't reach far enough back."""
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
    history_daily = daily_means(df)
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ticker": TICKER,
            "source": "Bitstamp",
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
        # Daily means for the all-time chart (~3.3k points, compact). The
        # backtest and forecast keys keep full hourly resolution.
        "history": to_record(history_daily, ["ds", "y"]),
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
