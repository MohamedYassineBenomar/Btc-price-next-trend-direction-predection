"""Streamlit-hosted BTC · Prophet dashboard.

Run locally:
    .venv/bin/streamlit run streamlit_app.py

Deploy:
    Push to GitHub → connect at share.streamlit.io → entry-point streamlit_app.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import HOLDOUT_DAYS, blind_backtest, fetch_history, forward_forecast

# ─── Theme ────────────────────────────────────────────────────
BG          = "#06070A"
SURFACE     = "rgba(255,255,255,0.025)"
BORDER      = "rgba(255,255,255,0.08)"
FG          = "#F4F4F6"
FG_MUTED    = "rgba(244,244,246,0.62)"
FG_DIM      = "rgba(244,244,246,0.42)"
GOLD        = "#E8B86B"
GOLD_BRIGHT = "#F2C27A"
BLUE        = "#6F86FF"
GREEN       = "#4ADE80"
RED         = "#F87171"

# ─── Page setup ───────────────────────────────────────────────
st.set_page_config(
    page_title="BTC · Prophet forecast",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS — Inter + Instrument Serif, dark surfaces, hairline borders.
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {{
    background: {BG} !important;
    color: {FG};
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
.stApp {{
    background:
        radial-gradient(ellipse 1100px 540px at 50% -160px, rgba(232,184,107,0.10), transparent 65%),
        radial-gradient(ellipse 700px 440px at 92% 18%, rgba(232,184,107,0.04), transparent 70%),
        radial-gradient(ellipse 800px 500px at 8% 62%, rgba(111,134,255,0.04), transparent 70%),
        {BG} !important;
}}
[data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ max-width: 1240px; padding-top: 2.5rem; padding-bottom: 4rem; }}

h1, h2, h3, h4, h5, h6 {{ color: {FG}; font-family: 'Inter', sans-serif; letter-spacing: -0.022em; font-weight: 600; }}
p, li, span, label {{ color: {FG_MUTED}; }}

/* Hero */
.hero-eyebrow {{
    display: inline-block;
    font-size: 11px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {GOLD};
    margin-bottom: 18px;
    font-weight: 500;
}}
.hero-title {{
    font-size: clamp(2.2rem, 4.6vw + 0.6rem, 4.4rem);
    line-height: 1.02;
    letter-spacing: -0.028em;
    font-weight: 600;
    color: {FG};
    margin: 0;
    text-wrap: balance;
}}
.hero-title em {{
    font-family: 'Instrument Serif', serif;
    font-style: italic;
    font-weight: 400;
    background: linear-gradient(180deg, {GOLD_BRIGHT}, {GOLD} 60%, #C99850);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    padding-right: 0.04em;
}}
.hero-sub {{
    margin-top: 22px;
    font-size: 17px;
    color: {FG_MUTED};
    max-width: 620px;
    line-height: 1.55;
}}
.mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.94em; }}

/* Nav badge */
.nav-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 18px;
    margin-bottom: 32px;
    border-bottom: 1px solid {BORDER};
}}
.logo {{ display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 14.5px; color: {FG}; }}
.logo-mark {{
    width: 30px; height: 30px; border-radius: 9px;
    background: linear-gradient(135deg, {GOLD_BRIGHT}, {GOLD} 50%, #B98C44);
    display: grid; place-items: center; color: #1A1208;
    font-weight: 700; font-size: 15px;
    box-shadow: 0 0 30px -8px rgba(232,184,107,0.45),
                inset 0 1px 0 rgba(255,255,255,0.25),
                inset 0 -1px 0 rgba(0,0,0,0.12);
}}
.logo-dot {{ margin: 0 6px; color: rgba(244,244,246,0.22); font-weight: 400; }}
.badge {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 12px; border-radius: 999px;
    font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase;
    color: {FG_MUTED}; background: {SURFACE}; border: 1px solid {BORDER};
}}
.pulse {{
    width: 6px; height: 6px; border-radius: 50%;
    background: {GREEN}; box-shadow: 0 0 10px rgba(74,222,128,0.7);
    animation: pulse 2.4s ease-in-out infinite;
}}
@keyframes pulse {{ 0%,100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.45; transform: scale(0.85); }} }}

/* KPI cards */
.kpi {{
    padding: 20px 22px;
    border: 1px solid {BORDER};
    background: {SURFACE};
    border-radius: 16px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset, 0 30px 60px -40px rgba(0,0,0,0.6);
    height: 100%;
    transition: transform 0.5s cubic-bezier(0.22,1,0.36,1), border-color 0.4s;
}}
.kpi:hover {{ transform: translateY(-2px); border-color: rgba(255,255,255,0.13); }}
.kpi-label {{
    font-size: 10.5px; letter-spacing: 0.18em; text-transform: uppercase;
    color: {FG_DIM}; margin-bottom: 14px;
}}
.kpi-value {{
    font-size: 30px; font-weight: 600; letter-spacing: -0.025em;
    font-variant-numeric: tabular-nums; color: {FG}; line-height: 1.1;
}}
.kpi-meta {{
    margin-top: 8px; font-size: 12.5px;
    color: {FG_MUTED}; font-variant-numeric: tabular-nums;
}}
.kpi-meta.up {{ color: {GREEN}; }}
.kpi-meta.down {{ color: {RED}; }}

/* Section heads */
.eyebrow-sm {{
    display: inline-block; font-size: 10.5px;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: {GOLD}; margin-bottom: 12px; font-weight: 500;
}}
h2.section-h {{ font-size: 26px; font-weight: 600; letter-spacing: -0.022em; color: {FG}; margin: 0; }}
.section-sub {{ font-size: 14px; color: {FG_MUTED}; margin-top: 8px; max-width: 700px; line-height: 1.55; }}

/* Chart card wrapper */
.chart-wrap {{
    border: 1px solid {BORDER};
    background: {SURFACE};
    border-radius: 22px;
    padding: 18px 6px 6px;
    margin-top: 16px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset, 0 30px 60px -40px rgba(0,0,0,0.6);
}}

/* Method steps */
.step {{
    display: flex; gap: 22px; padding: 18px 4px;
    border-bottom: 1px solid {BORDER};
}}
.step:last-child {{ border-bottom: 0; }}
.step-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: {FG_DIM}; min-width: 22px; margin-top: 3px;
    letter-spacing: 0.05em;
}}
.step strong {{ font-size: 14.5px; font-weight: 600; color: {FG}; }}
.step span.body {{ font-size: 13.5px; color: {FG_MUTED}; line-height: 1.5; display: block; margin-top: 4px; }}

/* Footer */
.footer {{
    margin-top: 64px; padding: 28px 0;
    border-top: 1px solid {BORDER};
    font-size: 12px; color: {FG_DIM};
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px;
}}

/* Streamlit button restyle */
.stButton > button {{
    background: {SURFACE};
    color: {FG_MUTED};
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 7px 16px;
    font-size: 12.5px;
    font-weight: 500;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    background: rgba(255,255,255,0.045);
    border-color: rgba(255,255,255,0.13);
    color: {FG};
}}

@media (max-width: 720px) {{
    .nav-row {{ flex-wrap: wrap; gap: 12px; }}
    .hero-title {{ font-size: 2.4rem; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Cached pipeline ──────────────────────────────────────────
@st.cache_data(ttl=60 * 30, show_spinner="Fetching BTC history…")
def cached_history() -> pd.DataFrame:
    return fetch_history()


@st.cache_data(ttl=60 * 30, show_spinner="Running blind backtest on the last 365 days…")
def cached_backtest(df: pd.DataFrame):
    backtest, metrics = blind_backtest(df)
    return backtest, metrics


@st.cache_data(ttl=60 * 30, show_spinner="Projecting 180 days forward…")
def cached_forecast(df: pd.DataFrame) -> pd.DataFrame:
    return forward_forecast(df)


# ─── Helpers ──────────────────────────────────────────────────
def fmt_usd(v: float, decimals: int = 0) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"${v:,.{decimals}f}"


def fmt_pct(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}%"


def fmt_date(d) -> str:
    if isinstance(d, str):
        d = pd.to_datetime(d)
    return d.strftime("%b %-d, %Y")


# ─── Plotly base layout ──────────────────────────────────────
def base_layout(height: int = 460) -> dict:
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=FG_MUTED, size=12),
        margin=dict(l=20, r=20, t=20, b=40),
        hoverlabel=dict(
            bgcolor="rgba(11,13,18,0.94)",
            bordercolor="rgba(255,255,255,0.13)",
            font=dict(family="Inter, sans-serif", color=FG, size=12),
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color=FG_MUTED,
            linecolor="rgba(255,255,255,0.07)",
            tickfont=dict(size=11),
            type="date",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
            color=FG_MUTED,
            tickfont=dict(size=11),
            tickprefix="$",
            tickformat=",.0f",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=FG_MUTED, size=11),
            itemclick=False,
            itemdoubleclick=False,
        ),
        showlegend=True,
    )


# ─── Charts ───────────────────────────────────────────────────
def main_chart(df: pd.DataFrame, forward: pd.DataFrame) -> go.Figure:
    last = df.iloc[-1]
    cutoff = pd.Timestamp(last["ds"]) - pd.Timedelta(days=3 * 365)
    visible = df[df["ds"] >= cutoff]

    fig = go.Figure()

    # Forecast band — upper then lower with fill='tonexty'
    fig.add_trace(go.Scatter(
        x=forward["ds"], y=forward["yhat_upper"],
        mode="lines", line=dict(width=0, color=BLUE),
        showlegend=False, hoverinfo="skip", name="upper",
    ))
    fig.add_trace(go.Scatter(
        x=forward["ds"], y=forward["yhat_lower"],
        mode="lines", line=dict(width=0, color=BLUE),
        fill="tonexty", fillcolor="rgba(111,134,255,0.16)",
        showlegend=True, name="80% range", hoverinfo="skip",
    ))

    # History area (gold)
    fig.add_trace(go.Scatter(
        x=visible["ds"], y=visible["y"],
        mode="lines",
        line=dict(color=GOLD, width=1.8, shape="spline", smoothing=0.4),
        fill="tozeroy",
        fillcolor="rgba(232,184,107,0.18)",
        name="BTC close",
        hovertemplate="<b>%{x|%b %-d, %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    # Forward forecast line (dashed blue) — bridged to last hist point
    bridge_x = pd.concat([pd.Series([last["ds"]]), forward["ds"]], ignore_index=True)
    bridge_y = pd.concat([pd.Series([last["y"]]), forward["yhat"]], ignore_index=True)
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="lines",
        line=dict(color=BLUE, width=2.2, dash="dash", shape="spline", smoothing=0.4),
        name="Forecast",
        hovertemplate="<b>%{x|%b %-d, %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(**base_layout(height=480))
    fig.update_yaxes(type="log", tickprefix="$", tickformat=",.0f")
    fig.update_xaxes(range=[cutoff, forward["ds"].max()])

    # "Today" vertical line — Plotly's add_vline computes a mean over X which
    # breaks on Timestamp objects in newer pandas, so pass the date as an ISO
    # string and Plotly handles axis-mapping itself.
    today_iso = pd.Timestamp(last["ds"]).strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=today_iso, x1=today_iso, xref="x",
        y0=0, y1=1, yref="paper",
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=today_iso, xref="x",
        y=1, yref="paper",
        yanchor="bottom",
        text="today",
        showarrow=False,
        font=dict(color=FG, size=10, family="Inter, sans-serif"),
        bgcolor="rgba(11,13,18,0.85)",
        borderpad=3,
    )
    return fig


def backtest_chart(backtest: pd.DataFrame) -> go.Figure:
    # Clip the band so a 365-day Prophet upper bound doesn't squash the y-axis.
    ys = pd.concat([backtest["y"], backtest["yhat"]])
    span = ys.max() - ys.min()
    clip_max = ys.max() + span * 0.55
    clip_min = max(0, ys.min() - span * 0.55)
    upper = backtest["yhat_upper"].clip(upper=clip_max)
    lower = backtest["yhat_lower"].clip(lower=clip_min)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=backtest["ds"], y=upper,
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=backtest["ds"], y=lower,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(111,134,255,0.15)",
        name="80% range", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=backtest["ds"], y=backtest["yhat"],
        mode="lines",
        line=dict(color=BLUE, width=1.8, dash="dash", shape="spline", smoothing=0.4),
        name="Predicted",
        hovertemplate="<b>%{x|%b %-d, %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=backtest["ds"], y=backtest["y"],
        mode="lines",
        line=dict(color=GOLD, width=2.6, shape="spline", smoothing=0.4),
        name="Actual",
        hovertemplate="<b>%{x|%b %-d, %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(**base_layout(height=440))
    return fig


# ─── Layout ───────────────────────────────────────────────────
def kpi(label: str, value: str, meta_html: str = "") -> str:
    return f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-meta">{meta_html or '&nbsp;'}</div>
    </div>
    """


# ─── Render ───────────────────────────────────────────────────
df = cached_history()
backtest, metrics = cached_backtest(df)
forward = cached_forecast(df)

last_close = float(df["y"].iloc[-1])
prev_close = float(df["y"].iloc[-2])
day_delta = ((last_close - prev_close) / prev_close) * 100

f90 = forward.iloc[min(89, len(forward) - 1)]
f90_val = float(f90["yhat"])
f90_delta = ((f90_val - last_close) / last_close) * 100

data_end = pd.Timestamp(df["ds"].iloc[-1])
data_start = pd.Timestamp(df["ds"].iloc[0])
generated = datetime.now(timezone.utc)

# ─── Nav row ──────────────────────────────────────────────────
st.markdown(
    f"""
<div class="nav-row">
  <div class="logo">
    <div class="logo-mark">₿</div>
    <span>Prophet<span class="logo-dot">·</span>BTC</span>
  </div>
  <div class="badge"><span class="pulse"></span><span>updated {fmt_date(data_end)}</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ─── Hero ─────────────────────────────────────────────────────
st.markdown(
    f"""
<div>
  <span class="hero-eyebrow">Time-series forecast · Facebook Prophet</span>
  <h1 class="hero-title">Predicting Bitcoin's next <em>direction</em>.</h1>
  <p class="hero-sub">A blind out-of-sample backtest on the last {HOLDOUT_DAYS} days of <span class="mono">BTC-USD</span>, with a forward 180-day projection generated from the full price history.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")  # spacing

# ─── KPIs ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4, gap="small")

day_arrow = "▲" if day_delta >= 0 else "▼"
day_class = "up" if day_delta >= 0 else "down"
c1.markdown(
    kpi(
        "Spot price",
        fmt_usd(last_close),
        f'<span class="{day_class}">{day_arrow} {abs(day_delta):.2f}%</span> &nbsp;·&nbsp; {fmt_date(data_end)}',
    ),
    unsafe_allow_html=True,
)

f_arrow = "▲" if f90_delta >= 0 else "▼"
f_class = "up" if f90_delta >= 0 else "down"
c2.markdown(
    kpi(
        "90-day forecast",
        fmt_usd(f90_val),
        f'<span class="{f_class}">{f_arrow} {abs(f90_delta):.2f}%</span> &nbsp;·&nbsp; by {pd.Timestamp(f90["ds"]).strftime("%b %Y")}',
    ),
    unsafe_allow_html=True,
)

c3.markdown(
    kpi("Backtest MAPE", fmt_pct(metrics["mape"]), f"on {metrics['n_test_points']} held-out days"),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi("Direction accuracy", fmt_pct(metrics["directional_accuracy"]), "daily up / down hit-rate"),
    unsafe_allow_html=True,
)

# ─── Section 1: Long view ─────────────────────────────────────
st.write("")
st.write("")
st.markdown(
    """
<div>
  <span class="eyebrow-sm">01 — Long view</span>
  <h2 class="section-h">All-time price &amp; forward projection</h2>
  <p class="section-sub">Last 3 years of BTC-USD plotted on a logarithmic scale, extended by Prophet's 180-day forecast and 80% uncertainty band.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(main_chart(df, forward), use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

# ─── Section 2: Blind backtest ────────────────────────────────
st.write("")
st.write("")
st.markdown(
    f"""
<div>
  <span class="eyebrow-sm">02 — Blind backtest</span>
  <h2 class="section-h">Last {HOLDOUT_DAYS} days · prediction vs reality</h2>
  <p class="section-sub">Prophet was trained <em>only</em> on data before this window, then asked to predict it cold. No look-ahead, no re-training, no leakage.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.plotly_chart(backtest_chart(backtest), use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4, gap="small")
m1.markdown(kpi("MAE", fmt_usd(metrics["mae"]), "mean absolute error"), unsafe_allow_html=True)
m2.markdown(kpi("RMSE", fmt_usd(metrics["rmse"]), "root-mean-squared error"), unsafe_allow_html=True)
m3.markdown(kpi("MAPE", fmt_pct(metrics["mape"]), "mean absolute % error"), unsafe_allow_html=True)
m4.markdown(kpi("Direction", fmt_pct(metrics["directional_accuracy"]), "daily up/down hit-rate"), unsafe_allow_html=True)

# ─── Methodology ──────────────────────────────────────────────
st.write("")
st.write("")
st.write("")
left, right = st.columns([1, 1.3], gap="large")
with left:
    st.markdown(
        """
<div>
  <span class="eyebrow-sm">Methodology</span>
  <h2 class="section-h" style="font-size:30px;font-weight:500;">Honest, reproducible, no leakage.</h2>
</div>
""",
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        f"""
<div>
  <div class="step"><span class="step-num">01</span><div><strong>Fetch</strong><span class="body">All-time daily BTC-USD closes from Yahoo Finance ({len(df):,} observations since {fmt_date(data_start)}).</span></div></div>
  <div class="step"><span class="step-num">02</span><div><strong>Hold out</strong><span class="body">Last {HOLDOUT_DAYS} days are removed from the training set and never seen by the model.</span></div></div>
  <div class="step"><span class="step-num">03</span><div><strong>Fit</strong><span class="body">Prophet trained on log-prices with yearly seasonality and a flexible changepoint prior.</span></div></div>
  <div class="step"><span class="step-num">04</span><div><strong>Score</strong><span class="body">Predictions over the held-out window scored on MAE, RMSE, MAPE, and directional hit-rate.</span></div></div>
  <div class="step"><span class="step-num">05</span><div><strong>Project</strong><span class="body">Re-fit on the full series, project 180 days forward with an 80% uncertainty band.</span></div></div>
</div>
""",
        unsafe_allow_html=True,
    )

# ─── Refresh + footer ─────────────────────────────────────────
st.write("")
st.write("")
left, right = st.columns([3, 1])
with left:
    st.markdown(
        f"""
<div class="footer">
  <div>{fmt_date(data_start)} → {fmt_date(data_end)} · {len(df):,} daily observations</div>
  <div>Source: Yahoo Finance · Model: Prophet · Generated {fmt_date(generated)}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with right:
    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
