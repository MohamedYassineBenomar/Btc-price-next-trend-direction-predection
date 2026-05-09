/* BTC · Prophet dashboard — chart + KPI rendering */

const COLORS = {
  gold:    '#E8B86B',
  goldDim: 'rgba(232, 184, 107, 0.18)',
  blue:    '#6F86FF',
  blueDim: 'rgba(111, 134, 255, 0.18)',
  fgMuted: 'rgba(244, 244, 246, 0.5)',
  border:  'rgba(255, 255, 255, 0.05)',
};

const fmtUSD0 = (v) => (v == null || isNaN(v) ? '—' : '$' + Math.round(v).toLocaleString('en-US'));
const fmtUSDk = (v) => {
  if (v >= 1_000_000) return '$' + (v / 1_000_000).toFixed(1) + 'M';
  if (v >= 1_000)     return '$' + (v / 1_000).toFixed(v >= 10_000 ? 0 : 1) + 'k';
  if (v >= 1)         return '$' + v.toFixed(0);
  return '$' + v.toFixed(2);
};
const fmtPct = (v) => v.toFixed(2) + '%';
const fmtDate = (s) => new Date(s).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
const fmtMonth = (s) => new Date(s).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
const ms = (s) => new Date(s).getTime();
const $ = (sel) => document.querySelector(sel);

// ─── Cursor glow ──────────────────────────────────────────────
(function initCursorGlow() {
  if (matchMedia('(hover: none)').matches) return;
  const glow = document.querySelector('.cursor-glow');
  if (!glow) return;
  let x = innerWidth / 2, y = innerHeight / 2;
  let tx = x, ty = y;
  let raf = null;
  document.addEventListener('mousemove', (e) => {
    tx = e.clientX; ty = e.clientY;
    glow.style.opacity = '1';
    if (!raf) raf = requestAnimationFrame(animate);
  });
  document.addEventListener('mouseleave', () => { glow.style.opacity = '0'; });
  function animate() {
    x += (tx - x) * 0.12;
    y += (ty - y) * 0.12;
    glow.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
    if (Math.abs(tx - x) > 0.5 || Math.abs(ty - y) > 0.5) raf = requestAnimationFrame(animate);
    else raf = null;
  }
})();

// ─── KPI hover spotlight ──────────────────────────────────────
document.querySelectorAll('.kpi').forEach((card) => {
  card.addEventListener('mousemove', (e) => {
    const r = card.getBoundingClientRect();
    card.style.setProperty('--mx', ((e.clientX - r.left) / r.width) * 100 + '%');
    card.style.setProperty('--my', ((e.clientY - r.top) / r.height) * 100 + '%');
  });
});

// ─── Load and render ──────────────────────────────────────────
fetch('data.json', { cache: 'no-cache' })
  .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
  .then(render)
  .catch((err) => {
    console.error(err);
    document.querySelector('.container').insertAdjacentHTML(
      'beforeend',
      `<div style="padding:48px;text-align:center;color:var(--red);font-size:14px">
         Failed to load <code>data.json</code>. Run <code>python pipeline.py</code> first.
       </div>`
    );
  });

function render(data) {
  paintKPIs(data);
  paintFooter(data);
  renderHistoryChart(data);
  const mainChart = renderMainChart(data);
  renderBacktestChart(data);
  wireScaleToggle(mainChart);
}

// ─── All-time history chart ───────────────────────────────────
function renderHistoryChart(data) {
  const points = data.history.map((d) => ({ x: ms(d.ds), y: d.y }));
  const opts = {
    ...commonOpts(),
    chart: { ...commonOpts().chart, id: 'history', type: 'area', height: 380 },
    series: [{ name: 'BTC close (15-day avg)', data: points }],
    colors: [COLORS.gold],
    stroke: { curve: 'smooth', width: 1.6 },
    fill: {
      type: 'gradient',
      gradient: {
        shadeIntensity: 0.35,
        type: 'vertical',
        opacityFrom: 0.42,
        opacityTo: 0.0,
        stops: [0, 100],
      },
    },
    markers: { size: 0, hover: { size: 0 } },
    yaxis: {
      ...commonOpts().yaxis,
      logarithmic: true,
      logBase: 10,
      min: 1,
    },
  };
  new ApexCharts(document.getElementById('chart-history'), opts).render();
}

// ─── KPIs ─────────────────────────────────────────────────────
function paintKPIs(data) {
  const m = data.meta;
  const last = m.current_price;

  $('#kpi-price').textContent = fmtUSD0(last);
  const dayDelta = ((last - m.previous_close) / m.previous_close) * 100;
  $('#kpi-price-meta').innerHTML =
    `<span class="${dayDelta >= 0 ? 'up' : 'down'}">` +
    `<span class="arrow">${dayDelta >= 0 ? '▲' : '▼'}</span>${Math.abs(dayDelta).toFixed(2)}%` +
    `</span>` +
    ` &nbsp;·&nbsp; ${fmtDate(m.data_end)}`;

  const fEnd = data.forecast[data.forecast.length - 1];
  $('#kpi-forecast').textContent = fmtUSD0(fEnd.yhat);
  const fdelta = ((fEnd.yhat - last) / last) * 100;
  $('#kpi-forecast-meta').innerHTML =
    `<span class="${fdelta >= 0 ? 'up' : 'down'}">` +
    `<span class="arrow">${fdelta >= 0 ? '▲' : '▼'}</span>${Math.abs(fdelta).toFixed(2)}%` +
    `</span>` +
    ` &nbsp;·&nbsp; by ${fmtMonth(fEnd.ds)}`;

  $('#kpi-mape').textContent = fmtPct(data.backtest.metrics.mape);
  $('#kpi-dir').textContent = fmtPct(data.backtest.metrics.directional_accuracy);

  $('#m-mae').textContent = fmtUSD0(data.backtest.metrics.mae);
  $('#m-rmse').textContent = fmtUSD0(data.backtest.metrics.rmse);
  $('#m-mape').textContent = fmtPct(data.backtest.metrics.mape);
  $('#m-dir').textContent = fmtPct(data.backtest.metrics.directional_accuracy);
}

function paintFooter(data) {
  const m = data.meta;
  $('#data-range').textContent = `${fmtDate(m.data_start)} → ${fmtDate(m.data_end)}`;
  $('#n-obs').textContent = m.n_observations.toLocaleString();
  $('#generated-at').textContent = fmtDate(m.generated_at);
  $('#updated-label').textContent = 'updated ' + fmtDate(m.generated_at);
}

// ─── Common chart options ─────────────────────────────────────
function commonOpts() {
  return {
    chart: {
      background: 'transparent',
      fontFamily: 'Inter, sans-serif',
      foreColor: COLORS.fgMuted,
      toolbar: { show: false },
      zoom: { enabled: true, type: 'x', autoScaleYaxis: true },
      animations: { enabled: true, easing: 'easeout', speed: 700, animateGradually: { enabled: false } },
    },
    grid: {
      borderColor: COLORS.border,
      strokeDashArray: 0,
      padding: { top: 8, right: 22, bottom: 4, left: 12 },
      xaxis: { lines: { show: false } },
      yaxis: { lines: { show: true } },
    },
    dataLabels: { enabled: false },
    legend: { show: false },
    xaxis: {
      type: 'datetime',
      axisBorder: { show: false },
      axisTicks: { color: COLORS.border },
      labels: { style: { colors: COLORS.fgMuted, fontSize: '11px', fontWeight: 400 } },
      crosshairs: { stroke: { color: 'rgba(255,255,255,0.18)', width: 1, dashArray: 3 } },
      tooltip: { enabled: false },
    },
    yaxis: {
      labels: {
        style: { colors: COLORS.fgMuted, fontSize: '11px', fontWeight: 400 },
        formatter: (v) => fmtUSDk(v),
      },
    },
    tooltip: {
      theme: 'dark',
      shared: true,
      intersect: false,
      x: { format: 'MMM dd, yyyy' },
      y: { formatter: (v) => fmtUSD0(v) },
    },
  };
}

// ─── Main chart: history + forward forecast ───────────────────
function renderMainChart(data) {
  const allHistMs = data.history.map((d) => ms(d.ds));
  const lastHist = data.history[data.history.length - 1];
  const lastMs = ms(lastHist.ds);

  // Default visible window: last ~3 years so the 180-day forecast at the
  // right edge is clearly readable. User can zoom out via the toggle.
  const windowStartMs = lastMs - 3 * 365 * 24 * 60 * 60 * 1000;
  const histPoints = data.history.map((d) => ({ x: ms(d.ds), y: d.y }));
  const forwardLine = [
    { x: lastMs, y: lastHist.y },
    ...data.forecast.map((d) => ({ x: ms(d.ds), y: d.yhat })),
  ];

  const todayMs = lastMs;
  const opts = {
    ...commonOpts(),
    chart: { ...commonOpts().chart, id: 'main', type: 'area', height: 520 },
    series: [
      { name: 'BTC close', data: histPoints  },
      { name: 'Forecast',  data: forwardLine },
    ],
    colors: [COLORS.gold, COLORS.blue],
    stroke: { curve: 'smooth', width: [1.6, 2.4], dashArray: [0, 5] },
    fill: {
      type: ['gradient', 'solid'],
      opacity: [1, 0],
      gradient: {
        shadeIntensity: 0.35,
        type: 'vertical',
        opacityFrom: 0.42,
        opacityTo: 0.0,
        stops: [0, 100],
      },
    },
    markers: { size: 0, hover: { size: 0 } },
    annotations: {
      xaxis: [
        {
          x: todayMs,
          strokeDashArray: 4,
          borderColor: 'rgba(255,255,255,0.28)',
          borderWidth: 1,
          label: {
            text: 'today',
            position: 'top',
            orientation: 'horizontal',
            offsetY: -4,
            style: {
              color: '#F4F4F6',
              background: 'rgba(11,13,18,0.85)',
              fontSize: '10px',
              fontWeight: '500',
              padding: { left: 8, right: 8, top: 3, bottom: 3 },
            },
            borderColor: 'transparent',
          },
        },
      ],
    },
    yaxis: {
      ...commonOpts().yaxis,
      logarithmic: true,
      logBase: 10,
      min: 1,
    },
    xaxis: {
      ...commonOpts().xaxis,
      min: windowStartMs,
      max: ms(data.forecast[data.forecast.length - 1].ds),
    },
  };

  const chart = new ApexCharts(document.getElementById('chart-main'), opts);
  chart.render();
  // Stash data for the range toggle.
  chart.__btc = { allHistMs, histPoints, forwardLine, lastMs };
  return chart;
}

// ─── Backtest chart: actual vs predicted vs prior-year ────────
function renderBacktestChart(data) {
  const preds = data.backtest.predictions;
  const predicted = preds.map((d) => ({ x: ms(d.ds), y: d.yhat }));
  const actual    = preds.map((d) => ({ x: ms(d.ds), y: d.y }));
  const priorYear = (data.backtest.prior_year || []).map((d) => ({ x: ms(d.ds), y: d.y_prior }));

  // ApexCharts' auto-scale on multi-series line charts can clip series whose
  // range falls outside the dominant one. Force the y-axis to include all
  // three series with a small pad.
  const allYs = [
    ...predicted.map((p) => p.y),
    ...actual.map((p) => p.y),
    ...priorYear.map((p) => p.y),
  ].filter((v) => v != null && !isNaN(v));
  const yMin = Math.min(...allYs);
  const yMax = Math.max(...allYs);
  const pad = (yMax - yMin) * 0.06;

  const opts = {
    ...commonOpts(),
    chart: { ...commonOpts().chart, id: 'backtest', type: 'line', height: 440 },
    series: [
      { name: 'Same period · prior year', data: priorYear },
      { name: 'Predicted', data: predicted },
      { name: 'Actual',    data: actual    },
    ],
    colors: ['#9CA3AF', COLORS.blue, COLORS.gold],
    stroke: { curve: 'smooth', width: [2, 2, 2.8], dashArray: [4, 5, 0] },
    fill: { type: ['solid', 'solid', 'solid'], opacity: [1, 1, 1] },
    markers: { size: 0, hover: { size: 5 } },
    yaxis: {
      ...commonOpts().yaxis,
      min: Math.max(0, yMin - pad),
      max: yMax + pad,
      forceNiceScale: true,
    },
  };

  new ApexCharts(document.getElementById('chart-backtest'), opts).render();
}

// ─── Toggle visible time range on main chart ──────────────────
function wireScaleToggle(chart) {
  const toggle = $('#range-toggle');
  if (!toggle) return;
  toggle.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    const range = btn.dataset.range;
    toggle.querySelectorAll('button').forEach((b) => {
      const active = b === btn;
      b.classList.toggle('active', active);
      b.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    const ctx = chart.__btc;
    if (!ctx) return;
    const lastForecastMs = ctx.forwardLine[ctx.forwardLine.length - 1].x;
    const yearMs = 365 * 24 * 60 * 60 * 1000;
    const min = range === 'all' ? ctx.allHistMs[0]
              : range === '1y'  ? ctx.lastMs - yearMs
              : ctx.lastMs - 3 * yearMs;
    chart.updateOptions(
      { xaxis: { min, max: lastForecastMs } },
      false,
      true
    );
  });
}
