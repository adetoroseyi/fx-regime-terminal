/**
 * FX Regime Terminal — Dashboard App
 *
 * Live-refreshing dashboard with OANDA prices, regime signals,
 * backtest results, model quality metrics, and forward testing.
 */

const PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'];
const REFRESH_INTERVAL = 60; // seconds
const DATA_BASE = 'data/';

// State
let signalsData = null;
let backtestResults = [];
let modelMeta = {};
let retrainHistory = [];
let forwardTests = [];
let livePrices = {};
let selectedPair = 'EURUSD';
let refreshTimer = null;
let countdown = REFRESH_INTERVAL;
let oandaAvailable = false;

// ── Initialisation ──────────────────────────────────────────

async function init() {
  showLoading(true);

  // Load all data in parallel
  await Promise.all([
    loadSignals(),
    loadBacktestResults(),
    loadModelMeta(),
    loadRetrainHistory(),
    loadForwardTests(),
    fetchLivePrices(),
  ]);

  render();
  showLoading(false);

  // Start auto-refresh
  startAutoRefresh();
}

function showLoading(show) {
  const el = document.getElementById('loading');
  if (el) el.style.display = show ? 'block' : 'none';
}

// ── Data Loading ────────────────────────────────────────────

async function loadSignals() {
  try {
    const resp = await fetch(DATA_BASE + 'signals.json');
    if (resp.ok) {
      signalsData = await resp.json();
    } else {
      signalsData = getSampleSignals();
    }
  } catch (e) {
    signalsData = getSampleSignals();
  }
}

async function loadBacktestResults() {
  try {
    const resp = await fetch(DATA_BASE + 'backtests.json');
    if (resp.ok) {
      backtestResults = await resp.json();
    }
  } catch (e) {
    backtestResults = [];
  }
}

async function loadModelMeta() {
  try {
    const resp = await fetch(DATA_BASE + 'models.json');
    if (resp.ok) {
      modelMeta = await resp.json();
    }
  } catch (e) {
    modelMeta = {};
  }
}

async function loadRetrainHistory() {
  try {
    const resp = await fetch(DATA_BASE + 'retrain_history.json');
    if (resp.ok) {
      retrainHistory = await resp.json();
    }
  } catch (e) {
    retrainHistory = [];
  }
}

async function loadForwardTests() {
  try {
    const resp = await fetch(DATA_BASE + 'forward_tests.json');
    if (resp.ok) {
      forwardTests = await resp.json();
    }
  } catch (e) {
    forwardTests = [];
  }
}

async function fetchLivePrices() {
  try {
    const resp = await fetch('/api/prices');
    if (resp.ok) {
      const data = await resp.json();
      if (data.prices && Object.keys(data.prices).length > 0) {
        livePrices = data.prices;
        oandaAvailable = true;
        updateConnectionStatus(true);
        return;
      }
    }
  } catch (e) {
    // OANDA not configured — fall back to signal data
  }
  oandaAvailable = false;
  updateConnectionStatus(false);
}

// ── Auto Refresh ────────────────────────────────────────────

function startAutoRefresh() {
  countdown = REFRESH_INTERVAL;
  updateCountdown();

  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(async () => {
    countdown--;
    updateCountdown();

    if (countdown <= 0) {
      countdown = REFRESH_INTERVAL;
      await refreshData();
    }
  }, 1000);
}

async function refreshData() {
  await Promise.all([
    loadSignals(),
    fetchLivePrices(),
  ]);
  render();
}

function updateCountdown() {
  const el = document.getElementById('refresh-countdown');
  if (el) el.textContent = `Refresh in ${countdown}s`;
}

function updateConnectionStatus(connected) {
  const el = document.getElementById('conn-status');
  if (!el) return;
  const dot = el.querySelector('.status-dot');
  const text = el.querySelector('.status-text');
  if (connected) {
    dot.className = 'status-dot online';
    text.textContent = 'OANDA Live';
  } else {
    dot.className = 'status-dot offline';
    text.textContent = 'Static Data';
  }
}

// ── Rendering ───────────────────────────────────────────────

function render() {
  renderPriceTicker();
  renderSignalCards();
  renderModelCards();
  renderPairTabs();
  renderBacktestTable();
  renderTradeLog();
  renderRetrainHistory();
  renderCharts();
  renderTimestamp();
}

function renderTimestamp() {
  const el = document.getElementById('last-updated');
  if (!el) return;

  if (signalsData && signalsData.generated_at) {
    const dt = new Date(signalsData.generated_at);
    el.textContent = `Signals: ${dt.toUTCString()}`;
  } else {
    el.textContent = 'Using sample data';
  }
}

// ── Price Ticker ────────────────────────────────────────────

function renderPriceTicker() {
  const container = document.getElementById('price-ticker');
  if (!container) return;

  if (!oandaAvailable || Object.keys(livePrices).length === 0) {
    container.style.display = 'none';
    return;
  }

  container.style.display = 'flex';
  container.innerHTML = PAIRS.map(pair => {
    const p = livePrices[pair];
    if (!p) return '';
    const decimals = pair.includes('JPY') ? 3 : 5;
    return `
      <div class="ticker-item">
        <span class="ticker-pair">${pair}</span>
        <span class="ticker-bid">${p.bid.toFixed(decimals)}</span>
        <span class="ticker-sep">/</span>
        <span class="ticker-ask">${p.ask.toFixed(decimals)}</span>
        <span class="ticker-spread">${p.spread} pip</span>
      </div>`;
  }).join('');
}

// ── Signal Cards ────────────────────────────────────────────

function renderSignalCards() {
  const container = document.getElementById('signal-cards');
  if (!container) return;

  const pairs = signalsData?.pairs || {};
  container.innerHTML = '';

  for (const pairCode of PAIRS) {
    const data = pairs[pairCode] || {
      pair: pairCode, regime: 'Unknown', confidence: 0,
      signal_direction: 'AVOID'
    };
    container.innerHTML += createSignalCard(data, pairCode);
  }
}

function createSignalCard(data, pairCode) {
  const direction = (data.signal_direction || 'AVOID').toLowerCase();
  const regimeClass = getRegimeClass(data.regime);

  // Live price section
  let priceHtml = '';
  const lp = livePrices[pairCode];
  if (lp) {
    const dec = pairCode.includes('JPY') ? 3 : 5;
    priceHtml = `
      <div class="live-price">
        <span class="price-value">${lp.mid.toFixed(dec)}</span>
        <span class="price-spread">${lp.spread} pip spread</span>
      </div>`;
  }

  // Signal details
  let detailsHtml = '';
  if (data.signal_direction === 'LONG' || data.signal_direction === 'SHORT') {
    const entry = data.entry_price ? data.entry_price.toFixed(5) : '—';
    const sl = data.stop_loss ? data.stop_loss.toFixed(5) : '—';
    const tp = data.take_profit ? data.take_profit.toFixed(5) : '—';

    detailsHtml = `
      <div class="signal-details">
        <div class="row"><span class="label">Strategy</span><span class="value">${data.active_strategy || '—'}</span></div>
        <div class="row"><span class="label">Entry</span><span class="value">${entry}</span></div>
        <div class="row"><span class="label">Stop Loss</span><span class="value sl">${sl}</span></div>
        <div class="row"><span class="label">Take Profit</span><span class="value tp">${tp}</span></div>
      </div>`;

    if (data.confirmations && Object.keys(data.confirmations).length > 0) {
      let confHtml = '<div class="confirmations">';
      for (const [label, met] of Object.entries(data.confirmations)) {
        const cls = met ? 'met' : 'not-met';
        const icon = met ? '✅' : '❌';
        confHtml += `<div class="confirmation-item ${cls}">${icon} ${label}</div>`;
      }
      confHtml += '</div>';
      detailsHtml += confHtml;
    }
  }

  const cooldownHtml = data.cooldown_active
    ? '<div class="cooldown-badge">Cooldown active</div>'
    : '';

  const confirmedHtml = data.regime_confirmed
    ? '<span class="confirmed-badge">Confirmed</span>'
    : '<span class="unconfirmed-badge">Unconfirmed</span>';

  return `
    <div class="card signal-card ${direction}">
      <div class="pair-name">${pairCode}</div>
      ${priceHtml}
      <div class="regime-row">
        <span class="regime-badge ${regimeClass}">${data.regime || 'Unknown'}</span>
        ${confirmedHtml}
      </div>
      <div class="confidence">Confidence: ${(data.confidence || 0).toFixed(1)}%</div>
      <div class="signal-direction ${direction}">${(data.signal_direction || 'AVOID').toUpperCase()}</div>
      ${detailsHtml}
      ${cooldownHtml}
    </div>`;
}

// ── Model Quality Cards ─────────────────────────────────────

function renderModelCards() {
  const container = document.getElementById('model-cards');
  if (!container) return;

  if (Object.keys(modelMeta).length === 0) {
    container.innerHTML = PAIRS.map(pair => `
      <div class="card model-card">
        <div class="model-pair">${pair}</div>
        <div class="model-empty">No model data yet</div>
      </div>`).join('');
    return;
  }

  container.innerHTML = PAIRS.map(pair => {
    const meta = modelMeta[pair];
    if (!meta) {
      return `<div class="card model-card">
        <div class="model-pair">${pair}</div>
        <div class="model-empty">No model</div>
      </div>`;
    }

    const regimeLabels = Object.values(meta.regime_map || {});
    const uniqueLabels = [...new Set(regimeLabels)];

    return `
      <div class="card model-card">
        <div class="model-pair">${pair}</div>
        <div class="model-stats">
          <div class="model-stat">
            <span class="stat-label">States</span>
            <span class="stat-value">${meta.n_states || '?'}</span>
          </div>
          <div class="model-stat">
            <span class="stat-label">BIC</span>
            <span class="stat-value">${meta.bic ? formatLargeNumber(meta.bic) : '—'}</span>
          </div>
          <div class="model-stat">
            <span class="stat-label">Cov Type</span>
            <span class="stat-value">${meta.covariance_type || '—'}</span>
          </div>
          <div class="model-stat">
            <span class="stat-label">Samples</span>
            <span class="stat-value">${meta.n_samples ? meta.n_samples.toLocaleString() : '—'}</span>
          </div>
        </div>
        <div class="model-regimes">
          ${uniqueLabels.map(l => `<span class="regime-tag ${getRegimeClass(l)}">${l}</span>`).join('')}
        </div>
      </div>`;
  }).join('');
}

// ── Pair Tabs ───────────────────────────────────────────────

function renderPairTabs() {
  const container = document.getElementById('pair-tabs');
  if (!container) return;

  container.innerHTML = PAIRS.map(p =>
    `<button class="pair-tab ${p === selectedPair ? 'active' : ''}"
             onclick="selectPair('${p}')">${p}</button>`
  ).join('');
}

function selectPair(pair) {
  selectedPair = pair;
  renderPairTabs();
  renderCharts();
}

// ── Backtest Table ──────────────────────────────────────────

function renderBacktestTable() {
  const tbody = document.getElementById('backtest-tbody');
  if (!tbody) return;

  if (backtestResults.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-state">No backtest results available. Will populate after next CI run.</td></tr>';
    return;
  }

  tbody.innerHTML = backtestResults.map(r => {
    const m = r.metrics || {};
    const retCls = (m.total_return_pct || 0) >= 0 ? 'positive' : 'negative';
    const alphaCls = (m.alpha_vs_buyhold || 0) >= 0 ? 'positive' : 'negative';
    return `<tr>
      <td>${m.pair || '—'}</td>
      <td>${(m.strategy || '—').toUpperCase()}</td>
      <td class="${retCls}">${(m.total_return_pct || 0).toFixed(2)}%</td>
      <td class="${alphaCls}">${(m.alpha_vs_buyhold || 0).toFixed(2)}%</td>
      <td>${(m.sharpe_ratio || 0).toFixed(2)}</td>
      <td class="negative">${(m.max_drawdown_pct || 0).toFixed(2)}%</td>
      <td>${(m.win_rate_pct || 0).toFixed(1)}%</td>
      <td>${m.total_trades || 0}</td>
      <td>${(m.avg_trade_duration || 0).toFixed(0)}h</td>
    </tr>`;
  }).join('');
}

// ── Trade Log ───────────────────────────────────────────────

function renderTradeLog() {
  const tbody = document.getElementById('trade-tbody');
  if (!tbody) return;

  const allTrades = [];
  for (const r of backtestResults) {
    if (r.trades) allTrades.push(...r.trades);
  }

  if (allTrades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-state">No trades recorded yet</td></tr>';
    return;
  }

  const recent = allTrades.slice(-50).reverse();
  tbody.innerHTML = recent.map(t => {
    const pnlCls = (t.pnl_pct || 0) >= 0 ? 'positive' : 'negative';
    return `<tr>
      <td>${t.pair || '—'}</td>
      <td>${t.strategy || '—'}</td>
      <td class="${t.direction === 'LONG' ? 'positive' : 'negative'}">${t.direction || '—'}</td>
      <td>${(t.entry_price || 0).toFixed(5)}</td>
      <td>${(t.exit_price || 0).toFixed(5)}</td>
      <td class="${pnlCls}">${(t.pnl_pct || 0).toFixed(3)}%</td>
      <td>${formatExitReason(t.exit_reason)}</td>
      <td>${t.duration_candles || 0}h</td>
      <td><span class="regime-tag-sm ${getRegimeClass(t.regime_at_entry)}">${t.regime_at_entry || '—'}</span></td>
    </tr>`;
  }).join('');
}

// ── Retrain History ─────────────────────────────────────────

function renderRetrainHistory() {
  const tbody = document.getElementById('retrain-tbody');
  if (!tbody) return;

  if (retrainHistory.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No retrain history available</td></tr>';
    return;
  }

  const recent = retrainHistory.slice(-20).reverse();
  tbody.innerHTML = recent.map(r => {
    const statusCls = r.status === 'OK' ? 'positive' : 'negative';
    return `<tr>
      <td>${r.date || '—'}</td>
      <td>${r.pair || '—'}</td>
      <td>${r.n_states || '—'}</td>
      <td>${r.score ? parseFloat(r.score).toFixed(0) : '—'}</td>
      <td>${r.bic ? formatLargeNumber(parseFloat(r.bic)) : '—'}</td>
      <td class="${r.converged === 'True' ? 'positive' : 'negative'}">${r.converged || '—'}</td>
      <td class="${statusCls}">${r.status || '—'}</td>
    </tr>`;
  }).join('');
}

// ── Charts ──────────────────────────────────────────────────

function renderCharts() {
  // Equity curve
  const pairResult = backtestResults.find(r =>
    r.metrics && r.metrics.pair === selectedPair
  );

  if (pairResult && pairResult.equity_curve) {
    renderEquityCurve('equity-chart', pairResult.equity_curve);
  } else {
    renderEmptyChart('equity-chart', 'No backtest data — run backtester');
  }

  // Regime distribution
  if (pairResult && pairResult.metrics && pairResult.metrics.regime_distribution) {
    renderRegimeDistribution('regime-dist-chart', pairResult.metrics.regime_distribution);
  } else {
    // Use model regime_map if available
    const meta = modelMeta[selectedPair];
    if (meta && meta.regime_map) {
      const dist = {};
      for (const label of Object.values(meta.regime_map)) {
        dist[label] = (dist[label] || 0) + 1;
      }
      renderRegimeDistribution('regime-dist-chart', dist);
    } else {
      renderEmptyChart('regime-dist-chart', 'No regime data available');
    }
  }

  // BIC chart — show sweep if available in retrain history
  renderBICChart();

  // Forward test confidence chart
  renderForwardChart();
}

function renderBICChart() {
  const pairHistory = retrainHistory.filter(r =>
    r.pair === selectedPair && r.sweep_bics
  );

  if (pairHistory.length === 0) {
    renderEmptyChart('bic-chart', 'No sweep data — will appear after retrain');
    return;
  }

  // Use the most recent sweep
  const latest = pairHistory[pairHistory.length - 1];
  let sweepData;
  try {
    // Parse Python dict format: {3: -123.0, 4: -456.0, ...}
    const cleaned = latest.sweep_bics
      .replace(/'/g, '"')
      .replace(/(\d+):/g, '"$1":');
    sweepData = JSON.parse(cleaned);
  } catch (e) {
    renderEmptyChart('bic-chart', 'Could not parse sweep data');
    return;
  }

  const nStates = Object.keys(sweepData).map(Number);
  const bics = Object.values(sweepData);
  const bestN = parseInt(latest.n_states);

  renderBICBarChart('bic-chart', nStates, bics, bestN);
}

function renderForwardChart() {
  if (forwardTests.length < 2) {
    renderEmptyChart('confidence-chart', 'Collecting data — will appear after 2+ days');
    return;
  }

  const pairData = forwardTests.map(ft => ({
    timestamp: ft.timestamp,
    confidence: ft.pairs?.[selectedPair]?.confidence || 0,
    regime: ft.pairs?.[selectedPair]?.regime || 'Unknown',
    signal: ft.pairs?.[selectedPair]?.signal_direction || 'HOLD',
  }));

  renderConfidenceTimeline('confidence-chart', pairData);
}

// ── Helpers ─────────────────────────────────────────────────

function getRegimeClass(regime) {
  if (!regime) return 'regime-noise';
  const lower = regime.toLowerCase();
  if (lower.includes('bull')) return 'regime-bull';
  if (lower.includes('bear')) return 'regime-bear';
  if (lower.includes('mean') || lower.includes('reversion')) return 'regime-mr';
  if (lower.includes('value') || lower.includes('gap') || lower.includes('fvg')) return 'regime-fvg';
  if (lower.includes('noise') || lower.includes('choppy')) return 'regime-noise';
  if (lower.includes('compression') || lower.includes('low vol')) return 'regime-compression';
  if (lower.includes('high vol') || lower.includes('expansion')) return 'regime-hv';
  return 'regime-noise';
}

function formatLargeNumber(num) {
  if (num === null || num === undefined) return '—';
  const abs = Math.abs(num);
  if (abs >= 1e9) return (num / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return (num / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  return num.toFixed(0);
}

function formatExitReason(reason) {
  const map = {
    'stop_loss': 'Stop Loss',
    'take_profit': 'Take Profit',
    'regime_exit': 'Regime Exit',
    'end_of_data': 'End of Data',
  };
  return map[reason] || reason || '—';
}

function getSampleSignals() {
  return {
    generated_at: null,
    pairs: {
      EURUSD: { pair: 'EURUSD', regime: 'Unknown', confidence: 0, signal_direction: 'HOLD', regime_confirmed: false },
      GBPUSD: { pair: 'GBPUSD', regime: 'Unknown', confidence: 0, signal_direction: 'HOLD', regime_confirmed: false },
      USDJPY: { pair: 'USDJPY', regime: 'Unknown', confidence: 0, signal_direction: 'HOLD', regime_confirmed: false },
      USDCHF: { pair: 'USDCHF', regime: 'Unknown', confidence: 0, signal_direction: 'HOLD', regime_confirmed: false },
    },
    signals_fired: 0,
  };
}

function sortTable(tableId, colIndex) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.rows);

  rows.sort((a, b) => {
    let aVal = a.cells[colIndex].textContent.replace('%', '').trim();
    let bVal = b.cells[colIndex].textContent.replace('%', '').trim();
    const aNum = parseFloat(aVal);
    const bNum = parseFloat(bVal);
    if (!isNaN(aNum) && !isNaN(bNum)) return bNum - aNum;
    return aVal.localeCompare(bVal);
  });

  rows.forEach(row => tbody.appendChild(row));
}

// ── Start ───────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
