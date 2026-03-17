/**
 * FX Regime Terminal — Dashboard App
 *
 * Loads signal and backtest data from static JSON files
 * and renders the dashboard UI.
 */

const PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'];

// Data paths (relative to dashboard/)
const SIGNALS_PATH = '../state/current_signals.json';
const BACKTEST_PATH_TEMPLATE = '../results/';

// State
let signalsData = null;
let backtestResults = [];
let selectedPair = 'EURUSD';

// ── Initialisation ──────────────────────────────────────────

async function init() {
  showLoading(true);
  await loadSignals();
  await loadBacktestResults();
  render();
  showLoading(false);
}

function showLoading(show) {
  const el = document.getElementById('loading');
  if (el) el.style.display = show ? 'block' : 'none';
}

// ── Data Loading ────────────────────────────────────────────

async function loadSignals() {
  try {
    const resp = await fetch(SIGNALS_PATH);
    if (resp.ok) {
      signalsData = await resp.json();
      console.log('Signals loaded:', signalsData);
    } else {
      console.warn('No signals file found — using sample data');
      signalsData = getSampleSignals();
    }
  } catch (e) {
    console.warn('Failed to load signals:', e);
    signalsData = getSampleSignals();
  }
}

async function loadBacktestResults() {
  // Try to load backtest results for each pair
  backtestResults = [];

  for (const pair of PAIRS) {
    for (const strategy of ['both', 'fvg', 'mr']) {
      try {
        // Try today's date first, then any available
        const today = new Date().toISOString().slice(0, 10).replace(/-/g, '');
        const url = `${BACKTEST_PATH_TEMPLATE}backtest_${pair}_${strategy}_${today}.json`;
        const resp = await fetch(url);
        if (resp.ok) {
          const data = await resp.json();
          backtestResults.push(data);
        }
      } catch (e) {
        // Silently skip missing files
      }
    }
  }
  console.log('Backtest results loaded:', backtestResults.length);
}

// ── Rendering ───────────────────────────────────────────────

function render() {
  renderSignalCards();
  renderPairTabs();
  renderBacktestTable();
  renderTradeLog();
  renderCharts();
  renderTimestamp();
}

function renderTimestamp() {
  const el = document.getElementById('last-updated');
  if (!el) return;

  if (signalsData && signalsData.generated_at) {
    const dt = new Date(signalsData.generated_at);
    el.textContent = `Last updated: ${dt.toUTCString()}`;
  } else {
    el.textContent = 'Using sample data';
  }
}

function renderSignalCards() {
  const container = document.getElementById('signal-cards');
  if (!container) return;

  const pairs = signalsData?.pairs || {};
  container.innerHTML = '';

  for (const pairCode of PAIRS) {
    const data = pairs[pairCode] || { pair: pairCode, regime: 'Unknown', confidence: 0, signal_direction: 'AVOID' };
    container.innerHTML += createSignalCard(data, pairCode);
  }
}

function createSignalCard(data, pairCode) {
  const direction = (data.signal_direction || 'AVOID').toLowerCase();
  const regimeClass = getRegimeClass(data.regime);
  const dirClass = direction;

  let detailsHtml = '';
  if (data.signal_direction === 'LONG' || data.signal_direction === 'SHORT') {
    const entry = data.entry_price ? data.entry_price.toFixed(5) : '—';
    const sl = data.stop_loss ? data.stop_loss.toFixed(5) : '—';
    const tp = data.take_profit ? data.take_profit.toFixed(5) : '—';

    detailsHtml = `
      <div class="signal-details">
        <div class="row"><span class="label">Strategy</span><span class="value">${data.active_strategy || '—'}</span></div>
        <div class="row"><span class="label">Entry</span><span class="value">${entry}</span></div>
        <div class="row"><span class="label">Stop Loss</span><span class="value">${sl}</span></div>
        <div class="row"><span class="label">Take Profit</span><span class="value">${tp}</span></div>
      </div>`;

    // Confirmations
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
    ? '<div style="margin-top:8px;font-size:12px;color:#d29922">⏳ Cooldown active</div>'
    : '';

  return `
    <div class="card signal-card ${direction}">
      <div class="pair-name">${pairCode}</div>
      <span class="regime-badge ${regimeClass}">${data.regime || 'Unknown'}</span>
      <div class="confidence">Confidence: ${(data.confidence || 0).toFixed(1)}%</div>
      <div class="signal-direction ${dirClass}">${(data.signal_direction || 'AVOID').toUpperCase()}</div>
      ${detailsHtml}
      ${cooldownHtml}
    </div>`;
}

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

function renderBacktestTable() {
  const tbody = document.getElementById('backtest-tbody');
  if (!tbody) return;

  if (backtestResults.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-muted)">No backtest results available. Run: python src/backtester.py --pair EURUSD --strategy both</td></tr>';
    return;
  }

  tbody.innerHTML = backtestResults.map(r => {
    const m = r.metrics || {};
    const retCls = (m.total_return_pct || 0) >= 0 ? 'positive' : 'negative';
    return `<tr>
      <td>${m.pair || '—'}</td>
      <td>${(m.strategy || '—').toUpperCase()}</td>
      <td class="${retCls}">${(m.total_return_pct || 0).toFixed(2)}%</td>
      <td>${(m.sharpe_ratio || 0).toFixed(2)}</td>
      <td class="negative">${(m.max_drawdown_pct || 0).toFixed(2)}%</td>
      <td>${(m.win_rate_pct || 0).toFixed(1)}%</td>
      <td>${m.total_trades || 0}</td>
    </tr>`;
  }).join('');
}

function renderTradeLog() {
  const tbody = document.getElementById('trade-tbody');
  if (!tbody) return;

  // Collect all trades from all backtest results
  const allTrades = [];
  for (const r of backtestResults) {
    if (r.trades) {
      allTrades.push(...r.trades);
    }
  }

  if (allTrades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text-muted)">No trades recorded yet</td></tr>';
    return;
  }

  // Show last 50 trades
  const recent = allTrades.slice(-50).reverse();
  tbody.innerHTML = recent.map(t => {
    const pnlCls = (t.pnl_pct || 0) >= 0 ? 'positive' : 'negative';
    const dir = t.direction || '—';
    return `<tr>
      <td>${t.pair || '—'}</td>
      <td>${t.strategy || '—'}</td>
      <td>${dir}</td>
      <td>${(t.entry_price || 0).toFixed(5)}</td>
      <td>${(t.exit_price || 0).toFixed(5)}</td>
      <td class="${pnlCls}">${(t.pnl_pct || 0).toFixed(3)}%</td>
      <td>${t.exit_reason || '—'}</td>
      <td>${t.duration_candles || 0}h</td>
    </tr>`;
  }).join('');
}

function renderCharts() {
  // Equity curve from selected pair's backtest
  const pairResult = backtestResults.find(r =>
    r.metrics && r.metrics.pair === selectedPair
  );

  if (pairResult && pairResult.equity_curve) {
    renderEquityCurve('equity-chart', pairResult.equity_curve);
  }

  // Regime distribution from selected pair's backtest
  if (pairResult && pairResult.metrics && pairResult.metrics.regime_distribution) {
    renderRegimeDistribution('regime-dist-chart', pairResult.metrics.regime_distribution);
  }
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

function getSampleSignals() {
  return {
    generated_at: new Date().toISOString(),
    pairs: {
      EURUSD: { pair: 'EURUSD', regime: 'Mean Reversion Range', confidence: 71.2, signal_direction: 'HOLD', regime_confirmed: true, cooldown_active: false },
      GBPUSD: { pair: 'GBPUSD', regime: 'Bull Trend', confidence: 83.5, signal_direction: 'HOLD', regime_confirmed: true, cooldown_active: false },
      USDJPY: { pair: 'USDJPY', regime: 'Noise', confidence: 58.3, signal_direction: 'AVOID', regime_confirmed: false, cooldown_active: false },
      USDCHF: { pair: 'USDCHF', regime: 'Value Gap Formation', confidence: 69.1, signal_direction: 'HOLD', regime_confirmed: true, cooldown_active: false },
    },
    signals_fired: 0,
  };
}

// Table sorting
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
