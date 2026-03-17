/**
 * FX Regime Terminal — Chart.js Visualisations
 *
 * Provides chart rendering for:
 * - Regime price overlay charts
 * - Equity curve
 * - Regime distribution doughnut
 */

// Regime colour mapping
const REGIME_COLORS = {
  'Bull Trend': 'rgba(63, 185, 80, 0.25)',
  'Bear Trend': 'rgba(248, 81, 73, 0.25)',
  'Mean Reversion Range': 'rgba(88, 166, 255, 0.25)',
  'Value Gap Formation': 'rgba(210, 153, 34, 0.25)',
  'Noise': 'rgba(110, 118, 129, 0.25)',
  'Low Volatility Compression': 'rgba(227, 179, 65, 0.25)',
  'High Volatility Expansion': 'rgba(188, 140, 255, 0.25)',
};

const REGIME_BORDER_COLORS = {
  'Bull Trend': '#3fb950',
  'Bear Trend': '#f85149',
  'Mean Reversion Range': '#58a6ff',
  'Value Gap Formation': '#d29922',
  'Noise': '#6e7681',
  'Low Volatility Compression': '#e3b341',
  'High Volatility Expansion': '#bc8cff',
};

const DOUGHNUT_COLORS = [
  '#3fb950', '#f85149', '#bc8cff', '#e3b341',
  '#58a6ff', '#d29922', '#6e7681'
];

// Store chart instances for cleanup
const chartInstances = {};

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

/**
 * Render a price chart with regime colour background overlay.
 */
function renderRegimeChart(canvasId, equityCurve, title) {
  destroyChart(canvasId);

  const canvas = document.getElementById(canvasId);
  if (!canvas || !equityCurve || equityCurve.length === 0) return;

  const labels = equityCurve.map(d => {
    const dt = new Date(d.datetime);
    return dt.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' });
  });
  const values = equityCurve.map(d => d.equity);

  // Create regime background segments
  const backgroundColors = equityCurve.map(d =>
    REGIME_COLORS[d.regime] || 'rgba(110, 118, 129, 0.1)'
  );

  const ctx = canvas.getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: title || 'Equity',
        data: values,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88, 166, 255, 0.05)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.1,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: { color: '#8b949e', font: { size: 11 } }
        },
        tooltip: {
          backgroundColor: '#1c2128',
          titleColor: '#e6edf3',
          bodyColor: '#8b949e',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {
            afterBody: function(items) {
              const idx = items[0].dataIndex;
              const regime = equityCurve[idx].regime;
              return `Regime: ${regime}`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#484f58', maxTicksLimit: 12, font: { size: 10 } },
          grid: { color: 'rgba(48, 54, 61, 0.5)' },
        },
        y: {
          ticks: { color: '#484f58', font: { size: 10 } },
          grid: { color: 'rgba(48, 54, 61, 0.5)' },
        }
      }
    }
  });
}

/**
 * Render equity curve line chart.
 */
function renderEquityCurve(canvasId, equityCurve) {
  renderRegimeChart(canvasId, equityCurve, 'Portfolio Equity');
}

/**
 * Render regime distribution doughnut chart.
 */
function renderRegimeDistribution(canvasId, regimeDistribution) {
  destroyChart(canvasId);

  const canvas = document.getElementById(canvasId);
  if (!canvas || !regimeDistribution) return;

  const labels = Object.keys(regimeDistribution);
  const values = Object.values(regimeDistribution);
  const total = values.reduce((a, b) => a + b, 0);

  const colors = labels.map((label, i) =>
    REGIME_BORDER_COLORS[label] || DOUGHNUT_COLORS[i % DOUGHNUT_COLORS.length]
  );

  const ctx = canvas.getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: colors.map(c => c + '33'),
        borderColor: colors,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: {
            color: '#8b949e',
            font: { size: 11 },
            padding: 12,
            generateLabels: function(chart) {
              const data = chart.data;
              return data.labels.map((label, i) => {
                const value = data.datasets[0].data[i];
                const pct = ((value / total) * 100).toFixed(1);
                return {
                  text: `${label} (${pct}%)`,
                  fillStyle: data.datasets[0].backgroundColor[i],
                  strokeStyle: data.datasets[0].borderColor[i],
                  lineWidth: 2,
                  index: i,
                };
              });
            }
          }
        },
        tooltip: {
          backgroundColor: '#1c2128',
          titleColor: '#e6edf3',
          bodyColor: '#8b949e',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {
            label: function(item) {
              const pct = ((item.raw / total) * 100).toFixed(1);
              return ` ${item.label}: ${item.raw} candles (${pct}%)`;
            }
          }
        }
      }
    }
  });
}

/**
 * Render a simple bar chart for backtest comparison.
 */
function renderBacktestComparison(canvasId, backtestData) {
  destroyChart(canvasId);

  const canvas = document.getElementById(canvasId);
  if (!canvas || !backtestData || backtestData.length === 0) return;

  const labels = backtestData.map(d => `${d.pair} (${d.strategy})`);
  const returns = backtestData.map(d => d.total_return_pct);
  const colors = returns.map(v => v >= 0 ? 'rgba(63, 185, 80, 0.7)' : 'rgba(248, 81, 73, 0.7)');

  const ctx = canvas.getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Total Return %',
        data: returns,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace('0.7', '1')),
        borderWidth: 1,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1c2128',
          titleColor: '#e6edf3',
          bodyColor: '#8b949e',
        }
      },
      scales: {
        x: {
          ticks: { color: '#484f58', font: { size: 10 } },
          grid: { color: 'rgba(48, 54, 61, 0.5)' },
        },
        y: {
          ticks: { color: '#484f58', font: { size: 10 } },
          grid: { color: 'rgba(48, 54, 61, 0.5)' },
        }
      }
    }
  });
}
