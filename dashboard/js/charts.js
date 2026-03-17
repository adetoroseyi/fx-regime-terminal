/**
 * FX Regime Terminal — Chart.js Visualisations
 *
 * Charts:
 * - Equity curve with regime overlay
 * - Regime distribution doughnut
 * - BIC model selection bar chart
 * - Forward test confidence timeline
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

// Common chart options
const COMMON_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    tooltip: {
      backgroundColor: '#1c2128',
      titleColor: '#e6edf3',
      bodyColor: '#8b949e',
      borderColor: '#30363d',
      borderWidth: 1,
    },
    legend: {
      labels: { color: '#8b949e', font: { size: 11 } }
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
};

// Store chart instances for cleanup
const chartInstances = {};

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

/**
 * Render empty chart placeholder.
 */
function renderEmptyChart(canvasId, message) {
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#484f58';
  ctx.font = '14px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

/**
 * Render equity curve with regime color overlay.
 */
function renderEquityCurve(canvasId, equityCurve) {
  destroyChart(canvasId);

  const canvas = document.getElementById(canvasId);
  if (!canvas || !equityCurve || equityCurve.length === 0) return;

  const labels = equityCurve.map(d => {
    const dt = new Date(d.datetime);
    return dt.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' });
  });
  const values = equityCurve.map(d => d.equity);

  const ctx = canvas.getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Portfolio Equity',
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
      ...COMMON_OPTS,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        ...COMMON_OPTS.plugins,
        tooltip: {
          ...COMMON_OPTS.plugins.tooltip,
          callbacks: {
            afterBody(items) {
              const idx = items[0].dataIndex;
              const regime = equityCurve[idx].regime;
              return `Regime: ${regime}`;
            },
            label(item) {
              return `Equity: $${item.raw.toFixed(2)}`;
            }
          }
        }
      }
    }
  });
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
      labels,
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
            generateLabels(chart) {
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
          ...COMMON_OPTS.plugins.tooltip,
          callbacks: {
            label(item) {
              const pct = ((item.raw / total) * 100).toFixed(1);
              return ` ${item.label}: ${item.raw.toLocaleString()} candles (${pct}%)`;
            }
          }
        }
      }
    }
  });
}

/**
 * Render BIC bar chart for model selection.
 */
function renderBICBarChart(canvasId, nStates, bics, bestN) {
  destroyChart(canvasId);

  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const colors = nStates.map(n =>
    n === bestN ? 'rgba(63, 185, 80, 0.7)' : 'rgba(88, 166, 255, 0.4)'
  );
  const borders = nStates.map(n =>
    n === bestN ? '#3fb950' : '#58a6ff'
  );

  const ctx = canvas.getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: nStates.map(n => `${n} states`),
      datasets: [{
        label: 'BIC',
        data: bics,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 2,
      }]
    },
    options: {
      ...COMMON_OPTS,
      plugins: {
        ...COMMON_OPTS.plugins,
        legend: { display: false },
        tooltip: {
          ...COMMON_OPTS.plugins.tooltip,
          callbacks: {
            afterLabel(item) {
              const n = nStates[item.dataIndex];
              return n === bestN ? '← Selected (lowest BIC)' : '';
            }
          }
        }
      }
    }
  });
}

/**
 * Render forward test confidence timeline.
 */
function renderConfidenceTimeline(canvasId, pairData) {
  destroyChart(canvasId);

  const canvas = document.getElementById(canvasId);
  if (!canvas || !pairData || pairData.length === 0) return;

  const labels = pairData.map(d => {
    const dt = new Date(d.timestamp);
    return dt.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' });
  });

  const confidences = pairData.map(d => d.confidence);
  const bgColors = pairData.map(d => {
    const regime = d.regime;
    return REGIME_COLORS[regime] || 'rgba(110, 118, 129, 0.25)';
  });

  // Signal markers
  const signalPoints = pairData.map(d => {
    if (d.signal === 'LONG' || d.signal === 'SHORT') return d.confidence;
    return null;
  });

  const ctx = canvas.getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Regime Confidence %',
          data: confidences,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88, 166, 255, 0.1)',
          borderWidth: 2,
          pointRadius: 2,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'Signals Fired',
          data: signalPoints,
          borderColor: 'transparent',
          backgroundColor: '#3fb950',
          pointRadius: 6,
          pointStyle: 'triangle',
          showLine: false,
        }
      ]
    },
    options: {
      ...COMMON_OPTS,
      interaction: { mode: 'index', intersect: false },
      scales: {
        ...COMMON_OPTS.scales,
        y: {
          ...COMMON_OPTS.scales.y,
          min: 0,
          max: 100,
          ticks: {
            ...COMMON_OPTS.scales.y.ticks,
            callback: v => v + '%',
          }
        }
      },
      plugins: {
        ...COMMON_OPTS.plugins,
        tooltip: {
          ...COMMON_OPTS.plugins.tooltip,
          callbacks: {
            afterBody(items) {
              const idx = items[0].dataIndex;
              const d = pairData[idx];
              return `Regime: ${d.regime}\nSignal: ${d.signal}`;
            }
          }
        }
      }
    }
  });
}
