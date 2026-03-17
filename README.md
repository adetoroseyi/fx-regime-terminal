# FX Regime Terminal

A professional forex regime-based trading signal system. Uses **Hidden Markov Models (HMM)** to detect market regimes, then layers two strategies — **Value Gap (FVG)** and **Mean Reversion** — to generate trade signals for EUR/USD, GBP/USD, USD/JPY, and USD/CHF.

> **Disclaimer:** This is a signal research tool for educational purposes only. Not financial advice. No trades are placed automatically. Past performance does not guarantee future results.

---

## How It Works

### Regime Detection (Layer 1)

The system uses a **GaussianHMM** from `hmmlearn` to classify the market into 7 hidden states:

| Regime | Description |
|--------|-------------|
| Bull Trend | High positive returns, rising momentum |
| Bear Trend | High negative returns, falling momentum |
| High Volatility Expansion | Large range, volume spikes |
| Low Volatility Compression | Tight range, low activity |
| Mean Reversion Range | Oscillating, low directional bias |
| Value Gap Formation | Imbalance + momentum divergence |
| Noise / Choppy | Low signal — avoid trading |

The HMM is trained on 3 features per candle:
1. **Log returns** (close-to-close)
2. **Normalised range** (high - low) / close
3. **Volume change %**

Regimes are auto-labelled from cluster characteristics after training. The Viterbi algorithm decodes the most likely state sequence.

**Key concept:** HMMs don't predict price. They classify the *hidden state* the market is in right now. Strategies then act based on which state is active.

### Signal Hysteresis

A regime must be confirmed for **3 consecutive candles** before any strategy can fire. This prevents false signals at regime boundaries.

### Strategy Layer (Layer 2)

**Value Gap (FVG):** Detects institutional imbalances where price leaves an unfilled gap. Enters when price retraces into the gap in the correct regime (Bull Trend / Value Gap Formation for longs).

**Mean Reversion:** Uses Bollinger Bands, RSI, and Stochastic to identify stretched conditions in ranging regimes. Requires 3 out of 4 confirmations to fire.

Both strategies include regime-based exits — positions close immediately if the regime flips to an incompatible state.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/fx-regime-terminal.git
cd fx-regime-terminal
pip install -r requirements.txt
```

### 2. Configure Telegram (optional)

```bash
cp .env.example .env
# Edit .env with your bot credentials:
# TELEGRAM_BOT_TOKEN=your_bot_token
# TELEGRAM_CHAT_ID=your_chat_id
```

To create a Telegram bot:
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow the prompts
3. Copy the bot token to `.env`
4. Send a message to your bot, then use the Telegram API to get your chat ID

---

## Usage

### Run backtests

```bash
# Single pair
python -m src.backtester --pair EURUSD --strategy both

# Specific strategy
python -m src.backtester --pair GBPUSD --strategy fvg
python -m src.backtester --pair USDJPY --strategy mr

# All pairs
python -m src.backtester --all-pairs --strategy both

# Custom date range
python -m src.backtester --pair EURUSD --strategy both --days 365
```

Results are saved to `/results/`.

### Run signal engine

```bash
python -m src.signal_engine
```

Outputs current regime, signal direction, entry/SL/TP for all 4 pairs to `/state/current_signals.json`.

### Retrain models

```bash
python -m src.retrain
```

Retrains all HMM models using latest data from Yahoo Finance. Saves models to `/models/` and logs to `/logs/retrain_log.csv`.

### Send Telegram notifications

```bash
python -m src.telegram_notifier
```

### Run tests

```bash
pytest tests/ -v
```

---

## GitHub Actions

The system runs automatically via GitHub Actions at **00:05 UTC daily**:

1. Retrains all 4 HMM models
2. Generates fresh signals
3. Sends Telegram notifications
4. Commits updated state files back to the repo

### Required secrets

Set these in your repo's Settings → Secrets → Actions:

| Secret | Description |
|--------|-------------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID |

---

## Dashboard

The dashboard is a static web app (HTML + vanilla JS + Chart.js) — no build step required.

### View locally

Open `dashboard/index.html` in your browser. It reads data from `state/current_signals.json` and `results/`.

### Deploy to Vercel

1. Push the repo to GitHub
2. Go to [vercel.com](https://vercel.com) and import the repo
3. Set the **Root Directory** to `dashboard`
4. Deploy — no build configuration needed

---

## Project Structure

```
fx-regime-terminal/
├── .github/workflows/daily_run.yml   # Daily automation
├── src/
│   ├── data_loader.py                # yfinance fetch, clean, cache
│   ├── features.py                   # Log returns, range, volume change
│   ├── hmm_model.py                  # GaussianHMM train, decode, label
│   ├── indicators.py                 # RSI, BB, ATR, Stochastic, MACD
│   ├── strategy_fvg.py               # Value Gap detection + signals
│   ├── strategy_mr.py                # Mean Reversion signals
│   ├── backtester.py                 # Walk-forward backtester
│   ├── signal_engine.py              # Forward signal generation
│   ├── retrain.py                    # Daily retrain entry point
│   └── telegram_notifier.py          # Telegram alerts + summaries
├── models/                           # Trained .pkl files
├── state/                            # Current signals + cooldown
├── results/                          # Backtest output JSON
├── logs/                             # Retrain logs
├── dashboard/                        # Static web dashboard
│   ├── index.html
│   ├── vercel.json
│   ├── js/app.js
│   ├── js/charts.js
│   └── css/style.css
├── tests/                            # pytest suite
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Data Source

All market data is fetched from **Yahoo Finance** via the `yfinance` library. Hourly (H1) candles, up to 730 days of history per pair. No broker API is used — this is a pure signal generation system.

## Supported Pairs

- EUR/USD
- GBP/USD
- USD/JPY
- USD/CHF
