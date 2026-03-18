# Polymarket RF Trading Engine

**A reality-check implementation of the viral "100+ signals, 38 indicators — 80% win rate with AI and math" claims.**

This project takes the concepts from [that viral X/Twitter post](https://x.com/noisybly1) about using Random Forest models on Polymarket prediction markets, implements them properly with honest validation, and shows what actually happens when you do it right.

## What This Does

1. **Generates synthetic Polymarket contract data** with 38 features across 6 categories (price, volume, liquidity, time, market structure, derived)
2. **Trains a Random Forest classifier** with walk-forward time-series cross-validation (not random splits)
3. **Simulates trading** with realistic transaction costs, Kelly position sizing, and MAE/MFE tracking
4. **Reports honest metrics** including Sharpe Ratio, max drawdown, and profit factor

## The Critique

| Claim | Reality |
|-------|---------|
| 80% win rate | ~56% with proper validation |
| 100+ signals, 38 indicators | Code shows 5 features |
| Sigmoid output function | RF doesn't use sigmoid — that's logistic regression |
| `market_price ≤ model_prob × 0.5` | Almost never triggers; ignores transaction costs |
| "$20,000+ a week" | Unverifiable; leads to Telegram channel |
| No train/test split shown | Classic overfitting setup |

**What's actually correct:** Random Forest ensemble mechanics, √n feature selection, Sharpe Ratio as risk-adjusted metric, log returns being additive, MAE/MFE trade diagnostics, Kelly Criterion for sizing.

## Project Structure

```
├── polymarket_rf.py      # Full trading engine (data gen → train → simulate → metrics)
├── dashboard.jsx          # React dashboard with 4 tabs (Overview, Model, Trades, Risk)
├── dashboard_data.json    # Pre-computed results for the dashboard
└── README.md
```

## Quick Start

### Python Engine
```bash
pip install numpy pandas scikit-learn
python polymarket_rf.py
```

### Dashboard
The `dashboard.jsx` is a self-contained React component using Recharts. It can be rendered in any React environment or in Claude.ai's artifact system.

## Key Results (Honest)

| Metric | Value |
|--------|-------|
| Win Rate | 56.0% |
| Total Trades | 109 |
| Total P&L | $2,236.54 |
| Sharpe Ratio | -12.53 (bad) |
| Max Drawdown | 28.9% |
| Profit Factor | 1.08 |
| Avg Win | $493.39 |
| Avg Loss | -$580.42 |

The strategy is marginally profitable but the negative Sharpe Ratio and the fact that average losses exceed average wins means this is not a viable trading system.

## Features Used (all 38)

**Price (8):** price, price_ma_7, price_ma_30, price_std_7, price_std_30, momentum_7d, momentum_30d, price_range_7d

**Volume (6):** volume_24h, volume_7d_avg, volume_ratio, buy_volume_pct, trade_count_24h, avg_trade_size

**Liquidity (5):** liquidity, spread, depth_yes, depth_no, depth_imbalance

**Time (5):** days_to_expiry, hours_since_last_trade, time_decay, days_since_creation, activity_recency

**Market Structure (6):** n_unique_traders, whale_pct, retail_sentiment, open_interest, maker_ratio, price_impact_1k

**Derived (8):** vol_price_corr, momentum_volume_interaction, price_deviation_from_ma, normalized_volume, time_weighted_momentum, liquidity_adjusted_spread, whale_momentum, sentiment_divergence

## License

MIT — do whatever you want with it. Just don't claim 80% win rates.
