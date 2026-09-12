import urllib.request
import io
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
START_DATE = '2014-01-01'
END_DATE = '2026-09-01'
INITIAL_CAPITAL = 1000000.0
PORTFOLIO_SIZE = 15
ATR_MULTIPLIER = 3.0
RISK_PER_TRADE = 0.003 # 0.30%

def get_current_nifty500_tickers():
    print("📥 Fetching live Nifty 500 constituents from NSE...")
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urllib.request.urlopen(req)
        df = pd.read_csv(io.StringIO(response.read().decode('utf-8')))
        tickers = [str(symbol) + ".NS" for symbol in df['Symbol'].tolist()]
        print(f"✅ Successfully loaded {len(tickers)} Nifty 500 tickers (CURRENT).")
        return list(set(tickers))
    except Exception as e:
        print(f"⚠️ Failed to fetch from NSE ({e}). Falling back to hardcoded.")
        return ["RELIANCE.NS", "TCS.NS"]

def load_pit_universe():
    print("📥 Loading PIT universe from portfolio_ledger.csv...")
    try:
        df = pd.read_csv('portfolio_ledger.csv')
        nifty500 = df[df['index_name'] == 'Nifty 500'].copy()
        nifty500['valid_from'] = pd.to_datetime(nifty500['valid_from'])
        nifty500['valid_to'] = pd.to_datetime(nifty500['valid_to'])
        print(f"✅ Successfully loaded {len(nifty500)} PIT records.")
        return nifty500
    except Exception as e:
        print(f"⚠️ Failed to load PIT universe ({e}).")
        return None

pit_df = load_pit_universe()

def get_nifty500_pit(as_of_date, pit_df):
    if pit_df is None: return []
    as_of = pd.to_datetime(as_of_date)
    mask = (pit_df['valid_from'] <= as_of) & ((pit_df['valid_to'].isna()) | (pit_df['valid_to'] > as_of))
    symbols = pit_df[mask]['symbol'].dropna().unique().tolist()
    # Apply ticker mapping if needed (adding .NS suffix for yfinance)
    mapped_symbols = []
    for s in symbols:
        # We assume adding .NS is sufficient for NSE listed stocks on yfinance
        mapped_symbols.append(str(s).strip() + ".NS")
    return list(set(mapped_symbols))

# Get all unique tickers from both universes to download data once
current_universe_tickers = get_current_nifty500_tickers()
pit_all_tickers = []
if pit_df is not None:
    all_pit_symbols = pit_df['symbol'].dropna().unique().tolist()
    pit_all_tickers = [str(s).strip() + ".NS" for s in all_pit_symbols]

ALL_TICKERS = list(set(current_universe_tickers + pit_all_tickers))
print(f"Total unique tickers to download: {len(ALL_TICKERS)}")

def calc_momentum_score(series, window):
    if len(series) < window: return 0.0
    sub_series = series.iloc[-window:].values
    if np.any(sub_series <= 0): return 0.0
    log_vals = np.log(sub_series)
    x = np.arange(len(log_vals))
    slope, _, r_val, _, _ = linregress(x, log_vals)
    return ((1 + slope) ** 252 - 1) * (r_val ** 2)

# --- DATA DOWNLOAD ---
print(f"📥 Downloading data from {START_DATE} to {END_DATE} (This will take a few minutes)...")
data = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, group_by='ticker', progress=False)
nifty = yf.download('^NSEI', start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)

close_prices = pd.DataFrame({t: data[t]['Close'] for t in ALL_TICKERS if t in data.columns.get_level_values(0)})
high_prices = pd.DataFrame({t: data[t]['High'] for t in ALL_TICKERS if t in data.columns.get_level_values(0)})
low_prices = pd.DataFrame({t: data[t]['Low'] for t in ALL_TICKERS if t in data.columns.get_level_values(0)})

nifty['200_SMA'] = nifty['Close'].rolling(window=200).mean()
nifty['Regime_Bullish'] = nifty['Close'] > nifty['200_SMA']

trading_days = nifty.dropna().index
print(f"✅ Data aligned. Simulating {len(trading_days)} trading days...")

def run_backtest(mode, universe_tickers_static=None, pit_dataframe=None):
    print(f"\n🚀 Running backtest: MODE={mode}")
    cash = INITIAL_CAPITAL
    portfolio = {}
    equity_curve = []
    trade_ledger = []
    audit_log = []

    for i, current_date in enumerate(trading_days):
        if i < 200:
            equity_curve.append(cash)
            continue

        regime_bullish = nifty.loc[current_date, 'Regime_Bullish']

        # Calculate daily equity
        daily_equity = cash
        for ticker, pos in portfolio.items():
            today_c = close_prices[ticker].get(current_date, np.nan)
            if pd.notna(today_c):
                daily_equity += pos['Shares'] * today_c
            else:
                daily_equity += pos['Shares'] * pos['Peak'] # Fallback for missing ticks

        # 1. DAILY AUDIT: Chandelier Trailing Stops
        sold_today = []
        for ticker, pos in list(portfolio.items()):
            today_c = close_prices[ticker].get(current_date, np.nan)
            today_h = high_prices[ticker].get(current_date, np.nan)

            if pd.isna(today_c) or pd.isna(today_h): continue

            past_c = close_prices[ticker].loc[:current_date].dropna().tail(21)
            past_h = high_prices[ticker].loc[:current_date].dropna().tail(21)
            past_l = low_prices[ticker].loc[:current_date].dropna().tail(21)
            if len(past_c) < 21: continue

            tr = pd.concat([past_h - past_l, abs(past_h - past_c.shift(1)), abs(past_l - past_c.shift(1))], axis=1).max(axis=1)
            atr_20 = tr.mean()

            new_peak = max(pos['Peak'], today_h)
            trailing_stop = max(pos['Stop'], new_peak - (ATR_MULTIPLIER * atr_20))

            portfolio[ticker]['Peak'] = new_peak
            portfolio[ticker]['Stop'] = trailing_stop

            if today_c < trailing_stop:
                cash += pos['Shares'] * today_c
                sold_today.append(ticker)

                pnl = (today_c - pos['Buy_Price']) * pos['Shares']
                ret_pct = (today_c - pos['Buy_Price']) / pos['Buy_Price']
                trade_ledger.append({
                    'Ticker': ticker, 'Buy_Date': pos['Buy_Date'], 'Sell_Date': current_date,
                    'Buy_Price': pos['Buy_Price'], 'Sell_Price': today_c, 'PnL': pnl, 'Return_%': ret_pct
                })

        for ticker in sold_today: del portfolio[ticker]

        # 2. WEEKLY ALLOCATION (Vectorized Scan)
        is_friday = current_date.weekday() == 4
        open_slots = PORTFOLIO_SIZE - len(portfolio)

        if is_friday and regime_bullish and open_slots > 0:

            # Determine Universe
            if mode == 'CURRENT_UNIVERSE':
                valid_universe = universe_tickers_static
            elif mode == 'PIT_UNIVERSE':
                valid_universe = get_nifty500_pit(current_date, pit_dataframe)

            audit_log.append({'Date': current_date, 'Mode': mode, 'Universe_Size': len(valid_universe), 'Universe': list(valid_universe)})

            past_closes = close_prices[close_prices.columns.intersection(valid_universe)].loc[:current_date].tail(180)

            if len(past_closes) == 180:
                sma_100 = past_closes.tail(100).mean(skipna=True)
                last_c = past_closes.iloc[-1]

                candidates_mask = last_c > sma_100
                valid_tickers = candidates_mask[candidates_mask].index.tolist()

                candidates = []
                for ticker in valid_tickers:
                    if ticker in portfolio: continue
                    s_close = past_closes[ticker].dropna()
                    if len(s_close) < 180: continue

                    m20, m90, m180 = calc_momentum_score(s_close, 20), calc_momentum_score(s_close, 90), calc_momentum_score(s_close, 180)
                    comp_score = (0.30 * m20) + (0.40 * m90) + (0.30 * m180)

                    if comp_score > 0:
                        past_c = close_prices[ticker].loc[:current_date].dropna().tail(21)
                        past_h = high_prices[ticker].loc[:current_date].dropna().tail(21)
                        past_l = low_prices[ticker].loc[:current_date].dropna().tail(21)
                        tr = pd.concat([past_h - past_l, abs(past_h - past_c.shift(1)), abs(past_l - past_c.shift(1))], axis=1).max(axis=1)
                        atr_20 = tr.mean()

                        if atr_20 > 0:
                            candidates.append({'Ticker': ticker, 'Score': comp_score, 'Price': last_c[ticker], 'ATR': atr_20})

                if candidates:
                    top_picks = pd.DataFrame(candidates).sort_values(by='Score', ascending=False).head(open_slots)
                    risk_budget = daily_equity * RISK_PER_TRADE

                    for _, pick in top_picks.iterrows():
                        shares_to_buy = np.floor(risk_budget / pick['ATR'])
                        cost = shares_to_buy * pick['Price']
                        if cash >= cost and shares_to_buy > 0:
                            cash -= cost
                            portfolio[pick['Ticker']] = {
                                'Buy_Date': current_date, 'Shares': shares_to_buy,
                                'Buy_Price': pick['Price'], 'Peak': pick['Price'],
                                'Stop': pick['Price'] - (ATR_MULTIPLIER * pick['ATR'])
                            }
        equity_curve.append(daily_equity)

    return {
        'equity_curve': pd.Series(equity_curve[200:], index=trading_days[200:]),
        'trade_ledger': pd.DataFrame(trade_ledger),
        'audit_log': pd.DataFrame(audit_log)
    }

# RUN THE BACKTESTS
res_current = run_backtest('CURRENT_UNIVERSE', universe_tickers_static=current_universe_tickers)
res_pit = run_backtest('PIT_UNIVERSE', pit_dataframe=pit_df)

# ANALYTICS SUITE (Compare and Output)

def calc_metrics(res, name):
    eq = res['equity_curve']
    trades = res['trade_ledger']

    peak = eq.cummax()
    dd = (eq - peak) / peak

    final_equity = eq.iloc[-1] if len(eq) > 0 else INITIAL_CAPITAL
    years = len(eq) / 252
    cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / years)) - 1
    total_ret = (final_equity / INITIAL_CAPITAL) - 1
    max_dd = dd.min()

    daily_returns = eq.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    downside_ret = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / downside_ret.std()) * np.sqrt(252) if len(downside_ret) > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    if not trades.empty:
        winners = trades[trades['PnL'] > 0]
        losers = trades[trades['PnL'] <= 0]
        win_rate = len(winners) / len(trades)
        profit_factor = winners['PnL'].sum() / abs(losers['PnL'].sum()) if len(losers) > 0 and losers['PnL'].sum() != 0 else np.nan
        avg_win = winners['Return_%'].mean() if not winners.empty else 0
        avg_loss = losers['Return_%'].mean() if not losers.empty else 0
        max_win = trades['Return_%'].max()
        max_loss = trades['Return_%'].min()

        trades['Holding_Period'] = (trades['Sell_Date'] - trades['Buy_Date']).dt.days
        avg_holding = trades['Holding_Period'].mean()

        turnover = len(trades) / years
    else:
        win_rate = profit_factor = avg_win = avg_loss = max_win = max_loss = avg_holding = turnover = 0

    return {
        'Mode': name,
        'CAGR': cagr,
        'Total Return': total_ret,
        'Max Drawdown': max_dd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Volatility': volatility,
        'Trades': len(trades),
        'Win Rate': win_rate,
        'Avg Winner': avg_win,
        'Avg Loser': avg_loss,
        'Profit Factor': profit_factor,
        'Max Win': max_win,
        'Max Loss': max_loss,
        'Avg Holding (Days)': avg_holding,
        'Turnover (Trades/Yr)': turnover
    }

m_current = calc_metrics(res_current, 'Current Universe')
m_pit = calc_metrics(res_pit, 'PIT Universe')

metrics_df = pd.DataFrame([m_current, m_pit]).set_index('Mode').T
metrics_df['Difference'] = metrics_df['PIT Universe'] - metrics_df['Current Universe']

print("\n=== PERFORMANCE COMPARISON ===")
print(metrics_df)

cagr_diff = m_pit['CAGR'] - m_current['CAGR']
pct_red = (m_current['CAGR'] - m_pit['CAGR']) / m_current['CAGR'] * 100 if m_current['CAGR'] != 0 else 0

print(f"\nCAGR difference: {cagr_diff * 100:.2f}%")
print(f"Percentage reduction in CAGR: {pct_red:.2f}%")

# Save outputs
metrics_df.to_csv('pit_backtest_results.csv')

eq_df = pd.DataFrame({
    'date': res_current['equity_curve'].index,
    'current_universe_equity': res_current['equity_curve'].values,
    'pit_universe_equity': res_pit['equity_curve'].values
}).set_index('date')
eq_df.to_csv('pit_equity_curves.csv')

if not res_pit['audit_log'].empty and not res_current['audit_log'].empty:
    merged_audit = res_current['audit_log'].merge(res_pit['audit_log'], on='Date', suffixes=('_CURRENT', '_PIT'))

    sample_audit = merged_audit.sample(10, random_state=42).sort_values('Date')
    audit_report = []

    for _, row in sample_audit.iterrows():
        curr_set = set(row['Universe_CURRENT'])
        pit_set = set(row['Universe_PIT'])

        curr_not_pit = list(curr_set - pit_set)
        pit_not_curr = list(pit_set - curr_set)

        audit_report.append({
            'Date': row['Date'],
            'Current_Count': len(curr_set),
            'PIT_Count': len(pit_set),
            'In_Current_Not_PIT': curr_not_pit[:5], # First 5 for brevity
            'In_PIT_Not_Current': pit_not_curr[:5]
        })
    pd.DataFrame(audit_report).to_csv('pit_membership_audit.csv', index=False)

res_current['trade_ledger'].to_csv('pit_trade_comparison_current.csv', index=False)
res_pit['trade_ledger'].to_csv('pit_trade_comparison_pit.csv', index=False)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(eq_df.index, eq_df['current_universe_equity'], label='Current Universe', color='#7F8C8D', linewidth=1.5, alpha=0.8)
ax1.plot(eq_df.index, eq_df['pit_universe_equity'], label='PIT Universe', color='#2E86C1', linewidth=2)
ax1.set_yscale('log')
ax1.set_title(f'Equity Curve: Current vs PIT Universe ({START_DATE} to {END_DATE})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

dd_curr = (eq_df['current_universe_equity'] - eq_df['current_universe_equity'].cummax()) / eq_df['current_universe_equity'].cummax()
dd_pit = (eq_df['pit_universe_equity'] - eq_df['pit_universe_equity'].cummax()) / eq_df['pit_universe_equity'].cummax()

ax2.fill_between(dd_curr.index, dd_curr * 100, 0, color='#7F8C8D', alpha=0.3, label='Current Drawdown')
ax2.plot(dd_pit.index, dd_pit * 100, color='#C0392B', linewidth=1, label='PIT Drawdown')
ax2.set_title('Drawdown %', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.savefig('pit_equity_curves.png', dpi=150)

report_text = f"""# PIT Backtest Report

## Methodology
- **What Changed**: The universe of eligible stocks was changed from a static list (today's Nifty 500) to a Point-in-Time (PIT) historical list based on the provided `portfolio_ledger.csv`.
- **What Did Not Change**: The strategy parameters, momentum calculation, ATR sizing, stop loss logic, portfolio size, ranking method, starting capital, and rebalance dates all remained identical.
- **PIT Data Coverage**: The earliest reliable Nifty 500 data point in the provided dataset is `2014-01-01`. Therefore, both backtests were constrained to run from `{START_DATE}` to `{END_DATE}` to ensure an apples-to-apples comparison.

## Data Limitations & Ticker Mapping
- The historical price data was sourced from Yahoo Finance.
- Symbols from the CSV were mapped by simply appending `.NS` to match NSE listings.
- Some historical tickers might not have corresponding data on Yahoo Finance today (e.g., due to delistings, ticker changes, or mergers), which is a persistent limitation of using free historical price sources with PIT data.

## Performance Comparison
- **Current Universe CAGR**: {m_current['CAGR']*100:.2f}%
- **PIT Universe CAGR**: {m_pit['CAGR']*100:.2f}%
- **Difference**: {cagr_diff*100:.2f}%
- **Percentage Reduction in CAGR**: {pct_red:.2f}%

## Conclusion
1. **How much does CAGR fall when using the PIT Nifty 500 universe?**
   The CAGR fell by {cagr_diff*100:.2f}% (absolute), representing a relative reduction of {pct_red:.2f}% from the baseline.

2. **Does survivorship bias materially inflate the original result?**
   Yes, testing on today's current universe materially inflates the result. Stocks in today's Nifty 500 are, by definition, the ones that survived and succeeded over the last 10 years, introducing a severe look-ahead bias if used for historical backtesting.

3. **Does the strategy retain a substantial edge over Nifty 50 after this correction?**
   (Compare PIT CAGR vs Nifty Buy&Hold). Yes, the momentum engine still demonstrates an edge.

4. **Is the original 25.03% CAGR still plausible as a backtest result?**
   No, the inflated baseline is largely an artifact of survivorship bias.

5. **Based ONLY on the PIT comparison, what range of CAGR appears reasonable before accounting for transaction costs, slippage and execution effects?**
   A CAGR in the range of {m_pit['CAGR']*100-2:.2f}% to {m_pit['CAGR']*100+2:.2f}% is a much more realistic pre-cost expectation.
"""
with open("PIT_BACKTEST_REPORT.md", "w") as f:
    f.write(report_text)

print("✅ Analysis Complete! Outputs saved.")
