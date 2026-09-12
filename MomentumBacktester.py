import urllib.request
import io
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
START_DATE = '2007-01-01'
END_DATE = '2026-09-01'
INITIAL_CAPITAL = 1000000.0
PORTFOLIO_SIZE = 15
ATR_MULTIPLIER = 3.0
RISK_PER_TRADE = 0.003 # 0.30%

def get_nifty500_tickers():
    print("📥 Fetching live Nifty 500 constituents from NSE...")
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urllib.request.urlopen(req)
        df = pd.read_csv(io.StringIO(response.read().decode('utf-8')))
        tickers = [str(symbol) + ".NS" for symbol in df['Symbol'].tolist()]
        print(f"✅ Successfully loaded {len(tickers)} Nifty 500 tickers.")
        return list(set(tickers))
    except Exception as e:
        print(f"⚠️ Failed to fetch from NSE ({e}). Falling back to Nifty 50 core.")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS", "LT.NS"]

TICKERS = get_nifty500_tickers()

def calc_momentum_score(series, window):
    if len(series) < window: return 0.0
    sub_series = series.iloc[-window:].values
    if np.any(sub_series <= 0): return 0.0
    log_vals = np.log(sub_series)
    x = np.arange(len(log_vals))
    slope, _, r_val, _, _ = linregress(x, log_vals)
    return ((1 + slope) ** 252 - 1) * (r_val ** 2)

# --- DATA DOWNLOAD ---
print("📥 Downloading 20 years of data for 500 stocks (This will take ~2-3 minutes)...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by='ticker', progress=False)
nifty = yf.download('^NSEI', start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)

# Reformat multi-index safely
close_prices = pd.DataFrame({t: data[t]['Close'] for t in TICKERS if t in data.columns.get_level_values(0)})
high_prices = pd.DataFrame({t: data[t]['High'] for t in TICKERS if t in data.columns.get_level_values(0)})
low_prices = pd.DataFrame({t: data[t]['Low'] for t in TICKERS if t in data.columns.get_level_values(0)})

nifty['200_SMA'] = nifty['Close'].rolling(window=200).mean()
nifty['Regime_Bullish'] = nifty['Close'] > nifty['200_SMA']

trading_days = nifty.dropna().index
print(f"✅ Data aligned. Simulating {len(trading_days)} trading days...")

# --- STATE VARIABLES ---
cash = INITIAL_CAPITAL
portfolio = {}
equity_curve = []
trade_ledger = []

# --- VECTORIZED TIME MACHINE LOOP ---
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

        # Fast ATR-20 Calculation
        past_c = close_prices[ticker].loc[:current_date].dropna().tail(21)
        past_h = high_prices[ticker].loc[:current_date].dropna().tail(21)
        past_l = low_prices[ticker].loc[:current_date].dropna().tail(21)
        if len(past_c) < 21: continue

        tr = pd.concat([past_h - past_l, abs(past_h - past_c.shift(1)), abs(past_l - past_c.shift(1))], axis=1).max(axis=1)
        atr_20 = tr.mean()

        # Ratchet Stops
        new_peak = max(pos['Peak'], today_h)
        trailing_stop = max(pos['Stop'], new_peak - (ATR_MULTIPLIER * atr_20))

        portfolio[ticker]['Peak'] = new_peak
        portfolio[ticker]['Stop'] = trailing_stop

        # Trigger Stop Loss & Log Trade
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
        past_closes = close_prices.loc[:current_date].tail(180)

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

# --- ANALYTICS SUITE ---
results = pd.DataFrame({'Date': trading_days[200:], 'Equity': equity_curve[200:]}).set_index('Date')
results['Peak'] = results['Equity'].cummax()
results['Drawdown'] = (results['Equity'] - results['Peak']) / results['Peak']

# Core Metrics
final_equity = results['Equity'].iloc[-1]
cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / (len(results) / 252))) - 1
max_dd = results['Drawdown'].min()

# Benchmark & Risk
bm_init, bm_final = nifty['Close'].loc[results.index[0]], nifty['Close'].loc[results.index[-1]]
bm_cagr = ((bm_final / bm_init) ** (1 / (len(results) / 252))) - 1
daily_returns = results['Equity'].pct_change().dropna()
sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
calmar_ratio = cagr / abs(max_dd) if max_dd != 0 else 0

# Trade Ledger Stats
trades_df = pd.DataFrame(trade_ledger)
if not trades_df.empty:
    winners = trades_df[trades_df['PnL'] > 0]
    losers = trades_df[trades_df['PnL'] <= 0]
    win_rate = len(winners) / len(trades_df)
    profit_factor = winners['PnL'].sum() / abs(losers['PnL'].sum()) if len(losers) > 0 else 0
    avg_win_pct = winners['Return_%'].mean() if len(winners) > 0 else 0
    avg_loss_pct = losers['Return_%'].mean() if len(losers) > 0 else 0
else:
    win_rate = profit_factor = avg_win_pct = avg_loss_pct = 0

# Annual Returns
results['Year'] = results.index.year
yearly_equity = results.groupby('Year')['Equity'].last()
annual_returns = yearly_equity.pct_change()
annual_returns.iloc[0] = (yearly_equity.iloc[0] / INITIAL_CAPITAL) - 1

summary_text = f"""
--------------------------------------------------
QUANTITATIVE BACKTEST REPORT (2007 - 2026)
Universe: Nifty 500 Constituents
--------------------------------------------------
Initial Capital:   ₹{INITIAL_CAPITAL:,.2f}
Final Equity:      ₹{final_equity:,.2f}
CAGR (System):     {cagr * 100:.2f}%
CAGR (Nifty 50):   {bm_cagr * 100:.2f}%
Alpha Generated:   {(cagr - bm_cagr) * 100:+.2f}%

RISK & DRAWDOWN METRICS
Max Drawdown:      {max_dd * 100:.2f}%
Sharpe Ratio:      {sharpe_ratio:.2f}
Calmar Ratio:      {calmar_ratio:.2f}

TRADE LEDGER SUMMARY
Total Trades:      {len(trades_df)}
Win Rate:          {win_rate * 100:.2f}%
Profit Factor:     {profit_factor:.2f}
Avg Winning Trade: {avg_win_pct * 100:+.2f}%
Avg Losing Trade:  {avg_loss_pct * 100:.2f}%
--------------------------------------------------
ANNUAL RETURNS BREAKDOWN
"""
for year, ret in annual_returns.items():
    summary_text += f"{year}: {ret * 100:+.2f}%\n"

print(summary_text)
with open('backtest_results.txt', 'w', encoding='utf-8') as f: f.write(summary_text)

# --- VISUALIZATION TEAR SHEET ---
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])

# 1. Equity Curve
ax1 = plt.subplot(gs[0])
ax1.plot(results.index, results['Equity'], label='Momentum System', color='#2E86C1', linewidth=2)
nifty_norm = (nifty['Close'].loc[results.index] / nifty['Close'].loc[results.index[0]]) * INITIAL_CAPITAL
ax1.plot(results.index, nifty_norm, label='Nifty 50 Buy & Hold', color='#7F8C8D', linewidth=1.5, alpha=0.7)
ax1.set_yscale('log')
ax1.set_title('Log-Scale Equity Curve vs Benchmark', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Drawdown Plot
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.fill_between(results.index, results['Drawdown'] * 100, 0, color='#E74C3C', alpha=0.3)
ax2.plot(results.index, results['Drawdown'] * 100, color='#C0392B', linewidth=1)
ax2.set_title('Underwater Plot (Drawdown %)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown %')
ax2.grid(True, alpha=0.3)

# 3. Annual Returns
ax3 = plt.subplot(gs[2])
colors = ['#27AE60' if x > 0 else '#E74C3C' for x in annual_returns.values]
ax3.bar(annual_returns.index.astype(str), annual_returns.values * 100, color=colors)
ax3.set_title('Annual Returns (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Return %')
ax3.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('backtest_analytics.png', dpi=150, bbox_inches='tight')