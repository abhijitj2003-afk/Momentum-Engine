import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
START_DATE = '2007-01-01'
END_DATE = '2026-09-01'
INITIAL_CAPITAL = 1000000.0
PORTFOLIO_SIZE = 15
ATR_MULTIPLIER = 3.0

# Using a smaller universe for speed. Expand to Nifty 500 when ready.
TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
           "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
           "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "M&M.NS", "SUNPHARMA.NS",
           "TITAN.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS"]

def calc_momentum_score(series, window):
    if len(series) < window: return 0.0
    sub_series = series.iloc[-window:].values
    # Filter out zero or negative values before log
    if np.any(sub_series <= 0): return 0.0
    log_vals = np.log(sub_series)
    x = np.arange(len(log_vals))
    slope, _, r_val, _, _ = linregress(x, log_vals)
    return ((1 + slope) ** 252 - 1) * (r_val ** 2)

print("📥 Downloading Historical Market Data (This may take a minute)...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by='ticker', progress=False)
nifty = yf.download('^NSEI', start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)

# Pre-calculate Nifty 200-SMA Macro Filter
nifty['200_SMA'] = nifty['Close'].rolling(window=200).mean()
nifty['Regime_Bullish'] = nifty['Close'] > nifty['200_SMA']

# Align all trading days
trading_days = nifty.dropna().index
print(f"✅ Data loaded. Simulating {len(trading_days)} trading days...")

# State Variables
cash = INITIAL_CAPITAL
portfolio = {} # Format: {ticker: {'Shares': x, 'Buy_Price': y, 'Peak': z, 'Stop': s}}
equity_curve = []

# --- THE TIME MACHINE LOOP ---
for i, current_date in enumerate(trading_days):
    if i < 200: 
        equity_curve.append(cash)
        continue # Wait for moving averages to warm up

    regime_bullish = nifty.loc[current_date, 'Regime_Bullish']
    
    # Calculate current portfolio value based on today's close
    daily_equity = cash
    for ticker, pos in portfolio.items():
        if current_date in data[ticker].index and not pd.isna(data[ticker].loc[current_date, 'Close']):
            daily_equity += pos['Shares'] * data[ticker].loc[current_date, 'Close']
        else:
            daily_equity += pos['Shares'] * pos['Buy_Price'] # Fallback if missing data
            
    # 1. DAILY AUDIT: Check Trailing Stops
    sold_today = []
    for ticker, pos in list(portfolio.items()):
        if current_date not in data[ticker].index: continue
        
        today_data = data[ticker].loc[current_date]
        if pd.isna(today_data['Close']): continue
            
        today_close = today_data['Close']
        today_high = today_data['High']
        
        # Calculate ATR-20 looking backward from today
        past_data = data[ticker].loc[:current_date]
        if len(past_data) < 21: continue
        
        tr1 = past_data['High'] - past_data['Low']
        tr2 = abs(past_data['High'] - past_data['Close'].shift(1))
        tr3 = abs(past_data['Low'] - past_data['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_20 = tr.rolling(20).mean().iloc[-1]
        
        # Ratchet Peak & Stop
        new_peak = max(pos['Peak'], today_high)
        calc_stop = new_peak - (ATR_MULTIPLIER * atr_20)
        trailing_stop = max(pos['Stop'], calc_stop)
        
        portfolio[ticker]['Peak'] = new_peak
        portfolio[ticker]['Stop'] = trailing_stop
        
        # Execute Stop Loss
        if today_close < trailing_stop:
            cash += pos['Shares'] * today_close
            sold_today.append(ticker)
            
    for ticker in sold_today:
        del portfolio[ticker]
        
    # 2. WEEKLY ALLOCATION: Every Friday, find new stocks if regime is bullish
    is_friday = current_date.weekday() == 4
    open_slots = PORTFOLIO_SIZE - len(portfolio)
    
    if is_friday and regime_bullish and open_slots > 0:
        candidates = []
        for ticker in TICKERS:
            if ticker in portfolio: continue
            
            past_data = data[ticker].loc[:current_date]
            if len(past_data) < 180: continue
            
            close_series = past_data['Close']
            last_c = close_series.iloc[-1]
            
            # 100-SMA Filter
            sma_100 = close_series.rolling(100).mean().iloc[-1]
            if last_c <= sma_100: continue
            
            # Composite Score
            m20 = calc_momentum_score(close_series, 20)
            m90 = calc_momentum_score(close_series, 90)
            m180 = calc_momentum_score(close_series, 180)
            comp_score = (0.30 * m20) + (0.40 * m90) + (0.30 * m180)
            
            # ATR for sizing
            tr1 = past_data['High'] - past_data['Low']
            tr2 = abs(past_data['High'] - past_data['Close'].shift(1))
            tr3 = abs(past_data['Low'] - past_data['Close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_20 = tr.rolling(20).mean().iloc[-1]
            
            if comp_score > 0 and atr_20 > 0:
                candidates.append({
                    'Ticker': ticker,
                    'Score': comp_score,
                    'Price': last_c,
                    'ATR': atr_20
                })
                
        # Sort and Buy
        if candidates:
            candidates_df = pd.DataFrame(candidates).sort_values(by='Score', ascending=False)
            top_picks = candidates_df.head(open_slots)
            
            risk_budget = daily_equity * (0.30 / 100) # 0.30% risk per trade
            
            for _, pick in top_picks.iterrows():
                shares_to_buy = np.floor(risk_budget / pick['ATR'])
                cost = shares_to_buy * pick['Price']
                
                if cash >= cost and shares_to_buy > 0:
                    cash -= cost
                    portfolio[pick['Ticker']] = {
                        'Shares': shares_to_buy,
                        'Buy_Price': pick['Price'],
                        'Peak': pick['Price'],
                        'Stop': pick['Price'] - (ATR_MULTIPLIER * pick['ATR'])
                    }

    equity_curve.append(daily_equity)

# --- ANALYTICS & VISUALIZATION ---
results = pd.DataFrame({'Date': trading_days[200:], 'Equity': equity_curve[200:]}).set_index('Date')
results['Peak'] = results['Equity'].cummax()
results['Drawdown'] = (results['Equity'] - results['Peak']) / results['Peak']

final_equity = results['Equity'].iloc[-1]
cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / (len(results) / 252))) - 1
max_dd = results['Drawdown'].min()

summary_text = (
    "-" * 40 + "\n" +
    f"🏁 BACKTEST COMPLETE ({START_DATE} to {END_DATE})\n" +
    f"Initial Capital: ₹{INITIAL_CAPITAL:,.2f}\n" +
    f"Final Equity:    ₹{final_equity:,.2f}\n" +
    f"CAGR:            {cagr * 100:.2f}%\n" +
    f"Max Drawdown:    {max_dd * 100:.2f}%\n" +
    "-" * 40
)
print(summary_text)

with open('backtest_results.txt', 'w') as f:
    f.write(summary_text + "\n")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['Equity'], label='System Equity', color='blue')
plt.title('High-Yield Momentum Engine (Backtest Curve)')
plt.yscale('log') # Log scale shows true compound growth
plt.ylabel('Portfolio Equity (Log Scale)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("backtest_curve.png")