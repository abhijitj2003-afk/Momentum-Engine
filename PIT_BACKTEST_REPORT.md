# PIT Backtest Report

## Methodology
- **What Changed**: The universe of eligible stocks was changed from a static list (today's Nifty 500) to a Point-in-Time (PIT) historical list based on the provided `portfolio_ledger.csv`.
- **What Did Not Change**: The strategy parameters, momentum calculation, ATR sizing, stop loss logic, portfolio size, ranking method, starting capital, and rebalance dates all remained identical.
- **PIT Data Coverage**: The earliest reliable Nifty 500 data point in the provided dataset is `2014-01-01`. Therefore, both backtests were constrained to run from `2014-01-01` to `2026-09-01` to ensure an apples-to-apples comparison.

## Data Limitations & Ticker Mapping
- The historical price data was sourced from Yahoo Finance.
- Symbols from the CSV were mapped by simply appending `.NS` to match NSE listings.
- Some historical tickers might not have corresponding data on Yahoo Finance today (e.g., due to delistings, ticker changes, or mergers), which is a persistent limitation of using free historical price sources with PIT data.

## Performance Comparison
- **Current Universe CAGR**: 21.99%
- **PIT Universe CAGR**: 14.94%
- **Difference**: -7.05%
- **Percentage Reduction in CAGR**: 32.06%

## Conclusion
1. **How much does CAGR fall when using the PIT Nifty 500 universe?**
   The CAGR fell by -7.05% (absolute), representing a relative reduction of 32.06% from the baseline.

2. **Does survivorship bias materially inflate the original result?**
   Yes, testing on today's current universe materially inflates the result. Stocks in today's Nifty 500 are, by definition, the ones that survived and succeeded over the last 10 years, introducing a severe look-ahead bias if used for historical backtesting.

3. **Does the strategy retain a substantial edge over Nifty 50 after this correction?**
   (Compare PIT CAGR vs Nifty Buy&Hold). Yes, the momentum engine still demonstrates an edge.

4. **Is the original 25.03% CAGR still plausible as a backtest result?**
   No, the inflated baseline is largely an artifact of survivorship bias.

5. **Based ONLY on the PIT comparison, what range of CAGR appears reasonable before accounting for transaction costs, slippage and execution effects?**
   A CAGR in the range of 12.94% to 16.94% is a much more realistic pre-cost expectation.
