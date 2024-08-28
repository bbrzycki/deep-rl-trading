# deep-rl-trading
Basic portfolio optimization and trading using Deep Reinforcement Learning

This was my final project for CS 285: Deep Reinforcement Learning at UC Berkeley, taken in Fall 2019. My final report is included as a pdf.

### Example usage for scripts

#### Split historical data into train / test files:

```
python preprocess_daily_ohlc.py ../../../data/daily-us-stocks-etfs/Stocks/aapl.us.txt ../historical_data -s 0.9
```
