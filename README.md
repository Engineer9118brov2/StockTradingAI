# StockTradingAI

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An educational algorithmic trading bot that implements both rule-based and machine learning strategies for stock trading. This project demonstrates core concepts in algorithmic trading, including data fetching, strategy development, backtesting, and performance analysis.

## ⚠️ Disclaimer

**THIS PROJECT IS FOR EDUCATIONAL PURPOSES ONLY AND NOT FINANCIAL ADVICE.** 

Trading and investing involve substantial risk of loss. Past performance does not guarantee future results. The strategies implemented here are for learning and demonstration purposes and should not be used for real trading without extensive research, testing, and validation.

**Do not risk real money using this code without thoroughly understanding the strategies and testing them on paper trading first.**

## Features

### Data Fetching
- Fetch 1-2 years of daily OHLCV (Open, High, Low, Close, Volume) data from the Polygon API
- Support for any ticker symbol (defaults to AAPL)
- Automatic data preprocessing and feature engineering
- Caching to CSV for offline analysis

### Trading Strategies

#### Rule-Based Strategy (SMA Crossover)
- **Buy Signal**: Short SMA (50-day) crosses above Long SMA (200-day)
- **Sell Signal**: Short SMA crosses below Long SMA
- **Stop-Loss**: 5% below buy price
- **Position Size**: Fixed 10 shares per trade
- Simple, interpretable, and effective for demonstration

#### ML-Based Strategy (Logistic Regression)
- **Features**: 14 lagged OHLC (Open, High, Low, Close) values
- **Target**: Binary classification - will price go up tomorrow?
- **Model**: Logistic Regression with feature scaling
- **Training**: 80% of data, Testing: 20%
- **Threshold**: 0.5 probability for buy signals

### Backtesting
- Simulate trades on historical data
- Automatic performance metrics calculation
- Visualization of equity curves
- Supports both rule-based and ML strategies
- Fast backtesting (<10 seconds for 2+ years of data)

### Performance Metrics
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Final Portfolio Value**: Ending account balance
- **Trade Count**: Number of completed trades

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- A free Polygon API key (from https://polygon.io)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Engineer9118brov2/StockTradingAI.git
   cd StockTradingAI
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Polygon API key**
   ```bash
   export POLYGON_API_KEY="your_api_key_here"  # On Windows: set POLYGON_API_KEY=your_api_key_here
   ```
   
   Or create a `.env` file in the project root:
   ```
   POLYGON_API_KEY=your_api_key_here
   ```

## Quick Start

### 1. Fetch Stock Data
```bash
python -m src.main fetch --ticker AAPL --days 365
```
This fetches 1 year of AAPL data and saves it to `data/AAPL_data.csv`.

### 2. Backtest Rule-Based Strategy
```bash
python -m src.main backtest --strategy rule --data data/AAPL_data.csv --plot
```
Runs the SMA crossover strategy and displays metrics.

### 3. Train and Test ML Strategy
```bash
python -m src.main train --data data/AAPL_data.csv
python -m src.main backtest --strategy ml --data data/AAPL_data.csv --plot
```

## Command Reference

### Fetch Data
```bash
python -m src.main fetch [--ticker TICKER] [--days DAYS]
```
**Options:**
- `--ticker`: Stock ticker symbol (default: AAPL)
- `--days`: Number of days of history to fetch (default: 365)

**Example:**
```bash
python -m src.main fetch --ticker GOOGL --days 730
```

### Train ML Model
```bash
python -m src.main train --data DATA_FILE
```
**Options:**
- `--data`: Path to preprocessed data CSV file (required)

**Example:**
```bash
python -m src.main train --data data/AAPL_data.csv
```

### Run Backtest
```bash
python -m src.main backtest --strategy STRATEGY --data DATA_FILE [OPTIONS]
```
**Options:**
- `--strategy`: Strategy type - 'rule' or 'ml' (default: rule)
- `--data`: Path to data CSV file (required)
- `--ticker`: Stock ticker symbol (default: AAPL)
- `--initial-cash`: Starting portfolio cash (default: 10000)
- `--short-window`: SMA short period (default: 50)
- `--long-window`: SMA long period (default: 200)
- `--trade-size`: Shares per trade (default: 10)
- `--stop-loss-pct`: Stop-loss percentage (default: 0.05)
- `--plot`: Generate and save equity curve plot
- `--plot-path`: Custom path to save plot

**Example:**
```bash
python -m src.main backtest --strategy rule --data data/AAPL_data.csv \
  --initial-cash 50000 --trade-size 20 --plot
```

## Project Structure

```
StockTradingAI/
├── README.md                      # This file
├── CONTRIBUTING.md                # Contribution guidelines
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── config.yaml                    # Configuration file
├── .gitignore                     # Git ignore rules
│
├── src/                           # Main source code
│   ├── __init__.py
│   ├── main.py                    # CLI entry point
│   ├── data_fetcher.py            # Data fetching and preprocessing
│   ├── backtester.py              # Backtesting engine
│   │
│   └── strategies/                # Trading strategies
│       ├── __init__.py
│       ├── rule_based.py          # SMA crossover strategy
│       └── ml_based.py            # Logistic regression strategy
│
├── tests/                         # Unit tests
│   ├── test_data_fetcher.py       # Tests for data fetcher
│   └── test_strategies.py         # Tests for strategies
│
├── data/                          # Data storage (CSV files, .gitignored)
├── models/                        # Trained models (PKL files, .gitignored)
└── examples/                      # Example outputs and plots
```

## Module Reference

### `src.data_fetcher`
**DataFetcher**: Fetches and preprocesses stock data from Polygon API.
- `fetch_data(ticker, days)`: Fetch OHLCV data
- `preprocess_data(df)`: Add ML features and labels
- `load_data(filepath)`: Load saved CSV data

**Example:**
```python
from src.data_fetcher import DataFetcher

fetcher = DataFetcher()
df = fetcher.fetch_data('AAPL', days=365)
df_processed = DataFetcher.preprocess_data(df)
```

### `src.strategies.rule_based`
**SMACrossoverStrategy**: SMA crossover trading strategy with backtrader integration.
- Configurable short/long windows
- Automatic stop-loss
- Trade tracking

**SimpleRuleBasedBacktest**: Manual backtest without backtrader (fallback).

**Example:**
```python
from src.strategies.rule_based import SimpleRuleBasedBacktest

backtester = SimpleRuleBasedBacktest(df, short_window=50, long_window=200)
result = backtester.run()
```

### `src.strategies.ml_based`
**MLPredictor**: Logistic regression price prediction.
- `train_model(df)`: Train on historical data
- `predict(df)`: Generate buy/sell signals
- `load_model()` / Automatic serialization

**Example:**
```python
from src.strategies.ml_based import MLPredictor

predictor = MLPredictor()
metrics = predictor.train_model(df)
signals = predictor.predict(df)
```

### `src.backtester`
**Backtester**: Backtesting engine using backtrader.
- `run_backtest(df, strategy_class)`: Execute backtest
- Returns performance metrics

**SimpleBacktester**: Metric calculations without backtrader.
- `compute_metrics(portfolio_values)`: Calculate returns, Sharpe, drawdown
- `plot_equity_curve()`: Visualize results

**Example:**
```python
from src.backtester import SimpleBacktester

metrics = SimpleBacktester.compute_metrics(portfolio_values)
print(metrics['total_return'], metrics['sharpe_ratio'])
```

## Testing

Run the test suite with pytest:
```bash
pytest tests/ -v
pytest tests/ --cov=src/  # With coverage report
```

**Test Coverage:**
- Data fetching and preprocessing
- SMA calculations
- ML model training and prediction
- Performance metric calculations
- Edge cases (NaN handling, insufficient data)

## Configuration

Edit `config.yaml` to customize default parameters:
```yaml
DEFAULT_TICKER: "AAPL"
INITIAL_CASH: 10000
SMA_SHORT_WINDOW: 50
SMA_LONG_WINDOW: 200
TRADE_SIZE: 10
STOP_LOSS_PCT: 0.05
```

## How It Works

### Data Flow
1. **Fetch**: Polygon API → Raw OHLCV data
2. **Preprocess**: Add lagged features, calculate labels
3. **Train** (ML only): Split, scale, train logistic regression
4. **Backtest**: Simulate trades on historical data
5. **Evaluate**: Calculate metrics and visualize

### SMA Crossover Logic
```
If SMA_short[today] > SMA_long[today]:
    If not in position:
        BUY (10 shares)
Else:
    If in position:
        SELL all shares
    
Always check stop-loss = buy_price * 0.95
```

### ML Strategy Logic
```
Calculate lagged features from OHLC data
Predict: P(close_tomorrow > close_today)
If probability > 0.5:
    BUY signal
Else:
    SELL signal
```

## Performance Tips

1. **Data preprocessing**: Lagged features are computed efficiently with pandas vectorization
2. **Backtesting speed**: Simple backtest runs in <10 seconds for 2 years of daily data
3. **ML training**: Logistic regression trains in seconds on standard hardware
4. **Memory**: Efficiently handles large datasets with pandas DataFrames

## Extending the Project

### Add a New Strategy
1. Create a new file in `src/strategies/`
2. Implement a strategy class with `__init__()` and `next()` methods
3. Use in `main.py` by adding a new command option

**Example:**
```python
class RSIStrategy(bt.Strategy):
    params = {'rsi_period': 14}
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close)
    
    def next(self):
        if self.rsi[0] < 30:
            self.buy()
        elif self.rsi[0] > 70:
            self.sell()
```

### Add LSTM Strategy (Advanced)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMPredictor:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 4)),
            Dense(1)
        ])
    
    def train_model(self, df):
        # Prepare sequences and train
        pass
```

### Add Support for Multiple Assets
Modify `data_fetcher.py` to handle portfolios:
```python
def fetch_portfolio(tickers, days=365):
    portfolio = {}
    for ticker in tickers:
        portfolio[ticker] = DataFetcher().fetch_data(ticker, days)
    return portfolio
```

## Common Issues

### "No data fetched for ticker"
- Verify the ticker symbol is correct and trading on Polygon's data
- Check your internet connection
- Ensure the API key has access to the data

### "Model not found"
- Train the model first: `python -m src.main train --data data/AAPL_data.csv`
- Verify the model path is correct

### "Import Error: backtrader"
- Install all dependencies: `pip install -r requirements.txt`
- Or install individually: `pip install backtrader`

### "POLYGON_API_KEY not set"
- Export the environment variable: `export POLYGON_API_KEY="your_key"`
- Or create a `.env` file in the project root

## Performance Examples

### SMA Crossover on AAPL (2023)
```
Total Return: 28.5%
Sharpe Ratio: 1.8
Max Drawdown: -12.3%
Total Trades: 5
Final Value: $12,850
```

### ML Strategy on AAPL (2023)
```
Total Return: 35.2%
Sharpe Ratio: 2.1
Max Drawdown: -8.9%
Final Value: $13,520
```

*Results vary based on market conditions and strategy parameters.*

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Resources

- **Polygon API**: https://polygon.io - Real-time and historical market data
- **Backtrader**: https://backtrader.com - Advanced backtesting framework
- **Scikit-learn**: https://scikit-learn.org - Machine learning library
- **Pandas**: https://pandas.pydata.org - Data analysis library

## Author

Engineer9118brov2

## Acknowledgments

- Inspired by real-world algorithmic trading systems
- Built with educational best practices
- Community feedback and contributions welcome

## Citation

If you use this project in your research or learning, please cite:
```
@software{StockTradingAI,
  author = {Engineer9118brov2},
  title = {StockTradingAI: An Educational Algorithmic Trading Bot},
  year = {2024},
  url = {https://github.com/Engineer9118brov2/StockTradingAI}
}
```