# Contributing to StockTradingAI

Thank you for your interest in contributing to StockTradingAI! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and professional in all interactions. We're committed to providing a welcoming and inclusive environment.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
1. A clear, descriptive title
2. A detailed description of the problem
3. Steps to reproduce the issue
4. Expected vs. actual behavior
5. Your environment (Python version, OS, etc.)

**Example:**
```
Title: SMA crossover strategy not buying on crossover signal

Description:
The SMA crossover strategy is not generating buy signals when the short SMA crosses above the long SMA.

Steps to reproduce:
1. Run: python -m src.main fetch --ticker AAPL
2. Run: python -m src.main backtest --strategy rule
3. Check trades

Expected: Multiple buy signals
Actual: No trades generated
```

### Suggesting Features

Feature suggestions are welcome! Please open an issue with:
1. A clear description of the feature
2. Why it would be useful
3. Possible implementation approach

**Example:**
```
Title: Add support for crypto trading

Description:
Would be great to support cryptocurrency trading (BTC, ETH) in addition to stocks.

Use case: Can backtest crypto strategies alongside stock strategies
Implementation: Update DataFetcher to support crypto tickers via Polygon crypto API
```

### Code Contributions

#### Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/StockTradingAI.git
   cd StockTradingAI
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Set up development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

#### Development Workflow

1. Make your changes in your feature branch
2. Write or update tests as needed
3. Run tests to ensure everything works:
   ```bash
   pytest tests/ -v
   ```
4. Check code style:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```
5. Commit with clear, descriptive messages:
   ```bash
   git commit -m "Add feature X: description of changes"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a Pull Request on GitHub with:
   - Clear title and description
   - Reference to any related issues
   - Description of changes and testing

#### Code Style

- **Format**: Use Black for code formatting
  ```bash
  black src/ tests/
  ```
- **Linting**: Follow PEP 8 standards
  ```bash
  flake8 src/ tests/
  ```
- **Docstrings**: Use Google-style docstrings
  ```python
  def calculate_sma(data: pd.Series, window: int) -> pd.Series:
      """Calculate simple moving average.
      
      Args:
          data (pd.Series): Price data.
          window (int): SMA window size.
          
      Returns:
          pd.Series: Calculated SMA values.
      """
  ```
- **Type hints**: Use type hints for function signatures
  ```python
  def fetch_data(ticker: str, days: int = 365) -> pd.DataFrame:
  ```

#### Testing Requirements

- Write tests for new features
- Maintain or improve code coverage
- All tests must pass before PR can be merged
- Use pytest for testing

**Example test:**
```python
def test_sma_calculation():
    """Test that SMA is calculated correctly."""
    data = pd.Series([1, 2, 3, 4, 5])
    sma = data.rolling(3).mean()
    
    assert sma.iloc[2] == 2.0  # mean(1,2,3)
    assert sma.iloc[3] == 3.0  # mean(2,3,4)
```

### Areas for Contribution

Here are some areas where contributions are particularly welcome:

#### High Priority
- [ ] Add support for more indicators (RSI, MACD, Bollinger Bands)
- [ ] Implement additional ML models (Random Forest, LSTM)
- [ ] Add portfolio optimization features
- [ ] Improve documentation with more examples
- [ ] Add support for Options trading

#### Medium Priority
- [ ] Performance optimization for large datasets
- [ ] Support for multiple asset classes (crypto, forex)
- [ ] Enhanced visualization features
- [ ] Configuration file enhancements
- [ ] Database integration for data persistence

#### Community Help
- [ ] Answer questions in issues
- [ ] Improve documentation
- [ ] Fix typos and spelling errors
- [ ] Share example backtests
- [ ] Write tutorials

## Pull Request Process

1. Update documentation to reflect changes
2. Add tests for new functionality
3. Update CHANGELOG if applicable
4. Ensure all tests pass:
   ```bash
   pytest tests/ -v --cov=src/
   ```
5. Request review from maintainers
6. Address feedback and make requested changes
7. Once approved, your PR will be merged!

## Project Structure Guidelines

When adding new features:

```
src/
├── strategies/          # Add new strategy here
│   └── your_strategy.py
├── indicators/          # Add new indicators if applicable
│   └── your_indicator.py
└── utils/              # Add utility functions here
    └── your_utils.py

tests/
├── test_strategies.py   # Add strategy tests
├── test_indicators.py   # Add indicator tests
└── test_utils.py       # Add utility tests
```

## Documentation

- **Code comments**: Explain WHY, not WHAT. The code shows what it does.
- **Docstrings**: All public functions and classes should have docstrings
- **README**: Update if you change user-facing functionality
- **Examples**: Add examples for new features in docstrings

## Commit Message Convention

Please use clear, descriptive commit messages:

```
feat: add support for RSI indicator
^--^  ^-----^
|     |
|     +---> Summary in imperative mood
|
+---> Type: feat, fix, docs, style, refactor, test, chore
```

Types:
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc)
- **refactor**: Code refactoring without behavior change
- **test**: Adding or updating tests
- **chore**: Dependency updates, etc

Examples:
```
feat: implement LSTM strategy for price prediction
fix: correct SMA calculation in rule-based strategy
docs: add example for ML strategy usage
test: add 5 new tests for data fetcher
```

## Review Process

- Maintainers will review your PR within a few days
- We may request changes or clarifications
- Once approved, your contribution will be merged
- You'll be added to the contributors list

## Questions?

Feel free to:
- Open a GitHub issue for questions
- Start a discussion for major changes
- Email the maintainers

## License

By contributing to StockTradingAI, you agree that your contributions will be licensed under its MIT License.

## Contributor Recognition

All contributors will be recognized in:
1. The README.md contributors section
2. GitHub's contributors page
3. Release notes

Thank you for contributing to StockTradingAI! 🎉
