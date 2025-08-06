# 📈 Stock Price Prediction using LSTM Neural Networks

A comprehensive machine learning project that predicts stock prices using LSTM neural networks with advanced technical indicators and data analysis techniques.

## 🎯 Project Overview

This project implements a stock price prediction system that combines:
- **Real-time data collection** from Yahoo Finance
- **Technical analysis** with 14+ indicators (RSI, MACD, Bollinger Bands, etc.)
- **Machine learning models** for price prediction
- **Comprehensive visualization** and performance analysis

## 📊 Results

**AAPL Stock Analysis (3 years of data):**
- 🎯 **Prediction Accuracy**: 96.3%
- 📉 **Root Mean Square Error**: $7.95
- 📈 **Mean Absolute Error**: $6.10
- 💹 **R² Score**: 0.7264

## 🚀 Features

### Data Processing
- Automated stock data fetching using `yfinance`
- Multi-timeframe analysis (1-10 years of historical data)
- Data cleaning and preprocessing with proper handling of missing values
- Feature engineering with technical indicators

### Technical Indicators
- **Moving Averages**: 7-day, 21-day, 50-day
- **RSI**: Relative Strength Index (14-period)
- **MACD**: Moving Average Convergence Divergence with signal line
- **Bollinger Bands**: Statistical price bands with position calculation
- **Volume Analysis**: Volume moving averages and ratios
- **Volatility Metrics**: Rolling standard deviation of returns

### Machine Learning
- LSTM neural network architecture for sequential data modeling
- RandomForest regression for robust baseline predictions
- Proper train/test splitting with time series considerations
- Feature scaling and normalization
- Model evaluation with multiple metrics

### Visualization
- 4-panel comprehensive analysis charts
- Training vs testing performance comparison
- Residual analysis for model diagnostics
- Performance metrics dashboard
- Publication-ready plots with professional styling

## 🛠️ Installation

### Prerequisites
```bash
python --version  # Requires Python 3.7+
```

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn yfinance scikit-learn
```

### Optional (for LSTM implementation)
```bash
pip install tensorflow  # For advanced LSTM models
```

## 🚀 Usage

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Venksaiabhishek/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm

# Run the main prediction script
python stock_prediction.py
```

### Configuration
You can modify prediction parameters in the script:
```python
# Configuration options
SYMBOL = 'AAPL'           # Stock symbol to analyze
YEARS = 3                 # Years of historical data
LOOKBACK_WINDOW = 60      # Days to look back for predictions  
TEST_SIZE = 0.2           # Fraction of data for testing
```

### Advanced Usage with LSTM Class
```python
from lstm_stock_predictor import StockPredictor

# Initialize predictor
predictor = StockPredictor(lookback_window=60, test_size=0.2)

# Fetch and prepare data
stock_data = predictor.fetch_stock_data(['AAPL', 'GOOGL'], years=3)
X_train, X_test, y_train, y_test, scaler = predictor.prepare_data('AAPL')

# Train model
model = predictor.build_model()
history = predictor.train_model(epochs=50)

# Evaluate and visualize
metrics = predictor.evaluate_model()
predictor.plot_results(metrics, symbol='AAPL')
```

## 📁 Project Structure

```
stock-price-prediction-lstm/
├── stock_prediction.py          # Main prediction script
├── lstm_stock_predictor.py      # Advanced LSTM class implementation
├── stock_prediction_analysis.ipynb  # Jupyter notebook for exploration
├── sample_prediction_chart.png  # Example output visualization
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This documentation
```

## 📊 Model Architecture

The project implements two complementary approaches:

### 1. RandomForest Baseline
- Ensemble learning with 100 decision trees
- Handles non-linear patterns effectively
- Fast training and reliable performance
- Used as the primary model in `stock_prediction.py`

### 2. LSTM Neural Network
```python
Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 14)),
    Dropout(0.2),
    LSTM(32, return_sequences=True),
    Dropout(0.2), 
    LSTM(16, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])
```

## 📈 Sample Output

```bash
🚀 Starting Stock Price Prediction Analysis
==================================================
📈 Fetching AAPL stock data for the last 3 years...
✅ Successfully fetched 751 records
🔧 Adding technical indicators...
✅ Technical indicators added successfully
📊 Dataset shape: (702, 14)
🚀 Training model...
✅ Training completed!

🎯 MODEL PERFORMANCE RESULTS:
========================================
📊 Test RMSE: $7.95
📊 Test MAE: $6.10
📊 Test R²: 0.7264
📊 Test Accuracy: 96.26%
========================================

🔮 SAMPLE PREDICTIONS:
------------------------------
Day  1: Actual=$214.15, Predicted=$216.45, Error=$2.30
Day  2: Actual=$213.76, Predicted=$216.26, Error=$2.50
...
```

## 🔬 Technical Details

### Data Pipeline
1. **Data Collection**: Real-time fetching from Yahoo Finance API
2. **Preprocessing**: Handling missing values, outlier detection
3. **Feature Engineering**: Technical indicator calculation
4. **Scaling**: MinMax normalization for neural networks
5. **Sequence Creation**: Time series windowing for LSTM input

### Evaluation Metrics
- **RMSE**: Root Mean Square Error for prediction accuracy
- **MAE**: Mean Absolute Error for average prediction deviation
- **R²**: Coefficient of determination for model fit quality
- **Accuracy**: Percentage-based accuracy measure
- **Residual Analysis**: Error distribution and patterns

### Visualization Components
1. **Full Timeline**: Complete training and testing predictions
2. **Test Focus**: Detailed view of test period performance
3. **Residuals**: Error analysis and model diagnostics
4. **Metrics**: Performance summary with key statistics

## 🎛️ Customization

### Adding New Stocks
```python
# Analyze multiple stocks
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
for symbol in symbols:
    # Run analysis for each stock
```

### Custom Technical Indicators
```python
def add_custom_indicator(df):
    # Add your custom technical analysis
    df['Custom_MA'] = df['Close'].rolling(window=15).mean()
    return df
```

### Model Tuning
```python
# Adjust model parameters
model = RandomForestRegressor(
    n_estimators=200,     # More trees
    max_depth=15,         # Deeper trees  
    random_state=42
)
```

## ⚠️ Disclaimer

This project is for educational and research purposes only. Stock market predictions are inherently uncertain and involve significant risk. Past performance does not guarantee future results. Always conduct thorough research and consider professional financial advice before making investment decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Yahoo Finance** for providing accessible financial data API
- **scikit-learn** for robust machine learning algorithms
- **TensorFlow** for advanced neural network capabilities
- **pandas** and **numpy** for efficient data processing
- **matplotlib** and **seaborn** for comprehensive data visualization

---

Made with ❤️ for the data science and finance community.
