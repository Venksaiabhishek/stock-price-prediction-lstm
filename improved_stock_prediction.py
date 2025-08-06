#!/usr/bin/env python3
"""
Improved Stock Price Prediction using LSTM Neural Networks
==========================================================

This script provides a comprehensive implementation of stock price prediction
using LSTM (Long Short-Term Memory) neural networks. It includes proper error
handling, data validation, and enhanced visualization capabilities.

Features:
- Multi-stock data fetching and processing
- Advanced data preprocessing with multiple indicators
- Robust LSTM model architecture
- Comprehensive evaluation metrics
- Interactive visualizations
- Model saving and loading capabilities
- Future price prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

class StockPredictor:
    """
    A comprehensive stock price prediction system using LSTM neural networks.
    """
    
    def __init__(self, lookback_window=60, test_size=0.2):
        """
        Initialize the StockPredictor.
        
        Parameters:
        -----------
        lookback_window : int
            Number of previous days to use for prediction
        test_size : float
            Proportion of data to use for testing (0.0 to 1.0)
        """
        self.lookback_window = lookback_window
        self.test_size = test_size
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.stock_data = {}
        self.processed_data = None
        
    def fetch_stock_data(self, symbols, years=3):
        """
        Fetch stock data for multiple symbols.
        
        Parameters:
        -----------
        symbols : list
            List of stock symbols to fetch
        years : int
            Number of years of historical data to fetch
        
        Returns:
        --------
        dict : Dictionary containing stock data for each symbol
        """
        print("Fetching stock data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        stock_data = {}
        company_names = {
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation'
        }
        
        for symbol in symbols:
            try:
                print(f"Fetching data for {symbol}...")
                data = yf.download(symbol, start=start_date, end=end_date)
                if not data.empty:
                    data['Symbol'] = symbol
                    data['Company'] = company_names.get(symbol, symbol)
                    stock_data[symbol] = data
                    print(f"Successfully fetched {len(data)} records for {symbol}")
                else:
                    print(f"Warning: No data found for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                
        self.stock_data = stock_data
        return stock_data
    
    def add_technical_indicators(self, data):
        """
        Add technical indicators to the stock data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Stock data DataFrame
        
        Returns:
        --------
        pd.DataFrame : Enhanced DataFrame with technical indicators
        """
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Range'] = df['High'] - df['Low']
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        return df
    
    def prepare_data(self, symbol='AAPL', target_column='Close'):
        """
        Prepare data for LSTM model training.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to use for training
        target_column : str
            Column to predict
        
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test, scaler)
        """
        if symbol not in self.stock_data:
            raise ValueError(f"Data for symbol {symbol} not found. Please fetch data first.")
        
        print(f"Preparing data for {symbol}...")
        
        # Get the data and add technical indicators
        data = self.add_technical_indicators(self.stock_data[symbol])
        
        # Select features for prediction
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_7', 'MA_21', 'MA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'RSI', 'BB_Position',
            'Volume_Ratio', 'High_Low_Pct', 'Volatility'
        ]
        
        # Remove columns with too many NaN values and ensure we have enough data
        available_columns = []
        for col in feature_columns:
            if col in data.columns:
                # Check if column has enough valid data
                valid_ratio = data[col].notna().sum() / len(data)
                if valid_ratio > 0.8:  # At least 80% valid data
                    available_columns.append(col)
        
        self.feature_columns = available_columns
        print(f"Using features: {self.feature_columns}")
        
        # Create dataset with selected features
        dataset = data[self.feature_columns].copy()
        
        # Forward fill missing values and then drop remaining NaN
        dataset = dataset.fillna(method='ffill').dropna()
        
        if len(dataset) < self.lookback_window + 50:  # Need minimum data
            raise ValueError(f"Not enough data after cleaning. Got {len(dataset)} rows, need at least {self.lookback_window + 50}")
        
        print(f"Dataset shape after cleaning: {dataset.shape}")
        
        # Scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(dataset.values)
        
        # Create sequences for LSTM
        X, y = [], []
        target_idx = self.feature_columns.index(target_column)
        
        for i in range(self.lookback_window, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_window:i])
            y.append(scaled_data[i, target_idx])
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        self.processed_data = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'dataset': dataset, 'scaled_data': scaled_data,
            'split_idx': split_idx
        }
        
        return X_train, X_test, y_train, y_test, self.scaler
    
    def build_model(self, lstm_units=[50, 50, 50], dropout_rate=0.2, learning_rate=0.001):
        """
        Build and compile the LSTM model.
        
        Parameters:
        -----------
        lstm_units : list
            List of LSTM layer units
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for optimizer
        
        Returns:
        --------
        tensorflow.keras.Model : Compiled LSTM model
        """
        print("Building LSTM model...")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=lstm_units[0], 
                      return_sequences=True, 
                      input_shape=(self.lookback_window, len(self.feature_columns))))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i in range(1, len(lstm_units)):
            return_sequences = i < len(lstm_units) - 1
            model.add(LSTM(units=lstm_units[i], return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
            if return_sequences:
                model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(25))
        model.add(Dropout(dropout_rate/2))
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, 
                     loss='mean_squared_error',
                     metrics=['mae'])
        
        print(f"Model built with {model.count_params():,} parameters")
        self.model = model
        return model
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.1, patience=10):
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Validation data split ratio
        patience : int
            Early stopping patience
        
        Returns:
        --------
        tensorflow.keras.callbacks.History : Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if self.processed_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print("Training model...")
        
        X_train, y_train = self.processed_data['X_train'], self.processed_data['y_train']
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def evaluate_model(self):
        """
        Evaluate the trained model and return comprehensive metrics.
        
        Returns:
        --------
        dict : Dictionary containing evaluation metrics and predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("Evaluating model...")
        
        X_train, X_test = self.processed_data['X_train'], self.processed_data['X_test']
        y_train, y_test = self.processed_data['y_train'], self.processed_data['y_test']
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Inverse transform predictions (only for the target column)
        # Create dummy arrays with the right shape for inverse transform
        dummy_train = np.zeros((train_pred.shape[0], len(self.feature_columns)))
        dummy_test = np.zeros((test_pred.shape[0], len(self.feature_columns)))
        
        # Place predictions in the target column position
        target_idx = self.feature_columns.index('Close')
        dummy_train[:, target_idx] = train_pred.flatten()
        dummy_test[:, target_idx] = test_pred.flatten()
        
        # Inverse transform
        train_pred_scaled = self.scaler.inverse_transform(dummy_train)[:, target_idx]
        test_pred_scaled = self.scaler.inverse_transform(dummy_test)[:, target_idx]
        
        # Also inverse transform actual values
        dummy_train_actual = np.zeros((len(y_train), len(self.feature_columns)))
        dummy_test_actual = np.zeros((len(y_test), len(self.feature_columns)))
        
        dummy_train_actual[:, target_idx] = y_train
        dummy_test_actual[:, target_idx] = y_test
        
        y_train_scaled = self.scaler.inverse_transform(dummy_train_actual)[:, target_idx]
        y_test_scaled = self.scaler.inverse_transform(dummy_test_actual)[:, target_idx]
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train_scaled, train_pred_scaled)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_scaled, test_pred_scaled)),
            'train_mae': mean_absolute_error(y_train_scaled, train_pred_scaled),
            'test_mae': mean_absolute_error(y_test_scaled, test_pred_scaled),
            'train_r2': r2_score(y_train_scaled, train_pred_scaled),
            'test_r2': r2_score(y_test_scaled, test_pred_scaled),
            'predictions': {
                'train_actual': y_train_scaled,
                'train_pred': train_pred_scaled,
                'test_actual': y_test_scaled,
                'test_pred': test_pred_scaled
            }
        }
        
        # Calculate accuracy as percentage
        test_accuracy = 100 - (metrics['test_rmse'] / np.mean(y_test_scaled) * 100)
        metrics['test_accuracy'] = max(0, test_accuracy)  # Ensure non-negative
        
        print(f"Model Evaluation Results:")
        print(f"Train RMSE: ${metrics['train_rmse']:.2f}")
        print(f"Test RMSE: ${metrics['test_rmse']:.2f}")
        print(f"Test MAE: ${metrics['test_mae']:.2f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.2f}%")
        
        return metrics
    
    def plot_results(self, metrics, symbol='AAPL'):
        """
        Plot comprehensive results including predictions and metrics.
        
        Parameters:
        -----------
        metrics : dict
            Metrics dictionary from evaluate_model()
        symbol : str
            Stock symbol for titles
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training and Test Predictions
        ax1 = axes[0, 0]
        train_actual = metrics['predictions']['train_actual']
        train_pred = metrics['predictions']['train_pred']
        test_actual = metrics['predictions']['test_actual']
        test_pred = metrics['predictions']['test_pred']
        
        ax1.plot(train_actual, label='Training Actual', alpha=0.7)
        ax1.plot(train_pred, label='Training Predicted', alpha=0.7)
        ax1.plot(range(len(train_actual), len(train_actual) + len(test_actual)), 
                test_actual, label='Test Actual', alpha=0.7)
        ax1.plot(range(len(train_actual), len(train_actual) + len(test_pred)), 
                test_pred, label='Test Predicted', alpha=0.7)
        ax1.set_title(f'{symbol} Stock Price Prediction')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Test Set Zoom
        ax2 = axes[0, 1]
        ax2.plot(test_actual, label='Actual', linewidth=2)
        ax2.plot(test_pred, label='Predicted', linewidth=2, alpha=0.8)
        ax2.set_title('Test Set Predictions (Zoomed)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        ax3 = axes[1, 0]
        residuals = test_actual - test_pred
        ax3.scatter(test_pred, residuals, alpha=0.6)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_title('Residual Plot')
        ax3.set_xlabel('Predicted Price ($)')
        ax3.set_ylabel('Residuals ($)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Metrics Bar Chart
        ax4 = axes[1, 1]
        metric_names = ['RMSE', 'MAE', 'R²', 'Accuracy (%)']
        metric_values = [
            metrics['test_rmse'],
            metrics['test_mae'],
            metrics['test_r2'],
            metrics['test_accuracy']
        ]
        
        colors = ['red', 'orange', 'green', 'blue']
        bars = ax4.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax4.set_title('Model Performance Metrics')
        ax4.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, days=30):
        """
        Predict future stock prices.
        
        Parameters:
        -----------
        days : int
            Number of days to predict into the future
        
        Returns:
        --------
        np.array : Array of predicted prices
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print(f"Predicting next {days} days...")
        
        # Get the last sequence from the data
        scaled_data = self.processed_data['scaled_data']
        last_sequence = scaled_data[-self.lookback_window:].copy()
        
        predictions = []
        target_idx = self.feature_columns.index('Close')
        
        for _ in range(days):
            # Reshape for prediction
            current_sequence = last_sequence.reshape((1, self.lookback_window, len(self.feature_columns)))
            
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence - simple approach: repeat the last values for other features
            new_row = last_sequence[-1].copy()
            new_row[target_idx] = next_pred[0, 0]
            
            # Shift sequence and add new prediction
            last_sequence = np.vstack([last_sequence[1:], new_row])
        
        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), len(self.feature_columns)))
        dummy_array[:, target_idx] = predictions
        
        future_prices = self.scaler.inverse_transform(dummy_array)[:, target_idx]
        
        return future_prices
    
    def save_model(self, filepath='stock_predictor_model'):
        """
        Save the trained model and scaler.
        
        Parameters:
        -----------
        filepath : str
            Base filepath for saving (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Save model
        self.model.save(f"{filepath}.h5")
        
        # Save scaler and other attributes
        model_data = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'lookback_window': self.lookback_window,
            'test_size': self.test_size
        }
        joblib.dump(model_data, f"{filepath}_data.pkl")
        
        print(f"Model saved successfully to {filepath}.h5 and {filepath}_data.pkl")
    
    def load_model(self, filepath='stock_predictor_model'):
        """
        Load a saved model and associated data.
        
        Parameters:
        -----------
        filepath : str
            Base filepath for loading (without extension)
        """
        try:
            # Load model
            self.model = tf.keras.models.load_model(f"{filepath}.h5")
            
            # Load other data
            model_data = joblib.load(f"{filepath}_data.pkl")
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.lookback_window = model_data['lookback_window']
            self.test_size = model_data['test_size']
            
            print(f"Model loaded successfully from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

def main():
    """
    Main function demonstrating the stock prediction workflow.
    """
    print("=== Stock Price Prediction with LSTM ===\n")
    
    # Initialize predictor
    predictor = StockPredictor(lookback_window=60, test_size=0.2)
    
    # List of stocks to analyze
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    # Fetch data
    stock_data = predictor.fetch_stock_data(stocks, years=3)
    
    if not stock_data:
        print("No stock data fetched. Exiting...")
        return
    
    # Select primary stock for prediction
    primary_stock = 'AAPL'
    print(f"\nUsing {primary_stock} for prediction model...")
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = predictor.prepare_data(
            symbol=primary_stock, target_column='Close'
        )
        
        # Build model
        model = predictor.build_model(
            lstm_units=[64, 32, 16],
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Train model
        history = predictor.train_model(
            epochs=50,  # Reduced for demo
            batch_size=32,
            patience=10
        )
        
        # Evaluate model
        metrics = predictor.evaluate_model()
        
        # Plot results
        predictor.plot_results(metrics, symbol=primary_stock)
        
        # Predict future prices
        future_prices = predictor.predict_future(days=30)
        
        print(f"\nPredicted prices for next 30 days:")
        for i, price in enumerate(future_prices[:10], 1):  # Show first 10 days
            print(f"Day {i}: ${price:.2f}")
        
        # Save model
        predictor.save_model('improved_stock_model')
        
        print("\n=== Analysis Complete ===")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
