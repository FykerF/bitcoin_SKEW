"""
Service layer for model operations
"""
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import yfinance as yf
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from generalized_fall_predictor import BitcoinFallPredictor, TargetConfig
from uptrend_detector import detect_uptrend
from app.core.config import settings
from app.core.exceptions import (
    ModelNotFoundError, 
    ModelLoadError, 
    DataFetchError,
    PredictionError
)

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing prediction models"""
    
    def __init__(self):
        self.current_predictor: Optional[BitcoinFallPredictor] = None
        self.current_metadata: Optional[Dict[str, Any]] = None
        self.current_ticker: Optional[str] = None
        self.model_path: Optional[str] = None
    
    def load_model(self, ticker: str) -> Dict[str, Any]:
        """Load model for a specific ticker"""
        try:
            model_base = Path(settings.MODEL_DIR) / f'{ticker.replace("-", "_")}_latest'
            
            if not model_base.exists():
                raise ModelNotFoundError(f"No model found for {ticker}")
            
            # Load metadata first
            with open(model_base / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            # Check if this is a demo model or real model
            predictor_file = model_base / 'predictor.pkl'
            demo_file = model_base / 'predictor_demo.txt'
            
            if predictor_file.exists():
                # Load real predictor
                with open(predictor_file, 'rb') as f:
                    predictor = pickle.load(f)
                logger.info(f"Loaded real trained model for {ticker}")
                
            elif demo_file.exists():
                # This is a demo model - create a mock predictor
                logger.warning(f"Loading demo model for {ticker} - predictions will be simulated")
                predictor = self._create_demo_predictor(ticker, metadata)
                
            else:
                raise ModelNotFoundError(f"No valid model files found for {ticker}")
            
            # Update instance variables
            self.current_predictor = predictor
            self.current_metadata = metadata
            self.current_ticker = ticker
            self.model_path = str(model_base)
            
            return {
                "ticker": ticker,
                "model_path": self.model_path,
                "metadata": metadata
            }
            
        except FileNotFoundError:
            raise ModelNotFoundError(f"Model files not found for {ticker}")
        except Exception as e:
            logger.error(f"Error loading model for {ticker}: {e}")
            raise ModelLoadError(f"Failed to load model: {str(e)}")
    
    def _create_demo_predictor(self, ticker: str, metadata: Dict[str, Any]):
        """Create a demo predictor that returns realistic but simulated predictions"""
        import yfinance as yf
        from datetime import datetime, timedelta
        
        # Create a simple demo predictor that fetches real price data
        # but returns simulated predictions
        class DemoPredictor:
            def __init__(self, ticker, metadata):
                self.ticker = ticker
                self.metadata = metadata
                self.global_selected_features = metadata.get('selected_features', [])
                self.models = {}
                
                # Create demo models
                for config in metadata['target_configs']:
                    self.models[config['name']] = DemoModel()
                
                # Get real price data for features
                try:
                    data = yf.download(ticker, period="30d")
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] for col in data.columns]
                    
                    # Create basic features dataframe
                    self.features_df = pd.DataFrame(index=data.index)
                    self.features_df['close'] = data['Close']
                    self.features_df['return'] = data['Close'].pct_change()
                    self.features_df['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
                    
                    # Add dummy features for the selected features
                    for feature in self.global_selected_features:
                        if feature not in self.features_df.columns:
                            self.features_df[feature] = np.random.randn(len(self.features_df)) * 0.1
                    
                    self.features_df = self.features_df.dropna()
                    
                except Exception as e:
                    logger.warning(f"Could not fetch real data for demo: {e}")
                    # Create minimal dummy data
                    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                    self.features_df = pd.DataFrame(index=dates)
                    self.features_df['close'] = 50000 + np.random.randn(30) * 1000
                    self.features_df['return'] = np.random.randn(30) * 0.02
                    self.features_df['log_return'] = np.random.randn(30) * 0.02
                    
                    for feature in self.global_selected_features:
                        if feature not in self.features_df.columns:
                            self.features_df[feature] = np.random.randn(30) * 0.1
            
            def _apply_cooldown(self, predictions, cooldown_days=30):
                """Apply cooldown logic (simplified for demo)"""
                result = []
                counter = 0
                
                for pred in predictions:
                    if pred == 1 and counter == 0:
                        counter = cooldown_days
                        result.append(1)
                    elif counter > 0:
                        counter -= 1
                        result.append(0)
                    else:
                        result.append(1)
                
                return pd.Series(result, index=self.features_df.index[:len(result)])
        
        class DemoModel:
            """Demo model that returns simulated predictions"""
            def predict(self, X):
                # Return realistic but random predictions
                # Bias towards staying in market (more 1s than 0s)
                predictions = np.random.choice([0, 1], size=len(X), p=[0.3, 0.7])
                return predictions
        
        return DemoPredictor(ticker, metadata)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        if self.current_predictor is None:
            return {
                "loaded": False,
                "ticker": None,
                "training_date": None,
                "data_end_date": None,
                "model_path": None,
                "model_version": None,
                "features_count": None
            }
        
        return {
            "loaded": True,
            "ticker": self.current_metadata.get('ticker'),
            "training_date": self.current_metadata.get('training_date'),
            "data_end_date": self.current_metadata.get('data_end_date'),
            "model_path": self.model_path,
            "model_version": self.current_metadata.get('training_date'),
            "features_count": len(self.current_metadata.get('selected_features', []))
        }
    
    def update_predictor_data(self) -> None:
        """Update predictor with latest market data"""
        if self.current_predictor is None:
            raise ModelLoadError("No model loaded")
        
        try:
            # Get latest data
            latest_date = self.current_predictor.features_df.index[-1]
            today = pd.Timestamp.now()
            
            if latest_date < today - timedelta(days=1):
                logger.info(f"Updating data from {latest_date} to {today}")
                
                # Download new data
                new_data = yf.download(
                    self.current_predictor.ticker, 
                    start=latest_date + timedelta(days=1)
                )
                
                if not new_data.empty:
                    # Fix column names if multi-level
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data.columns = [col[0] for col in new_data.columns]
                    
                    # Process new data
                    for idx in new_data.index:
                        if idx not in self.current_predictor.data.index:
                            row_data = {
                                'close': new_data.loc[idx, 'Close'],
                                'high': new_data.loc[idx, 'High'],
                                'low': new_data.loc[idx, 'Low'],
                                'open': new_data.loc[idx, 'Open'],
                                'volume': np.log(new_data.loc[idx, 'Volume']) if new_data.loc[idx, 'Volume'] > 0 else 0
                            }
                            
                            # Add to existing data
                            self.current_predictor.data.loc[idx] = row_data
                    
                    # Recalculate returns
                    self.current_predictor.data['return'] = self.current_predictor.data['close'].pct_change()
                    self.current_predictor.data['log_return'] = np.log(
                        self.current_predictor.data['close'] / self.current_predictor.data['close'].shift(1)
                    )
                    
                    # Re-engineer features for the updated data
                    self.current_predictor.engineer_features()
                    
                    logger.info(f"Data updated successfully to {self.current_predictor.features_df.index[-1]}")
                    
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            raise DataFetchError(self.current_ticker, str(e))
    
    def ensure_model_loaded(self, ticker: str) -> None:
        """Ensure model is loaded for the specified ticker"""
        if self.current_predictor is None or self.current_ticker != ticker:
            self.load_model(ticker)
        self.update_predictor_data()
    
    def get_current_prediction(self, ticker: str) -> Dict[str, Any]:
        """Get current prediction for a ticker"""
        try:
            # Ensure model is loaded and data is updated
            self.ensure_model_loaded(ticker)
            
            # Get target configs from metadata
            target_configs = [
                TargetConfig(**config) 
                for config in self.current_metadata['target_configs']
            ]
            
            # Get individual model predictions
            individual_predictions = {}
            for config in target_configs:
                model = self.current_predictor.models[config.name]
                X_selected = self.current_predictor.features_df[self.current_predictor.global_selected_features]
                predictions = model.predict(X_selected)
                binary_exits = self.current_predictor._apply_cooldown(
                    predictions, 
                    cooldown_days=config.cooldown_days
                )
                individual_predictions[config.name] = int(binary_exits.iloc[-1])
            
            # Calculate ensemble score
            ensemble_weights = self.current_metadata.get('ensemble_weights', settings.ENSEMBLE_WEIGHTS)
            ensemble_score = sum(
                individual_predictions[config.name] * weight 
                for config, weight in zip(target_configs, ensemble_weights)
            )
            
            # Apply uptrend filter
            uptrend = detect_uptrend(self.current_predictor.features_df['close'])
            uptrend_active = bool(uptrend.iloc[-1])
            
            signal_with_uptrend = int(ensemble_score * uptrend.iloc[-1])
            signal_without_uptrend = float(ensemble_score)
            
            return {
                "ticker": ticker,
                "date": self.current_predictor.features_df.index[-1].strftime('%Y-%m-%d'),
                "signal_with_uptrend": signal_with_uptrend,
                "signal_without_uptrend": signal_without_uptrend,
                "ensemble_score": float(ensemble_score),
                "individual_signals": individual_predictions,
                "uptrend_active": uptrend_active,
                "metadata": {
                    "ensemble_weights": ensemble_weights,
                    "target_configs": self.current_metadata['target_configs'],
                    "model_version": self.current_metadata.get('training_date'),
                    "data_last_updated": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction for {ticker}: {e}")
            raise PredictionError(str(e))
    
    def get_historical_predictions(self, ticker: str, days: int) -> Dict[str, Any]:
        """Get historical predictions for a ticker"""
        try:
            # Ensure model is loaded and data is updated
            self.ensure_model_loaded(ticker)
            
            # Get target configs
            target_configs = [
                TargetConfig(**config) 
                for config in self.current_metadata['target_configs']
            ]
            
            # Generate full predictions
            ensemble_predictions = self.current_predictor.generate_predictions(
                target_configs, 
                self.current_metadata.get('ensemble_weights', settings.ENSEMBLE_WEIGHTS)
            )
            
            # Get last N days
            history_df = self.current_predictor.features_df.iloc[-days:].copy()
            history_signals = ensemble_predictions.iloc[-days:]
            
            # Align indices
            common_index = history_df.index.intersection(history_signals.index)
            history_df = history_df.loc[common_index]
            history_signals = history_signals.loc[common_index]
            
            history_df['ensemble_signal'] = history_signals
            
            # Calculate returns for performance
            history_df['strategy_return'] = history_df['log_return'].shift(-1) * history_df['ensemble_signal']
            
            # Prepare predictions list
            predictions = []
            for idx in history_df.index:
                predictions.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'close': float(history_df.loc[idx, 'close']),
                    'signal': int(history_df.loc[idx, 'ensemble_signal']),
                    'return': float(history_df.loc[idx, 'return']) if not pd.isna(history_df.loc[idx, 'return']) else 0,
                    'strategy_return': float(history_df.loc[idx, 'strategy_return']) if not pd.isna(history_df.loc[idx, 'strategy_return']) else 0
                })
            
            # Calculate performance metrics
            strategy_returns = history_df['strategy_return'].dropna()
            benchmark_returns = history_df['log_return'].dropna()
            
            strategy_cumret = (1 + strategy_returns).cumprod()
            benchmark_cumret = (1 + benchmark_returns).cumprod()
            
            # Calculate metrics
            total_return = float(strategy_cumret.iloc[-1] - 1) if len(strategy_cumret) > 0 else 0
            benchmark_return = float(benchmark_cumret.iloc[-1] - 1) if len(benchmark_cumret) > 0 else 0
            
            # Calculate additional metrics
            sharpe_ratio = None
            max_drawdown = None
            volatility = None
            
            if len(strategy_returns) > 0:
                volatility = float(strategy_returns.std() * np.sqrt(252))
                if volatility > 0:
                    sharpe_ratio = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))
                
                # Calculate max drawdown
                cumret_series = strategy_cumret
                peak = cumret_series.cummax()
                drawdown = (cumret_series - peak) / peak
                max_drawdown = float(drawdown.min())
            
            performance_metrics = {
                'total_return': total_return,
                'benchmark_return': benchmark_return,
                'num_signals': int(history_df['ensemble_signal'].sum()),
                'win_rate': float((strategy_returns > 0).mean()) if len(strategy_returns) > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility
            }
            
            return {
                'ticker': ticker,
                'start_date': history_df.index[0].strftime('%Y-%m-%d'),
                'end_date': history_df.index[-1].strftime('%Y-%m-%d'),
                'predictions': predictions,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating historical predictions for {ticker}: {e}")
            raise PredictionError(str(e))


# Global service instance
model_service = ModelService()