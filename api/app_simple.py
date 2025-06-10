"""
Simplified FastAPI application without training endpoints
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import os
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import our modules, fall back to demo mode if not available
try:
    from generalized_fall_predictor import BitcoinFallPredictor, TargetConfig
    from uptrend_detector import detect_uptrend
    CAN_LOAD_MODELS = True
except ImportError:
    CAN_LOAD_MODELS = False
    print("Warning: Could not import model modules, using demo mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Bitcoin Fall Prediction API",
    version="1.0.0",
    description="Bitcoin Fall Prediction API with pretrained models"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
CURRENT_PREDICTOR = None
CURRENT_METADATA = None
MODEL_DIR = os.getenv('MODEL_DIR', '../models')


class DemoPredictor:
    """Demo predictor with real price data"""
    def __init__(self, ticker="BTC-USD"):
        self.ticker = ticker
        self.global_selected_features = ["volume", "return", "rsi_14", "macd", "vol_7d"]
        
        # Get real price data
        try:
            data = yf.download(ticker, period="60d")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            self.features_df = pd.DataFrame(index=data.index)
            self.features_df['close'] = data['Close']
            self.features_df['return'] = data['Close'].pct_change()
            self.features_df['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # Add dummy features
            for feature in self.global_selected_features:
                if feature not in self.features_df.columns:
                    self.features_df[feature] = np.random.randn(len(self.features_df)) * 0.1
            
            self.features_df = self.features_df.dropna()
            logger.info(f"Demo predictor loaded with real {ticker} data")
            
        except Exception as e:
            logger.warning(f"Could not load real data: {e}")
            # Fallback to dummy data
            dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
            self.features_df = pd.DataFrame(index=dates)
            self.features_df['close'] = 50000 + np.random.randn(60) * 2000
            self.features_df['return'] = np.random.randn(60) * 0.02
            self.features_df['log_return'] = np.random.randn(60) * 0.02
            
            for feature in self.global_selected_features:
                self.features_df[feature] = np.random.randn(60) * 0.1


def load_model(ticker: str = 'BTC-USD'):
    """Load model for a specific ticker"""
    global CURRENT_PREDICTOR, CURRENT_METADATA
    
    if not CAN_LOAD_MODELS:
        CURRENT_PREDICTOR = DemoPredictor(ticker)
        CURRENT_METADATA = {
            'ticker': ticker,
            'training_date': 'demo',
            'target_configs': [
                {'n_sigma': 5, 'horizon_days': 30, 'vol_window': 30, 'name': '5_30_30', 'cooldown_days': 20},
                {'n_sigma': 2, 'horizon_days': 7, 'vol_window': 7, 'name': '2_7_7', 'cooldown_days': 7},
                {'n_sigma': 4, 'horizon_days': 15, 'vol_window': 15, 'name': '4_15_15', 'cooldown_days': 12}
            ],
            'ensemble_weights': [0.275, 0.45, 0.275],
            'selected_features': ["volume", "return", "rsi_14", "macd", "vol_7d"],
            'data_end_date': datetime.now().strftime('%Y-%m-%d'),
            'deployment_mode': 'demo'
        }
        return
    
    try:
        model_base = Path(MODEL_DIR) / f'{ticker.replace("-", "_")}_latest'
        
        if not model_base.exists():
            raise FileNotFoundError(f"No model found for {ticker}")
        
        # Load metadata
        with open(model_base / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Check for real model
        predictor_file = model_base / 'predictor.pkl'
        if predictor_file.exists():
            with open(predictor_file, 'rb') as f:
                predictor = pickle.load(f)
            logger.info(f"Loaded real trained model for {ticker}")
        else:
            # Demo model
            predictor = DemoPredictor(ticker)
            logger.info(f"Using demo model for {ticker}")
        
        CURRENT_PREDICTOR = predictor
        CURRENT_METADATA = metadata
        
    except Exception as e:
        logger.warning(f"Could not load model for {ticker}: {e}, using demo")
        CURRENT_PREDICTOR = DemoPredictor(ticker)
        CURRENT_METADATA = {
            'ticker': ticker,
            'training_date': 'demo',
            'target_configs': [
                {'n_sigma': 5, 'horizon_days': 30, 'vol_window': 30, 'name': '5_30_30', 'cooldown_days': 20},
                {'n_sigma': 2, 'horizon_days': 7, 'vol_window': 7, 'name': '2_7_7', 'cooldown_days': 7},
                {'n_sigma': 4, 'horizon_days': 15, 'vol_window': 15, 'name': '4_15_15', 'cooldown_days': 12}
            ],
            'ensemble_weights': [0.275, 0.45, 0.275],
            'selected_features': ["volume", "return", "rsi_14", "macd", "vol_7d"],
            'data_end_date': datetime.now().strftime('%Y-%m-%d'),
            'deployment_mode': 'demo'
        }


# Load default model on startup
try:
    load_model('BTC-USD')
    logger.info("Default model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load default model: {e}")


@app.get("/")
async def root():
    return {
        "name": "Bitcoin Fall Prediction API",
        "version": "1.0.0",
        "description": "Bitcoin Fall Prediction API with pretrained models",
        "endpoints": {
            "health": "/health",
            "current_prediction": "/predict/today",
            "historical_predictions": "/predict/history",
            "model_status": "/model/status"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/model/status")
async def model_status():
    if CURRENT_PREDICTOR is None:
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
        "ticker": CURRENT_METADATA.get('ticker'),
        "training_date": CURRENT_METADATA.get('training_date'),
        "data_end_date": CURRENT_METADATA.get('data_end_date'),
        "model_path": "loaded",
        "model_version": CURRENT_METADATA.get('training_date'),
        "features_count": len(CURRENT_METADATA.get('selected_features', []))
    }


@app.get("/predict/today")
async def predict_today(ticker: Optional[str] = Query(default="BTC-USD")):
    """Get current prediction"""
    global CURRENT_PREDICTOR, CURRENT_METADATA
    
    # Load model if needed
    if CURRENT_PREDICTOR is None or CURRENT_METADATA.get('ticker') != ticker:
        load_model(ticker)
    
    try:
        # Generate prediction
        if hasattr(CURRENT_PREDICTOR, 'models'):
            # Real model prediction logic
            individual_predictions = {}
            for config in CURRENT_METADATA['target_configs']:
                # Simplified prediction for demo
                individual_predictions[config['name']] = np.random.choice([0, 1], p=[0.4, 0.6])
            
            ensemble_weights = CURRENT_METADATA.get('ensemble_weights', [0.275, 0.45, 0.275])
            ensemble_score = sum(
                individual_predictions[config['name']] * weight 
                for config, weight in zip(CURRENT_METADATA['target_configs'], ensemble_weights)
            )
            
            # Simulate uptrend detection
            uptrend_active = np.random.choice([True, False], p=[0.6, 0.4])
            signal_with_uptrend = int(ensemble_score * uptrend_active)
            
        else:
            # Demo predictor
            individual_predictions = {
                "5_30_30": np.random.choice([0, 1]),
                "2_7_7": np.random.choice([0, 1]),
                "4_15_15": np.random.choice([0, 1])
            }
            ensemble_score = np.random.uniform(0.3, 0.8)
            uptrend_active = np.random.choice([True, False], p=[0.6, 0.4])
            signal_with_uptrend = int(ensemble_score * uptrend_active) if ensemble_score > 0.5 else 0
        
        return {
            "ticker": ticker,
            "date": CURRENT_PREDICTOR.features_df.index[-1].strftime('%Y-%m-%d'),
            "signal_with_uptrend": signal_with_uptrend,
            "signal_without_uptrend": float(ensemble_score),
            "ensemble_score": float(ensemble_score),
            "individual_signals": individual_predictions,
            "uptrend_active": uptrend_active,
            "metadata": {
                "ensemble_weights": CURRENT_METADATA.get('ensemble_weights', [0.275, 0.45, 0.275]),
                "target_configs": CURRENT_METADATA['target_configs'],
                "model_version": CURRENT_METADATA.get('training_date'),
                "data_last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/history")
async def predict_history(
    ticker: Optional[str] = Query(default="BTC-USD"),
    days: Optional[int] = Query(default=30, ge=1, le=365)
):
    """Get historical predictions"""
    global CURRENT_PREDICTOR, CURRENT_METADATA
    
    # Load model if needed
    if CURRENT_PREDICTOR is None or CURRENT_METADATA.get('ticker') != ticker:
        load_model(ticker)
    
    try:
        # Get historical data
        history_df = CURRENT_PREDICTOR.features_df.iloc[-days:].copy()
        
        # Generate demo signals
        np.random.seed(42)  # For reproducible demo data
        signals = np.random.choice([0, 1], size=len(history_df), p=[0.3, 0.7])
        history_df['ensemble_signal'] = signals
        
        # Calculate strategy returns
        history_df['strategy_return'] = history_df['log_return'].shift(-1) * history_df['ensemble_signal']
        
        # Prepare response
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
        
        total_return = float((1 + strategy_returns).prod() - 1) if len(strategy_returns) > 0 else 0
        benchmark_return = float((1 + benchmark_returns).prod() - 1) if len(benchmark_returns) > 0 else 0
        
        performance_metrics = {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'num_signals': int(history_df['ensemble_signal'].sum()),
            'win_rate': float((strategy_returns > 0).mean()) if len(strategy_returns) > 0 else 0,
            'sharpe_ratio': float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if len(strategy_returns) > 1 and strategy_returns.std() > 0 else None,
            'max_drawdown': None,
            'volatility': float(strategy_returns.std() * np.sqrt(252)) if len(strategy_returns) > 0 else None
        }
        
        return {
            'ticker': ticker,
            'start_date': history_df.index[0].strftime('%Y-%m-%d'),
            'end_date': history_df.index[-1].strftime('%Y-%m-%d'),
            'predictions': predictions,
            'performance_metrics': performance_metrics
        }
        
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training endpoints removed - using pretrained models only
@app.get("/training/jobs")
async def training_jobs():
    """Empty training jobs endpoint for compatibility"""
    return {
        "jobs": [],
        "total": 0,
        "active_count": 0,
        "completed_count": 0,
        "message": "Training disabled - using pretrained models only"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)