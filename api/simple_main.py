"""
Simplified main FastAPI application with minimal dependencies
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Bitcoin Fall Prediction API",
    version="1.0.0",
    description="Simple Bitcoin Fall Prediction API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple demo predictor
class SimpleDemoPredictor:
    def __init__(self):
        self.ticker = "BTC-USD"
        # Get real BTC data
        try:
            data = yf.download("BTC-USD", period="30d")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            self.price_data = data
            self.current_price = float(data['Close'].iloc[-1])
            self.current_date = data.index[-1].strftime('%Y-%m-%d')
            logger.info(f"Loaded real BTC data. Current price: ${self.current_price:,.2f}")
        except Exception as e:
            logger.warning(f"Could not load real data: {e}")
            self.current_price = 50000.0
            self.current_date = datetime.now().strftime('%Y-%m-%d')
    
    def get_prediction(self):
        # Simple demo logic: bias towards staying in market
        signal_raw = np.random.choice([0, 1], p=[0.3, 0.7])
        ensemble_score = np.random.uniform(0.4, 0.9)
        uptrend_active = np.random.choice([True, False], p=[0.6, 0.4])
        
        return {
            "ticker": self.ticker,
            "date": self.current_date,
            "signal_with_uptrend": signal_raw if uptrend_active else 0,
            "signal_without_uptrend": ensemble_score,
            "ensemble_score": ensemble_score,
            "individual_signals": {
                "5_30_30": np.random.choice([0, 1]),
                "2_7_7": np.random.choice([0, 1]),
                "4_15_15": np.random.choice([0, 1])
            },
            "uptrend_active": uptrend_active,
            "metadata": {
                "ensemble_weights": [0.275, 0.45, 0.275],
                "target_configs": [
                    {"name": "5_30_30", "n_sigma": 5, "horizon_days": 30},
                    {"name": "2_7_7", "n_sigma": 2, "horizon_days": 7},
                    {"name": "4_15_15", "n_sigma": 4, "horizon_days": 15}
                ],
                "model_version": "demo",
                "data_last_updated": datetime.now().isoformat()
            }
        }
    
    def get_history(self, days=30):
        # Generate demo historical data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        predictions = []
        for i, date in enumerate(dates):
            signal = np.random.choice([0, 1], p=[0.3, 0.7])
            ret = np.random.normal(0.001, 0.03)  # Daily return
            strategy_ret = ret * signal if signal == 1 else 0
            
            predictions.append({
                "date": date.strftime('%Y-%m-%d'),
                "close": 50000 + np.random.normal(0, 2000),
                "signal": signal,
                "return": ret,
                "strategy_return": strategy_ret
            })
        
        # Calculate performance metrics
        total_return = sum(p["strategy_return"] for p in predictions)
        benchmark_return = sum(p["return"] for p in predictions)
        num_signals = sum(p["signal"] for p in predictions)
        returns = [p["strategy_return"] for p in predictions if p["signal"] == 1]
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        
        return {
            "ticker": self.ticker,
            "start_date": dates[0].strftime('%Y-%m-%d'),
            "end_date": dates[-1].strftime('%Y-%m-%d'),
            "predictions": predictions,
            "performance_metrics": {
                "total_return": total_return,
                "benchmark_return": benchmark_return,
                "num_signals": num_signals,
                "win_rate": win_rate,
                "sharpe_ratio": None,
                "max_drawdown": None,
                "volatility": None
            }
        }

# Global predictor instance
predictor = SimpleDemoPredictor()

@app.get("/")
async def root():
    return {
        "name": "Bitcoin Fall Prediction API",
        "version": "1.0.0",
        "description": "Simple demo version",
        "endpoints": {
            "health": "/health",
            "prediction": "/predict/today",
            "history": "/predict/history",
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
    return {
        "loaded": True,
        "ticker": "BTC-USD",
        "training_date": "demo",
        "data_end_date": predictor.current_date,
        "model_path": "demo",
        "model_version": "demo",
        "features_count": 30
    }

@app.get("/predict/today")
async def predict_today(ticker: str = "BTC-USD"):
    try:
        prediction = predictor.get_prediction()
        prediction["ticker"] = ticker
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/history")
async def predict_history(ticker: str = "BTC-USD", days: int = 30):
    try:
        if days > 365:
            days = 365
        if days < 1:
            days = 1
            
        history = predictor.get_history(days)
        history["ticker"] = ticker
        return history
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/jobs")
async def training_jobs():
    return {
        "jobs": [],
        "total": 0,
        "active_count": 0,
        "completed_count": 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)