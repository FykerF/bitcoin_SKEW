"""
FastAPI application for Bitcoin Fall Prediction
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import logging
import os
import sys

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generalized_fall_predictor import BitcoinFallPredictor, TargetConfig, ModelConfig
from uptrend_detector import detect_uptrend

app = FastAPI(title="Bitcoin Fall Prediction API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
MODEL_DIR = os.getenv('MODEL_DIR', '../models')
CURRENT_MODEL_PATH = None
CURRENT_PREDICTOR = None
CURRENT_METADATA = None


class PredictionResponse(BaseModel):
    ticker: str
    date: str
    signal_with_uptrend: int
    signal_without_uptrend: float
    ensemble_score: float
    individual_signals: Dict[str, int]
    uptrend_active: bool
    metadata: Dict


class HistoricalPredictionResponse(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    predictions: List[Dict]
    performance_metrics: Dict


class TrainRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = '2018-01-01'


class ModelStatus(BaseModel):
    loaded: bool
    ticker: Optional[str]
    training_date: Optional[str]
    data_end_date: Optional[str]
    model_path: Optional[str]


def load_model(ticker: str = 'BTC-USD'):
    """Load the latest model for a given ticker"""
    global CURRENT_MODEL_PATH, CURRENT_PREDICTOR, CURRENT_METADATA
    
    model_base = Path(MODEL_DIR) / f'{ticker.replace("-", "_")}_latest'
    
    if not model_base.exists():
        raise HTTPException(status_code=404, detail=f"No model found for {ticker}")
    
    # Load predictor
    with open(model_base / 'predictor.pkl', 'rb') as f:
        CURRENT_PREDICTOR = pickle.load(f)
    
    # Load metadata
    with open(model_base / 'metadata.json', 'r') as f:
        CURRENT_METADATA = json.load(f)
    
    CURRENT_MODEL_PATH = str(model_base)
    logger.info(f"Loaded model from {CURRENT_MODEL_PATH}")


def update_predictor_data(predictor: BitcoinFallPredictor):
    """Update predictor with latest data"""
    # Get latest data
    latest_date = predictor.features_df.index[-1]
    today = pd.Timestamp.now()
    
    if latest_date < today - timedelta(days=1):
        logger.info(f"Updating data from {latest_date} to {today}")
        
        # Download new data
        new_data = yf.download(predictor.ticker, start=latest_date + timedelta(days=1))
        
        if not new_data.empty:
            # Fix column names if multi-level
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = [col[0] for col in new_data.columns]
            
            # Process new data
            for idx in new_data.index:
                row_data = {
                    'close': new_data.loc[idx, 'Close'],
                    'high': new_data.loc[idx, 'High'],
                    'low': new_data.loc[idx, 'Low'],
                    'open': new_data.loc[idx, 'Open'],
                    'volume': np.log(new_data.loc[idx, 'Volume'])
                }
                
                # Add to existing data
                predictor.data.loc[idx] = row_data
            
            # Recalculate returns
            predictor.data['return'] = predictor.data['close'].pct_change()
            predictor.data['log_return'] = np.log(predictor.data['close'] / predictor.data['close'].shift(1))
            
            # Re-engineer features for the updated data
            predictor.engineer_features()


@app.on_event("startup")
async def startup_event():
    """Load default model on startup"""
    try:
        load_model('BTC-USD')
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Bitcoin Fall Prediction API", "version": "1.0.0"}


@app.get("/model/status", response_model=ModelStatus)
async def get_model_status():
    """Get current model status"""
    if CURRENT_PREDICTOR is None:
        return ModelStatus(loaded=False)
    
    return ModelStatus(
        loaded=True,
        ticker=CURRENT_METADATA.get('ticker'),
        training_date=CURRENT_METADATA.get('training_date'),
        data_end_date=CURRENT_METADATA.get('data_end_date'),
        model_path=CURRENT_MODEL_PATH
    )


@app.get("/predict/today", response_model=PredictionResponse)
async def predict_today(ticker: Optional[str] = 'BTC-USD'):
    """Get prediction for today"""
    # Load model if needed
    if CURRENT_PREDICTOR is None or CURRENT_METADATA.get('ticker') != ticker:
        load_model(ticker)
    
    # Update data
    update_predictor_data(CURRENT_PREDICTOR)
    
    # Get target configs from metadata
    target_configs = [
        TargetConfig(**config) 
        for config in CURRENT_METADATA['target_configs']
    ]
    
    # Generate predictions
    ensemble_weights = CURRENT_METADATA.get('ensemble_weights', [0.275, 0.45, 0.275])
    
    # Get individual model predictions
    individual_predictions = {}
    for config in target_configs:
        model = CURRENT_PREDICTOR.models[config.name]
        X_selected = CURRENT_PREDICTOR.features_df[CURRENT_PREDICTOR.global_selected_features]
        predictions = model.predict(X_selected)
        binary_exits = CURRENT_PREDICTOR._apply_cooldown(predictions, cooldown_days=config.cooldown_days)
        individual_predictions[config.name] = binary_exits.iloc[-1]
    
    # Calculate ensemble score
    ensemble_score = sum(
        individual_predictions[config.name] * weight 
        for config, weight in zip(target_configs, ensemble_weights)
    )
    
    # Apply uptrend filter
    uptrend = detect_uptrend(CURRENT_PREDICTOR.features_df['close'])
    uptrend_active = bool(uptrend.iloc[-1])
    
    signal_with_uptrend = int(ensemble_score * uptrend.iloc[-1])
    signal_without_uptrend = float(ensemble_score)
    
    return PredictionResponse(
        ticker=ticker,
        date=CURRENT_PREDICTOR.features_df.index[-1].strftime('%Y-%m-%d'),
        signal_with_uptrend=signal_with_uptrend,
        signal_without_uptrend=signal_without_uptrend,
        ensemble_score=float(ensemble_score),
        individual_signals={k: int(v) for k, v in individual_predictions.items()},
        uptrend_active=uptrend_active,
        metadata={
            'ensemble_weights': ensemble_weights,
            'target_configs': CURRENT_METADATA['target_configs']
        }
    )


@app.get("/predict/history", response_model=HistoricalPredictionResponse)
async def predict_history(
    ticker: Optional[str] = 'BTC-USD',
    days: Optional[int] = 30
):
    """Get historical predictions"""
    # Load model if needed
    if CURRENT_PREDICTOR is None or CURRENT_METADATA.get('ticker') != ticker:
        load_model(ticker)
    
    # Update data
    update_predictor_data(CURRENT_PREDICTOR)
    
    # Get target configs
    target_configs = [
        TargetConfig(**config) 
        for config in CURRENT_METADATA['target_configs']
    ]
    
    # Generate full predictions
    ensemble_predictions = CURRENT_PREDICTOR.generate_predictions(
        target_configs, 
        CURRENT_METADATA.get('ensemble_weights', [0.275, 0.45, 0.275])
    )
    
    # Get last N days
    history_df = CURRENT_PREDICTOR.features_df.iloc[-days:].copy()
    history_df['ensemble_signal'] = ensemble_predictions.iloc[-days:]
    
    # Calculate returns for performance
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
    strategy_cumret = (1 + history_df['strategy_return']).cumprod().dropna()
    benchmark_cumret = (1 + history_df['log_return']).cumprod().dropna()
    
    metrics = {
        'total_return': float(strategy_cumret.iloc[-1] - 1) if len(strategy_cumret) > 0 else 0,
        'benchmark_return': float(benchmark_cumret.iloc[-1] - 1) if len(benchmark_cumret) > 0 else 0,
        'num_signals': int(history_df['ensemble_signal'].sum()),
        'win_rate': float((history_df['strategy_return'] > 0).mean()) if len(history_df) > 0 else 0
    }
    
    return HistoricalPredictionResponse(
        ticker=ticker,
        start_date=history_df.index[0].strftime('%Y-%m-%d'),
        end_date=history_df.index[-1].strftime('%Y-%m-%d'),
        predictions=predictions,
        performance_metrics=metrics
    )


@app.post("/train", response_model=Dict)
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train a new model for a given ticker"""
    # Import here to avoid circular imports
    from train_and_save_model import train_and_save_models
    
    # Add training task to background
    background_tasks.add_task(
        train_and_save_models,
        request.ticker,
        request.start_date,
        MODEL_DIR
    )
    
    return {
        "message": f"Training started for {request.ticker}",
        "status": "processing"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)