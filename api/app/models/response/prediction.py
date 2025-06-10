"""
Response models for prediction endpoints
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class PredictionMetadata(BaseModel):
    """Metadata for predictions"""
    ensemble_weights: List[float] = Field(description="Weights used for ensemble")
    target_configs: List[Dict[str, Any]] = Field(description="Target configurations used")
    model_version: Optional[str] = Field(description="Model version identifier")
    data_last_updated: Optional[str] = Field(description="When data was last updated")


class CurrentPredictionResponse(BaseModel):
    """Response model for current prediction"""
    ticker: str = Field(description="Stock/Crypto ticker symbol")
    date: str = Field(description="Date of prediction")
    signal_with_uptrend: int = Field(
        description="Binary signal with uptrend filter (1=in market, 0=exit)",
        ge=0,
        le=1
    )
    signal_without_uptrend: float = Field(
        description="Ensemble score without uptrend filter",
        ge=0.0,
        le=1.0
    )
    ensemble_score: float = Field(
        description="Raw ensemble score",
        ge=0.0,
        le=1.0
    )
    individual_signals: Dict[str, int] = Field(
        description="Individual model signals"
    )
    uptrend_active: bool = Field(description="Whether uptrend is currently active")
    metadata: PredictionMetadata = Field(description="Prediction metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "BTC-USD",
                "date": "2024-01-15",
                "signal_with_uptrend": 1,
                "signal_without_uptrend": 0.725,
                "ensemble_score": 0.725,
                "individual_signals": {
                    "5_30_30": 1,
                    "2_7_7": 0,
                    "4_15_15": 1
                },
                "uptrend_active": True,
                "metadata": {
                    "ensemble_weights": [0.275, 0.45, 0.275],
                    "target_configs": [],
                    "model_version": "20241201_143022",
                    "data_last_updated": "2024-01-15T10:30:00Z"
                }
            }
        }


class HistoricalDataPoint(BaseModel):
    """Single data point in historical predictions"""
    date: str = Field(description="Date in YYYY-MM-DD format")
    close: float = Field(description="Closing price")
    signal: int = Field(description="Trading signal (1=in market, 0=exit)")
    return_: float = Field(
        alias="return",
        description="Daily return"
    )
    strategy_return: float = Field(description="Strategy return for the day")
    
    class Config:
        allow_population_by_field_name = True


class PerformanceMetrics(BaseModel):
    """Performance metrics for strategy"""
    total_return: float = Field(description="Total strategy return")
    benchmark_return: float = Field(description="Buy & hold return")
    num_signals: int = Field(description="Number of signals generated")
    win_rate: float = Field(description="Percentage of profitable trades")
    sharpe_ratio: Optional[float] = Field(description="Sharpe ratio")
    max_drawdown: Optional[float] = Field(description="Maximum drawdown")
    volatility: Optional[float] = Field(description="Strategy volatility")


class HistoricalPredictionResponse(BaseModel):
    """Response model for historical predictions"""
    ticker: str = Field(description="Stock/Crypto ticker symbol")
    start_date: str = Field(description="Start date of data")
    end_date: str = Field(description="End date of data")
    predictions: List[HistoricalDataPoint] = Field(description="Historical predictions")
    performance_metrics: PerformanceMetrics = Field(description="Performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "BTC-USD",
                "start_date": "2024-01-01",
                "end_date": "2024-01-30",
                "predictions": [
                    {
                        "date": "2024-01-01",
                        "close": 42000.0,
                        "signal": 1,
                        "return": 0.025,
                        "strategy_return": 0.025
                    }
                ],
                "performance_metrics": {
                    "total_return": 0.15,
                    "benchmark_return": 0.12,
                    "num_signals": 20,
                    "win_rate": 0.65,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -0.05,
                    "volatility": 0.15
                }
            }
        }