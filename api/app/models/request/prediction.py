"""
Request models for prediction endpoints
"""
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for getting predictions"""
    ticker: Optional[str] = Field(
        default="BTC-USD",
        description="Stock/Crypto ticker symbol",
        example="BTC-USD"
    )
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if v and len(v.strip()) == 0:
            raise ValueError("Ticker cannot be empty")
        return v.upper() if v else v


class HistoricalPredictionRequest(BaseModel):
    """Request model for historical predictions"""
    ticker: Optional[str] = Field(
        default="BTC-USD",
        description="Stock/Crypto ticker symbol",
        example="BTC-USD"
    )
    days: Optional[int] = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to retrieve (1-365)",
        example=30
    )
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if v and len(v.strip()) == 0:
            raise ValueError("Ticker cannot be empty")
        return v.upper() if v else v