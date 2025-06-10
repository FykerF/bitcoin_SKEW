"""
Common response models
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Health status")
    timestamp: datetime = Field(description="Current timestamp")
    version: str = Field(description="API version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0"
            }
        }


class ModelStatusResponse(BaseModel):
    """Model status response"""
    loaded: bool = Field(description="Whether a model is loaded")
    ticker: Optional[str] = Field(description="Currently loaded ticker")
    training_date: Optional[str] = Field(description="When the model was trained")
    data_end_date: Optional[str] = Field(description="End date of training data")
    model_path: Optional[str] = Field(description="Path to the model files")
    model_version: Optional[str] = Field(description="Model version identifier")
    features_count: Optional[int] = Field(description="Number of features in the model")
    
    class Config:
        schema_extra = {
            "example": {
                "loaded": True,
                "ticker": "BTC-USD",
                "training_date": "20241201_143022",
                "data_end_date": "2024-12-01",
                "model_path": "/app/models/BTC_USD_latest",
                "model_version": "v1.0",
                "features_count": 30
            }
        }


class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(description="Success message")
    data: Optional[Dict[str, Any]] = Field(description="Additional data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {}
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False, description="Operation success status")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(description="Error details")
    error_code: Optional[str] = Field(description="Error code for debugging")
    timestamp: datetime = Field(description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "message": "Model not found",
                "details": {
                    "ticker": "INVALID-TICKER"
                },
                "error_code": "MODEL_NOT_FOUND",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class TrainingResponse(BaseModel):
    """Training operation response"""
    success: bool = Field(description="Training request success")
    message: str = Field(description="Training status message")
    job_id: Optional[str] = Field(description="Training job identifier")
    ticker: str = Field(description="Ticker being trained")
    estimated_duration: Optional[int] = Field(description="Estimated training time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Training started for ETH-USD",
                "job_id": "train_ETH_USD_20241215_143022",
                "ticker": "ETH-USD",
                "estimated_duration": 1800
            }
        }