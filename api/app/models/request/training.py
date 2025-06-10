"""
Request models for training endpoints
"""
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime, date


class TrainingRequest(BaseModel):
    """Request model for training a new model"""
    ticker: str = Field(
        description="Stock/Crypto ticker symbol to train model for",
        example="ETH-USD"
    )
    start_date: Optional[str] = Field(
        default="2018-01-01",
        description="Start date for training data in YYYY-MM-DD format",
        example="2018-01-01"
    )
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Ticker is required and cannot be empty")
        return v.upper().strip()
    
    @validator('start_date')
    def validate_start_date(cls, v):
        if v:
            try:
                parsed_date = datetime.strptime(v, '%Y-%m-%d').date()
                if parsed_date >= date.today():
                    raise ValueError("Start date must be in the past")
                if parsed_date < date(2010, 1, 1):
                    raise ValueError("Start date cannot be before 2010-01-01")
                return v
            except ValueError as e:
                if "does not match format" in str(e):
                    raise ValueError("Date must be in YYYY-MM-DD format")
                raise e
        return v


class TrainingStatus(BaseModel):
    """Model for training status"""
    job_id: str = Field(description="Training job identifier")
    ticker: str = Field(description="Ticker being trained")
    status: str = Field(description="Current status (pending, running, completed, failed)")
    progress: Optional[float] = Field(description="Progress percentage (0-100)")
    message: Optional[str] = Field(description="Status message")
    created_at: datetime = Field(description="Job creation time")
    updated_at: datetime = Field(description="Last update time")