"""
Validation dependencies for public API
"""
from fastapi import HTTPException, Query
from typing import Optional
import re
import logging

logger = logging.getLogger(__name__)


def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize ticker symbol
    """
    if not ticker:
        raise HTTPException(
            status_code=422,
            detail="Ticker symbol is required"
        )
    
    # Remove whitespace and convert to uppercase
    ticker = ticker.strip().upper()
    
    # Basic ticker format validation (letters, numbers, hyphens)
    if not re.match(r'^[A-Z0-9\-]+$', ticker):
        raise HTTPException(
            status_code=422,
            detail="Invalid ticker format. Use only letters, numbers, and hyphens."
        )
    
    # Length validation
    if len(ticker) < 2 or len(ticker) > 20:
        raise HTTPException(
            status_code=422,
            detail="Ticker symbol must be between 2 and 20 characters"
        )
    
    return ticker


def validate_days(days: int) -> int:
    """
    Validate number of days parameter
    """
    if days < 1:
        raise HTTPException(
            status_code=422,
            detail="Days must be at least 1"
        )
    
    if days > 365:
        raise HTTPException(
            status_code=422,
            detail="Days cannot exceed 365"
        )
    
    return days


async def get_validated_ticker(
    ticker: Optional[str] = Query(
        default="BTC-USD",
        description="Stock/Crypto ticker symbol"
    )
) -> str:
    """
    Dependency to get and validate ticker parameter
    """
    return validate_ticker(ticker)


async def get_validated_days(
    days: Optional[int] = Query(
        default=30,
        ge=1,
        le=365,
        description="Number of days to retrieve"
    )
) -> int:
    """
    Dependency to get and validate days parameter
    """
    return validate_days(days)


def validate_date_format(date_str: str) -> str:
    """
    Validate date string format (YYYY-MM-DD)
    """
    from datetime import datetime, date
    
    try:
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # Check if date is not in the future
        if parsed_date >= date.today():
            raise HTTPException(
                status_code=422,
                detail="Date must be in the past"
            )
        
        # Check if date is not too old (before 2010)
        if parsed_date < date(2010, 1, 1):
            raise HTTPException(
                status_code=422,
                detail="Date cannot be before 2010-01-01"
            )
        
        return date_str
        
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid date format. Use YYYY-MM-DD"
        )