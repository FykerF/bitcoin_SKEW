"""
Custom exceptions for the Bitcoin Fall Prediction API
"""
from fastapi import HTTPException
from typing import Any, Dict, Optional


class BitcoinPredictionException(Exception):
    """Base exception class for Bitcoin prediction errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ModelNotFoundError(BitcoinPredictionException):
    """Raised when a model is not found"""
    pass


class ModelLoadError(BitcoinPredictionException):
    """Raised when model loading fails"""
    pass


class DataFetchError(BitcoinPredictionException):
    """Raised when data fetching fails"""
    pass


class PredictionError(BitcoinPredictionException):
    """Raised when prediction generation fails"""
    pass


class TrainingError(BitcoinPredictionException):
    """Raised when model training fails"""
    pass


class ValidationError(BitcoinPredictionException):
    """Raised when input validation fails"""
    pass


# HTTP Exception creators
def create_http_exception(
    status_code: int,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create an HTTP exception with details"""
    return HTTPException(
        status_code=status_code,
        detail={
            "message": message,
            "details": details or {}
        }
    )


def model_not_found_exception(ticker: str) -> HTTPException:
    """Create a model not found exception"""
    return create_http_exception(
        status_code=404,
        message=f"No model found for ticker {ticker}",
        details={"ticker": ticker}
    )


def model_load_exception(error: str) -> HTTPException:
    """Create a model loading exception"""
    return create_http_exception(
        status_code=500,
        message="Failed to load model",
        details={"error": str(error)}
    )


def data_fetch_exception(ticker: str, error: str) -> HTTPException:
    """Create a data fetching exception"""
    return create_http_exception(
        status_code=500,
        message=f"Failed to fetch data for {ticker}",
        details={"ticker": ticker, "error": str(error)}
    )


def prediction_exception(error: str) -> HTTPException:
    """Create a prediction generation exception"""
    return create_http_exception(
        status_code=500,
        message="Failed to generate prediction",
        details={"error": str(error)}
    )


def validation_exception(field: str, error: str) -> HTTPException:
    """Create a validation exception"""
    return create_http_exception(
        status_code=422,
        message=f"Validation error for field {field}",
        details={"field": field, "error": str(error)}
    )