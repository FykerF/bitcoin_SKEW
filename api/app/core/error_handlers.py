"""
Global error handlers for the Bitcoin Fall Prediction API
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
from datetime import datetime
from typing import Union

from app.core.exceptions import BitcoinPredictionException

logger = logging.getLogger(__name__)


async def bitcoin_prediction_exception_handler(
    request: Request, 
    exc: BitcoinPredictionException
) -> JSONResponse:
    """Handle custom Bitcoin prediction exceptions"""
    logger.error(f"Bitcoin prediction error: {exc.message} - Details: {exc.details}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": exc.message,
            "details": exc.details,
            "error_code": exc.__class__.__name__.upper(),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )


async def http_exception_handler(
    request: Request, 
    exc: Union[HTTPException, StarletteHTTPException]
) -> JSONResponse:
    """Handle HTTP exceptions"""
    # Extract request ID if available
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"[{request_id}] HTTP {exc.status_code}: {exc.detail}")
    
    # Handle different status codes
    if exc.status_code == 404:
        message = "Resource not found"
    elif exc.status_code == 422:
        message = "Validation error"
    elif exc.status_code == 429:
        message = "Rate limit exceeded"
    elif exc.status_code >= 500:
        message = "Internal server error"
    else:
        message = "Request failed"
    
    # Prepare error details
    error_detail = exc.detail
    if isinstance(error_detail, dict):
        details = error_detail.get("details", {})
        message = error_detail.get("message", message)
    else:
        details = {"original_detail": str(error_detail)}
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": message,
            "details": details,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            "request_id": request_id
        }
    )


async def validation_exception_handler(
    request: Request, 
    exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"[{request_id}] Validation error: {exc.errors()}")
    
    # Extract validation error details
    validation_errors = []
    for error in exc.errors():
        validation_errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Request validation failed",
            "details": {
                "validation_errors": validation_errors
            },
            "error_code": "VALIDATION_ERROR",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            "request_id": request_id
        }
    )


async def general_exception_handler(
    request: Request, 
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(
        f"[{request_id}] Unexpected error: {type(exc).__name__}: {str(exc)}",
        exc_info=True
    )
    
    # Don't expose internal error details in production
    from app.core.config import settings
    
    if settings.DEBUG:
        details = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)
        }
    else:
        details = {}
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected error occurred",
            "details": details,
            "error_code": "INTERNAL_ERROR",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path),
            "request_id": request_id
        }
    )


def setup_error_handlers(app: FastAPI):
    """Setup all error handlers"""
    
    # Custom exception handlers
    app.add_exception_handler(
        BitcoinPredictionException, 
        bitcoin_prediction_exception_handler
    )
    
    # HTTP exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Validation exception handler
    app.add_exception_handler(
        RequestValidationError, 
        validation_exception_handler
    )
    
    # General exception handler (catch-all)
    app.add_exception_handler(Exception, general_exception_handler)