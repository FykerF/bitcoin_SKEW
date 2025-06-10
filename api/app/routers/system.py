"""
Router for system endpoints (health, status, etc.)
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging

from app.models.response.common import HealthResponse, ModelStatusResponse, ErrorResponse
from app.services.model_service import model_service
from app.core.config import settings

router = APIRouter(tags=["system"])
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is healthy and operational",
    responses={
        200: {"description": "API is healthy"},
        500: {"model": ErrorResponse, "description": "API is unhealthy"}
    }
)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns the current health status of the API along with
    timestamp and version information.
    """
    try:
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version=settings.VERSION
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="API is unhealthy"
        )


@router.get(
    "/model/status",
    response_model=ModelStatusResponse,
    summary="Get model status",
    description="Get information about the currently loaded model",
    responses={
        200: {"description": "Model status retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_model_status():
    """
    Get detailed information about the currently loaded model.
    
    Returns model metadata including ticker, training date,
    data coverage, and feature information.
    """
    try:
        logger.info("Getting model status")
        
        # Get model status from service
        status_data = model_service.get_model_status()
        
        return ModelStatusResponse(**status_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in get_model_status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/",
    summary="API information",
    description="Get basic API information",
    responses={
        200: {"description": "API information"}
    }
)
async def root():
    """
    Root endpoint providing basic API information.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "description": "Bitcoin Fall Prediction API for cryptocurrency market analysis",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "current_prediction": "/predict/today",
            "historical_predictions": "/predict/history", 
            "start_training": "/training/start",
            "training_status": "/training/status/{job_id}",
            "model_status": "/model/status"
        }
    }