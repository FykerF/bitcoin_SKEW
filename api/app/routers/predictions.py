"""
Router for prediction endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
import logging

from app.models.request.prediction import PredictionRequest, HistoricalPredictionRequest
from app.models.response.prediction import CurrentPredictionResponse, HistoricalPredictionResponse
from app.models.response.common import ErrorResponse
from app.services.model_service import model_service
from app.core.exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    DataFetchError,
    PredictionError,
    model_not_found_exception,
    model_load_exception,
    data_fetch_exception,
    prediction_exception
)

router = APIRouter(prefix="/predict", tags=["predictions"])
logger = logging.getLogger(__name__)


@router.get(
    "/today",
    response_model=CurrentPredictionResponse,
    summary="Get current prediction",
    description="Get the current trading signal for today with and without uptrend filtering",
    responses={
        200: {"description": "Current prediction retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Model not found for ticker"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_current_prediction(
    ticker: Optional[str] = Query(
        default="BTC-USD",
        description="Stock/Crypto ticker symbol",
        example="BTC-USD"
    )
):
    """
    Get current trading prediction for the specified ticker.
    
    Returns both filtered (with uptrend) and unfiltered signals along with
    individual model outputs and ensemble scoring.
    """
    try:
        logger.info(f"Getting current prediction for {ticker}")
        
        # Validate and normalize ticker
        ticker = ticker.upper().strip()
        
        # Get prediction from service
        prediction_data = model_service.get_current_prediction(ticker)
        
        return CurrentPredictionResponse(**prediction_data)
        
    except ModelNotFoundError:
        raise model_not_found_exception(ticker)
    except ModelLoadError as e:
        raise model_load_exception(str(e))
    except DataFetchError as e:
        raise data_fetch_exception(ticker, str(e))
    except PredictionError as e:
        raise prediction_exception(str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_current_prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/history",
    response_model=HistoricalPredictionResponse,
    summary="Get historical predictions",
    description="Get historical predictions and performance metrics for the specified period",
    responses={
        200: {"description": "Historical predictions retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Model not found for ticker"},
        422: {"model": ErrorResponse, "description": "Invalid parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_historical_predictions(
    ticker: Optional[str] = Query(
        default="BTC-USD",
        description="Stock/Crypto ticker symbol",
        example="BTC-USD"
    ),
    days: Optional[int] = Query(
        default=30,
        ge=1,
        le=365,
        description="Number of days to retrieve (1-365)",
        example=30
    )
):
    """
    Get historical trading predictions and performance metrics.
    
    Returns historical signals, performance comparison with buy-and-hold,
    and downloadable data for the specified time period.
    """
    try:
        logger.info(f"Getting historical predictions for {ticker} ({days} days)")
        
        # Validate and normalize inputs
        ticker = ticker.upper().strip()
        
        # Get historical predictions from service
        history_data = model_service.get_historical_predictions(ticker, days)
        
        return HistoricalPredictionResponse(**history_data)
        
    except ModelNotFoundError:
        raise model_not_found_exception(ticker)
    except ModelLoadError as e:
        raise model_load_exception(str(e))
    except DataFetchError as e:
        raise data_fetch_exception(ticker, str(e))
    except PredictionError as e:
        raise prediction_exception(str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_historical_predictions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")