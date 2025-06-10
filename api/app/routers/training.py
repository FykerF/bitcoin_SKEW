"""
Router for training endpoints
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import logging

from app.models.request.training import TrainingRequest, TrainingStatus
from app.models.response.common import TrainingResponse, SuccessResponse, ErrorResponse
from app.services.training_service import training_service
from app.core.exceptions import TrainingError

router = APIRouter(prefix="/training", tags=["training"])
logger = logging.getLogger(__name__)


@router.post(
    "/start",
    response_model=TrainingResponse,
    summary="Start model training",
    description="Start training a new model for the specified ticker",
    responses={
        200: {"description": "Training started successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        429: {"model": ErrorResponse, "description": "Too many concurrent training jobs"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def start_training(request: TrainingRequest):
    """
    Start training a new prediction model for the specified ticker.
    
    The training process runs in the background and can take several minutes
    to complete. Use the job_id from the response to check training status.
    """
    try:
        logger.info(f"Starting training for {request.ticker} from {request.start_date}")
        
        # Start training
        result = await training_service.start_training(
            ticker=request.ticker,
            start_date=request.start_date
        )
        
        return TrainingResponse(**result)
        
    except TrainingError as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(
            status_code=400 if "Maximum concurrent" in str(e) else 500,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in start_training: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/status/{job_id}",
    response_model=TrainingStatus,
    summary="Get training job status",
    description="Get the current status of a training job",
    responses={
        200: {"description": "Training status retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Training job not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_training_status(job_id: str):
    """
    Get the current status of a training job.
    
    Returns information about the training progress, current status,
    and any error messages if the training failed.
    """
    try:
        logger.info(f"Getting status for training job {job_id}")
        
        # Get job status
        job_status = training_service.get_job_status(job_id)
        
        if job_status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Training job {job_id} not found"
            )
        
        return TrainingStatus(**job_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_training_status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/jobs",
    summary="List training jobs",
    description="List all training jobs with optional status filtering",
    responses={
        200: {"description": "Training jobs retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def list_training_jobs(
    status: Optional[str] = None
):
    """
    List all training jobs with optional status filtering.
    
    Available statuses: pending, running, completed, failed
    """
    try:
        logger.info(f"Listing training jobs with status filter: {status}")
        
        # Validate status if provided
        valid_statuses = ["pending", "running", "completed", "failed"]
        if status and status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )
        
        # Get jobs list
        jobs_data = training_service.list_jobs(status)
        
        return jobs_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in list_training_jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete(
    "/cleanup",
    response_model=SuccessResponse,
    summary="Cleanup old training jobs",
    description="Remove old completed training jobs from memory",
    responses={
        200: {"description": "Cleanup completed successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def cleanup_training_jobs(
    max_age_hours: Optional[int] = 24
):
    """
    Clean up old completed training jobs from memory.
    
    This removes training job records older than the specified age
    to prevent memory buildup over time.
    """
    try:
        logger.info(f"Cleaning up training jobs older than {max_age_hours} hours")
        
        # Perform cleanup
        training_service.cleanup_old_jobs(max_age_hours)
        
        return SuccessResponse(
            message=f"Successfully cleaned up training jobs older than {max_age_hours} hours"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in cleanup_training_jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")