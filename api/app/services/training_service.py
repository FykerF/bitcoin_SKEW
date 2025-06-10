"""
Service layer for model training operations
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from train_and_save_model import train_and_save_models
from app.core.config import settings
from app.core.exceptions import TrainingError

logger = logging.getLogger(__name__)


class TrainingJob:
    """Training job representation"""
    
    def __init__(self, job_id: str, ticker: str, start_date: str):
        self.job_id = job_id
        self.ticker = ticker
        self.start_date = start_date
        self.status = "pending"
        self.progress = 0.0
        self.message = "Training job created"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.result: Optional[str] = None
        self.error: Optional[str] = None
    
    def update_status(self, status: str, progress: float = None, message: str = None):
        """Update job status"""
        self.status = status
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "ticker": self.ticker,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class TrainingService:
    """Service for managing model training"""
    
    def __init__(self):
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingJob] = {}
        self.max_concurrent_jobs = settings.MAX_CONCURRENT_TRAINING
    
    def get_running_jobs_count(self) -> int:
        """Get number of currently running jobs"""
        return len([job for job in self.active_jobs.values() if job.status == "running"])
    
    def can_start_new_job(self) -> bool:
        """Check if a new training job can be started"""
        return self.get_running_jobs_count() < self.max_concurrent_jobs
    
    def create_job_id(self, ticker: str) -> str:
        """Create a unique job ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"train_{ticker.replace('-', '_')}_{timestamp}_{unique_id}"
    
    async def start_training(self, ticker: str, start_date: str) -> Dict[str, Any]:
        """Start a new training job"""
        try:
            # Check if we can start a new job
            if not self.can_start_new_job():
                raise TrainingError(
                    f"Maximum concurrent training jobs ({self.max_concurrent_jobs}) reached. Please wait."
                )
            
            # Create job
            job_id = self.create_job_id(ticker)
            job = TrainingJob(job_id, ticker, start_date)
            self.active_jobs[job_id] = job
            
            logger.info(f"Starting training job {job_id} for {ticker}")
            
            # Start training in background
            asyncio.create_task(self._run_training(job))
            
            return {
                "success": True,
                "message": f"Training started for {ticker}",
                "job_id": job_id,
                "ticker": ticker,
                "estimated_duration": settings.TRAINING_TIMEOUT
            }
            
        except Exception as e:
            logger.error(f"Error starting training for {ticker}: {e}")
            raise TrainingError(str(e))
    
    async def _run_training(self, job: TrainingJob):
        """Run the actual training process"""
        try:
            job.update_status("running", 0.0, "Initializing training...")
            logger.info(f"Running training job {job.job_id}")
            
            # Update progress
            job.update_status("running", 10.0, "Downloading data...")
            
            # Run training in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Create a wrapper function for training
            def train_wrapper():
                try:
                    job.update_status("running", 30.0, "Engineering features...")
                    result = train_and_save_models(
                        ticker=job.ticker,
                        start_date=job.start_date,
                        model_dir=settings.MODEL_DIR
                    )
                    return result
                except Exception as e:
                    logger.error(f"Training failed for job {job.job_id}: {e}")
                    raise e
            
            # Execute training
            job.update_status("running", 50.0, "Training models...")
            result = await loop.run_in_executor(None, train_wrapper)
            
            job.update_status("running", 90.0, "Saving models...")
            
            # Mark as completed
            job.update_status("completed", 100.0, "Training completed successfully")
            job.result = result
            
            # Move to completed jobs
            self.completed_jobs[job.job_id] = job
            del self.active_jobs[job.job_id]
            
            logger.info(f"Training job {job.job_id} completed successfully")
            
        except Exception as e:
            # Mark as failed
            job.update_status("failed", job.progress, f"Training failed: {str(e)}")
            job.error = str(e)
            
            # Move to completed jobs
            self.completed_jobs[job.job_id] = job
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            logger.error(f"Training job {job.job_id} failed: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job"""
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].to_dict()
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id].to_dict()
        
        return None
    
    def list_jobs(self, status: Optional[str] = None) -> Dict[str, Any]:
        """List all jobs with optional status filter"""
        all_jobs = {}
        all_jobs.update(self.active_jobs)
        all_jobs.update(self.completed_jobs)
        
        if status:
            filtered_jobs = {
                job_id: job for job_id, job in all_jobs.items() 
                if job.status == status
            }
        else:
            filtered_jobs = all_jobs
        
        return {
            "jobs": [job.to_dict() for job in filtered_jobs.values()],
            "total": len(filtered_jobs),
            "active_count": len(self.active_jobs),
            "completed_count": len(self.completed_jobs)
        }
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        jobs_to_remove = [
            job_id for job_id, job in self.completed_jobs.items()
            if job.updated_at < cutoff_time
        ]
        
        for job_id in jobs_to_remove:
            del self.completed_jobs[job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old training jobs")


# Global service instance
training_service = TrainingService()