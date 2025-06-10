"""
Main FastAPI application for Bitcoin Fall Prediction API
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.middleware import setup_cors_middleware, setup_custom_middleware
from app.core.error_handlers import setup_error_handlers
from app.routers import predictions, training, system
from app.services.model_service import model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    
    # Try to load default model on startup
    try:
        model_service.load_model(settings.DEFAULT_TICKER)
        logger.info(f"Successfully loaded default model for {settings.DEFAULT_TICKER}")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")
        logger.info("API will still function, but models need to be loaded on first request")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    # Add any cleanup logic here if needed


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="""
    Bitcoin Fall Prediction API for cryptocurrency market analysis.
    
    This API provides machine learning-based predictions for Bitcoin market falls
    using an ensemble of logistic regression models with comprehensive feature engineering.
    
    ## Features
    
    * **Real-time Predictions**: Get current market position signals
    * **Historical Analysis**: View past performance and download data
    * **Model Training**: Train new models for different tickers
    * **Uptrend Filtering**: Signals with and without uptrend detection
    * **Ensemble Scoring**: Multiple model combination for robust predictions
    
    ## Model Details
    
    The system uses three specialized models:
    - **5_30_30**: 5-sigma fall prediction over 30 days
    - **2_7_7**: 2-sigma fall prediction over 7 days  
    - **4_15_15**: 4-sigma fall prediction over 15 days
    
    Ensemble weights: [0.275, 0.45, 0.275]
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Setup middleware
setup_cors_middleware(app)
setup_custom_middleware(app)

# Setup error handlers
setup_error_handlers(app)

# Include routers
app.include_router(system.router)
app.include_router(predictions.router)
app.include_router(training.router)

# Health check endpoint at root level
@app.get("/health", tags=["system"])
async def health():
    """Simple health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )