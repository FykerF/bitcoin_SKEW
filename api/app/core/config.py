"""
Core configuration for the Bitcoin Fall Prediction API
"""
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    APP_NAME: str = "Bitcoin Fall Prediction API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_METHODS: List[str] = ["*"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # Model Settings
    MODEL_DIR: str = os.getenv('MODEL_DIR', '../models')
    DEFAULT_TICKER: str = "BTC-USD"
    DEFAULT_START_DATE: str = "2018-01-01"
    
    # Ensemble Settings
    ENSEMBLE_WEIGHTS: List[float] = [0.275, 0.45, 0.275]
    
    # Target Configurations
    TARGET_CONFIGS: List[dict] = [
        {"n_sigma": 5, "horizon_days": 30, "vol_window": 30, "name": "5_30_30", "cooldown_days": 20},
        {"n_sigma": 2, "horizon_days": 7, "vol_window": 7, "name": "2_7_7", "cooldown_days": 7},
        {"n_sigma": 4, "horizon_days": 15, "vol_window": 15, "name": "4_15_15", "cooldown_days": 12}
    ]
    
    # Data Settings
    MAX_HISTORY_DAYS: int = 365
    DEFAULT_HISTORY_DAYS: int = 30
    
    # Training Settings
    TRAINING_TIMEOUT: int = 3600  # 1 hour
    MAX_CONCURRENT_TRAINING: int = 2
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()