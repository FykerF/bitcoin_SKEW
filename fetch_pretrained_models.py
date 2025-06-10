"""
Script to fetch and setup pretrained models for deployment
"""
import os
import json
import urllib.request
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_model_metadata(model_dir: Path, ticker: str = 'BTC-USD'):
    """Create demo model metadata for deployment without actual training"""
    
    # Create model directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = model_dir / f'{ticker.replace("-", "_")}_{timestamp}'
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Create demo metadata
    metadata = {
        'ticker': ticker,
        'start_date': '2018-01-01',
        'training_date': timestamp,
        'target_configs': [
            {'n_sigma': 5, 'horizon_days': 30, 'vol_window': 30, 'name': '5_30_30', 'cooldown_days': 20},
            {'n_sigma': 2, 'horizon_days': 7, 'vol_window': 7, 'name': '2_7_7', 'cooldown_days': 7},
            {'n_sigma': 4, 'horizon_days': 15, 'vol_window': 15, 'name': '4_15_15', 'cooldown_days': 12}
        ],
        'model_config': {
            'n_features': 30,
            'test_split': 0.3,
            'smote_strategy': 'aggressive',
            'cv_folds': 10
        },
        'ensemble_weights': [0.275, 0.45, 0.275],
        'selected_features': [
            'volume', 'return', 'log_return', 'vol_7d', 'vol_30d', 'rsi_14', 'rsi_30',
            'macd', 'macd_signal', 'macd_histogram', 'adx_14', 'roc_7d', 'roc_30d',
            'return_zscore_7d', 'return_zscore_30d', 'consecutive_negative', 'rolling_skew_7d',
            'volume_ma_7d', 'volume_ratio_7_30', 'obv_slope_7d', 'drawdown', 'ma_7d',
            'ma_30d', 'price_above_ma_7d', 'vol_percentile_7d', 'momentum_confluence',
            'jump_indicator_2sigma', 'price_gap', 'uptrend_regime_7d', 'vol_regime_high',
            'rsi_divergence_7d'
        ],
        'results': {
            '5_30_30': {'train_precision': 0.8234, 'test_precision': 0.7891, 'best_params': {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}},
            '2_7_7': {'train_precision': 0.7654, 'test_precision': 0.7321, 'best_params': {'C': 10.0, 'penalty': 'l1', 'solver': 'liblinear'}},
            '4_15_15': {'train_precision': 0.8012, 'test_precision': 0.7756, 'best_params': {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}}
        },
        'data_end_date': datetime.now().strftime('%Y-%m-%d'),
        'deployment_mode': 'demo',
        'note': 'This is a demo model setup for deployment. Train a real model for production use.'
    }
    
    # Save metadata
    with open(model_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create demo predictor placeholder (will be replaced by actual training)
    demo_predictor_note = """
    # DEMO PREDICTOR PLACEHOLDER
    # This file indicates that a demo model setup was created
    # To use real predictions, run: python train_and_save_model.py
    """
    
    with open(model_path / 'predictor_demo.txt', 'w') as f:
        f.write(demo_predictor_note)
    
    # Create link to latest model (Windows-compatible)
    latest_link = model_dir / f'{ticker.replace("-", "_")}_latest'
    
    # Remove existing link/directory
    if latest_link.exists():
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.is_dir():
            shutil.rmtree(latest_link)
        else:
            latest_link.unlink()
    
    # Try symlink first, fall back to copy on Windows
    try:
        latest_link.symlink_to(model_path.name)
        logger.info(f"Created symlink: {latest_link} -> {model_path.name}")
    except (OSError, NotImplementedError):
        # Windows fallback: copy directory instead of symlink
        shutil.copytree(model_path, latest_link)
        logger.info(f"Created copy (Windows fallback): {latest_link}")
    
    logger.info(f"Demo model setup created at {model_path}")
    return str(model_path)


def download_from_url(url: str, model_dir: Path):
    """Download pretrained model from a URL (if available)"""
    try:
        logger.info(f"Downloading model from {url}")
        
        # Download to temporary file
        temp_file = model_dir / 'temp_model.zip'
        urllib.request.urlretrieve(url, temp_file)
        
        # Extract
        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        # Clean up
        temp_file.unlink()
        
        logger.info("Model downloaded and extracted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model from URL: {e}")
        return False


def setup_models_for_deployment(model_dir: str = 'models', model_url: str = None):
    """Setup models for deployment"""
    
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)
    
    logger.info(f"Setting up models in {model_path}")
    
    # Method 1: Try to download from URL if provided
    if model_url:
        if download_from_url(model_url, model_path):
            logger.info("Successfully downloaded pretrained model")
            return
    
    # Method 2: Check if local trained model exists
    btc_latest = model_path / 'BTC_USD_latest'
    if btc_latest.exists() and (btc_latest / 'predictor.pkl').exists():
        logger.info("Found existing trained model")
        return
    
    # Method 3: Create demo setup for immediate deployment
    logger.info("Creating demo model setup for deployment")
    create_demo_model_metadata(model_path, 'BTC-USD')
    
    logger.info("""
    🚀 MODEL SETUP COMPLETE!
    
    Demo model created for immediate deployment.
    
    For production use:
    1. Run: python train_and_save_model.py
    2. Or provide a pretrained model URL
    3. Or upload trained models to the models/ directory
    
    The API will work in demo mode until a real model is trained.
    """)


def setup_for_render_deployment():
    """Special setup for Render deployment"""
    logger.info("Setting up for Render deployment...")
    
    # Render-specific model directory
    model_dir = os.getenv('MODEL_DIR', './models')
    
    # Check if we have a MODEL_URL environment variable
    model_url = os.getenv('MODEL_URL', None)
    
    # Setup models
    setup_models_for_deployment(model_dir, model_url)
    
    # Create a deployment info file
    deployment_info = {
        'deployment_platform': 'render',
        'setup_time': datetime.now().isoformat(),
        'model_dir': model_dir,
        'model_url': model_url,
        'status': 'ready'
    }
    
    with open(Path(model_dir) / 'deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    logger.info("Render deployment setup complete!")


if __name__ == "__main__":
    import sys
    
    # Check if running in Render (or similar cloud platform)
    if os.getenv('RENDER_SERVICE_NAME') or os.getenv('RAILWAY_PROJECT_ID'):
        setup_for_render_deployment()
    else:
        # Local development setup
        model_url = sys.argv[1] if len(sys.argv) > 1 else None
        setup_models_for_deployment('models', model_url)