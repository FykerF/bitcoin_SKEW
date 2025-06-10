"""
Script to train and save Bitcoin fall prediction models
"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
import logging

from generalized_fall_predictor import BitcoinFallPredictor, TargetConfig, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_save_models(ticker='BTC-USD', start_date='2018-01-01', model_dir='models'):
    """Train models and save them with metadata"""
    
    # Create models directory
    Path(model_dir).mkdir(exist_ok=True)
    
    # Initialize predictor
    logger.info(f"Training models for {ticker} from {start_date}")
    predictor = BitcoinFallPredictor(ticker=ticker, start_date=start_date, random_state=42)
    
    # Collect and engineer features
    predictor.collect_data()
    predictor.engineer_features()
    
    # Define target configurations
    target_configs = [
        TargetConfig(n_sigma=5, horizon_days=30, vol_window=30, name='5_30_30', cooldown_days=20),
        TargetConfig(n_sigma=2, horizon_days=7, vol_window=7, name='2_7_7', cooldown_days=7),
        TargetConfig(n_sigma=4, horizon_days=15, vol_window=15, name='4_15_15', cooldown_days=12)
    ]
    
    # Generate targets
    predictor.generate_targets(target_configs)
    
    # Train models
    model_config = ModelConfig(n_features=30, test_split=0.3, smote_strategy='aggressive')
    results = predictor.train_models(target_configs, model_config, smote_threshold=0.25)
    
    # Save models and metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = Path(model_dir) / f'{ticker.replace("-", "_")}_{timestamp}'
    model_path.mkdir(exist_ok=True)
    
    # Save the predictor object (contains all models and selected features)
    with open(model_path / 'predictor.pkl', 'wb') as f:
        pickle.dump(predictor, f)
    
    # Save target configurations
    target_configs_dict = [
        {
            'n_sigma': tc.n_sigma,
            'horizon_days': tc.horizon_days,
            'vol_window': tc.vol_window,
            'name': tc.name,
            'cooldown_days': tc.cooldown_days
        }
        for tc in target_configs
    ]
    
    # Save metadata
    metadata = {
        'ticker': ticker,
        'start_date': start_date,
        'training_date': timestamp,
        'target_configs': target_configs_dict,
        'model_config': {
            'n_features': model_config.n_features,
            'test_split': model_config.test_split,
            'smote_strategy': model_config.smote_strategy,
            'cv_folds': model_config.cv_folds
        },
        'ensemble_weights': [0.275, 0.45, 0.275],
        'selected_features': predictor.global_selected_features,
        'results': {
            name: {
                'train_precision': res['train_precision'],
                'test_precision': res['test_precision'],
                'best_params': res['best_params']
            }
            for name, res in results.items()
        },
        'data_end_date': predictor.features_df.index[-1].strftime('%Y-%m-%d')
    }
    
    with open(model_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create a link to latest model (Windows-compatible)
    latest_link = Path(model_dir) / f'{ticker.replace("-", "_")}_latest'
    
    # Remove existing link/directory
    if latest_link.exists():
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.is_dir():
            import shutil
            shutil.rmtree(latest_link)
        else:
            latest_link.unlink()
    
    # Try symlink first, fall back to copy on Windows
    try:
        latest_link.symlink_to(model_path.name)
        logger.info(f"Created symlink: {latest_link} -> {model_path.name}")
    except (OSError, NotImplementedError):
        # Windows fallback: copy directory instead of symlink
        import shutil
        shutil.copytree(model_path, latest_link)
        logger.info(f"Created copy (Windows fallback): {latest_link}")
    
    logger.info(f"Models saved to {model_path}")
    
    return str(model_path)


if __name__ == "__main__":
    # Train default BTC model
    model_path = train_and_save_models('BTC-USD', '2018-01-01')
    print(f"Models trained and saved to: {model_path}")