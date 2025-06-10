"""
Minimal predictor that exactly mimics rouf_baseline structure for debugging
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from imblearn.over_sampling import SMOTE
from uptrend_detector import detect_uptrend

def create_balanced_data(X, y, strategy='aggressive'):
    """Create balanced training data - EXACT copy from rouf_baseline"""
    if strategy == 'aggressive':
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
    elif strategy == 'moderate':
        smote = SMOTE(sampling_strategy=0.3, random_state=42)
    else:
        smote = SMOTE(sampling_strategy=0.15, random_state=42)
    
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"Original distribution: {np.bincount(y)}")
    print(f"Balanced distribution: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced

def check_future_falls(returns, n, d, w, cumulative=True):
    """EXACT copy from rouf_baseline"""
    rolling_std = returns.rolling(window=w, min_periods=1).std()
    result = pd.Series(0, index=returns.index)
    
    for i in range(len(returns) - d):
        current_std = rolling_std.iloc[i]
        
        if pd.isna(current_std) or current_std == 0:
            continue
        
        threshold = -n * current_std
        future_returns = returns.iloc[i+1:i+d+1]
        
        if cumulative:
            for j in range(1, len(future_returns) + 1):
                cum_return = (1 + future_returns.iloc[:j]).prod() - 1
                if cum_return <= threshold:
                    result.iloc[i] = 1
                    break
        else:
            if (future_returns <= threshold).any():
                result.iloc[i] = 1
    
    return result

def apply_cooldown(predictions, cooldown_days=30):
    """EXACT copy from rouf_baseline logic"""
    result_exit = []
    counter = 0
    
    for i in range(len(predictions)):
        if predictions.iloc[i] == 1 and counter == 0:
            counter = cooldown_days
            result_exit.append(1)
        elif counter != 0:
            counter -= 1
            result_exit.append(0)
        elif counter == 0:
            result_exit.append(1)  # Stay in market when no signal and no cooldown
    
    return pd.Series(result_exit, index=predictions.index)

def run_minimal_prediction():
    """Run prediction exactly like rouf_baseline"""
    print("Running minimal prediction...")
    
    # Download and prepare data (simplified)
    btc_data = yf.download('BTC-USD', start='2023-01-01')
    if isinstance(btc_data.columns, pd.MultiIndex):
        btc_data.columns = [col[0] for col in btc_data.columns]
    
    # Create basic features (simplified set)
    df = pd.DataFrame(index=btc_data.index)
    df['close'] = btc_data['Close']
    df['volume'] = np.log(btc_data['Volume'])
    df['return'] = btc_data['Close'].pct_change()
    df['vol_7d'] = df['return'].rolling(7).std() * np.sqrt(252)
    df['vol_30d'] = df['return'].rolling(30).std() * np.sqrt(252)
    df['vol_60d'] = df['return'].rolling(60).std() * np.sqrt(252)
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['ma_30d'] = df['close'].rolling(30).mean()
    df['drawdown'] = (df['close'] / df['close'].cummax()) - 1
    
    # Drop NaN
    df = df.dropna()
    
    # Create target
    target = check_future_falls(df['return'], 6, 60, 60)
    
    # Feature selection - EXACT like rouf_baseline
    feature_cols = ['volume', 'return', 'vol_7d', 'vol_30d', 'vol_60d', 'rsi_14', 'ma_30d', 'drawdown']
    available_features = [f for f in feature_cols if f in df.columns]
    
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(8, len(available_features)))
    X_selected = selector_mi.fit_transform(df[available_features], target)
    
    # Split data - EXACT like rouf_baseline
    split_idx = int(0.7 * len(target))
    X_train = X_selected[:split_idx]
    X_test = X_selected[split_idx:]
    y_train = target.iloc[:split_idx]
    y_test = target.iloc[split_idx:]
    
    X_train_orig = X_train.copy()
    y_train_orig = y_train.copy()
    
    # Balance data - EXACT like rouf_baseline
    X_train, y_train = create_balanced_data(X_train, y_train, strategy='aggressive')
    
    # Train model - EXACT like rouf_baseline
    tscv = TimeSeriesSplit(n_splits=10)
    lr_param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr_grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid=lr_param_grid,
        cv=tscv,
        scoring='precision_weighted',
        n_jobs=-1
    )
    lr_grid_search.fit(X_train, y_train)
    lr_model = lr_grid_search.best_estimator_
    
    # Make predictions - EXACT like rouf_baseline
    y_train_pred = lr_model.predict(X_train_orig)
    y_test_pred = lr_model.predict(X_test)
    
    # Combine predictions
    full_predictions = np.concatenate([y_train_pred, y_test_pred])
    full_predictions_series = pd.Series(full_predictions, index=df.index)
    
    # Apply cooldown - EXACT like rouf_baseline
    binary_exits = apply_cooldown(full_predictions_series, cooldown_days=30)
    
    # Apply uptrend filter
    uptrend = detect_uptrend(df['close'])
    final_exits = binary_exits * uptrend.shift(1)
    
    return {
        'predictions_sum': full_predictions_series.sum(),
        'binary_exits_sum': binary_exits.sum(),
        'final_exits_sum': final_exits.sum(),
        'first_10_predictions': full_predictions[:10].tolist()
    }

def calculate_rsi(data, period):
    """Calculate RSI"""
    delta = data.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

def test_minimal_consistency(n_runs=3):
    """Test consistency of minimal predictor"""
    print("Testing minimal predictor consistency...")
    
    results = []
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")
        
        # Reset random seed before each run
        np.random.seed(42)
        
        result = run_minimal_prediction()
        results.append(result)
        
        print(f"Predictions sum: {result['predictions_sum']}")
        print(f"First 10: {result['first_10_predictions']}")
    
    # Check consistency
    consistent = True
    for i in range(1, n_runs):
        if results[i]['first_10_predictions'] != results[0]['first_10_predictions']:
            print(f"❌ Run {i+1} differs from run 1!")
            consistent = False
            print(f"Run 1: {results[0]['first_10_predictions']}")
            print(f"Run {i+1}: {results[i]['first_10_predictions']}")
    
    if consistent:
        print("✅ Minimal predictor is consistent!")
    else:
        print("❌ Minimal predictor is not consistent!")
    
    return results, consistent

if __name__ == "__main__":
    test_minimal_consistency()