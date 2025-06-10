import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from uptrend_detector import detect_uptrend


# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


# Download Bitcoin data
btc_data = yf.download('BTC-USD', start='2018-01-01')

# Fix column names if multi-level
if isinstance(btc_data.columns, pd.MultiIndex):
    btc_data.columns = [col[0] for col in btc_data.columns]

# Initialize master DataFrame
master_df = pd.DataFrame(index=btc_data.index)

# Add Bitcoin price data
master_df['btc_open'] = btc_data['Open']
master_df['btc_high'] = btc_data['High']
master_df['btc_low'] = btc_data['Low']
master_df['btc_close'] = btc_data['Close']
master_df['btc_volume'] = np.log(btc_data['Volume'])

# Calculate returns
master_df['btc_return'] = btc_data['Close'].pct_change()
master_df['btc_log_return'] = np.log(btc_data['Close'] / btc_data['Close'].shift(1))

# Calculate rolling volatility (annualized)
master_df['rolling_vol_7d'] = master_df['btc_return'].rolling(7).std() * np.sqrt(252)
master_df['rolling_vol_30d'] = master_df['btc_return'].rolling(30).std() * np.sqrt(252)
master_df['rolling_vol_60d'] = master_df['btc_return'].rolling(60).std() * np.sqrt(252)

# Calculate rolling skewness
master_df['rolling_skew_7d'] = master_df['btc_return'].rolling(7).skew()
master_df['rolling_skew_30d'] = master_df['btc_return'].rolling(30).skew()
master_df['rolling_skew_60d'] = master_df['btc_return'].rolling(60).skew()

# Calculate rolling kurtosis
master_df['rolling_kurt_30d'] = master_df['btc_return'].rolling(30).kurt()

print(f"Master DataFrame shape: {master_df.shape}")


# Calculate RSI (Relative Strength Index) for 14 and 30 periods
def calculate_rsi(data, period):
    """Calculate RSI for a given period"""
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Add RSI indicators
master_df['rsi_7'] = calculate_rsi(master_df['btc_close'], 7)
master_df['rsi_14'] = calculate_rsi(master_df['btc_close'], 14)
master_df['rsi_30'] = calculate_rsi(master_df['btc_close'], 30)
master_df['rsi_60'] = calculate_rsi(master_df['btc_close'], 60)


print(f"RSI indicators added. Current shape: {master_df.shape}")


# Calculate ADX (Average Directional Index)
def calculate_adx(high, low, close, period=14):
    """Calculate ADX indicator"""
    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate directional movements
    dm_plus = high - high.shift(1)
    dm_minus = low.shift(1) - low
    
    # Filter directional movements
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Calculate smoothed TR and DM
    tr_smooth = tr.rolling(window=period).mean()
    dm_plus_smooth = dm_plus.rolling(window=period).mean()
    dm_minus_smooth = dm_minus.rolling(window=period).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)
    
    # Calculate DX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # Calculate ADX
    adx = dx.rolling(window=period).mean()
    
    return adx

# Since we only have Close prices, we'll approximate High/Low from Close
# This is a simplification - ideally we'd have OHLC data
high = master_df['btc_high']   # Approximate high as close + 0.1%
low = master_df['btc_low']   # Approximate low as close - 0.1%

# Add ADX indicator
master_df['adx_7'] = calculate_adx(high, low, master_df['btc_close'], period=7)
master_df['adx_14'] = calculate_adx(high, low, master_df['btc_close'], period=14)
master_df['adx_30'] = calculate_adx(high, low, master_df['btc_close'], period=30)
master_df['adx_60'] = calculate_adx(high, low, master_df['btc_close'], period=60)



print(f"ADX indicator added. Current shape: {master_df.shape}")
print(f"Master DataFrame now has {master_df.shape[1]} columns")

# Price-based metrics
master_df['btc_cumulative_return'] = (1 + master_df['btc_return']).cumprod() - 1
master_df['btc_drawdown'] = (master_df['btc_close'] / master_df['btc_close'].cummax()) - 1

# Moving averages
master_df['btc_ma_7d'] = master_df['btc_close'].rolling(7).mean()
master_df['btc_ma_30d'] = master_df['btc_close'].rolling(30).mean()
master_df['btc_ma_60d'] = master_df['btc_close'].rolling(60).mean()
master_df['btc_ma_200d'] = master_df['btc_close'].rolling(200).mean()

# Volatility metrics
master_df['vol_change_7d'] = master_df['rolling_vol_7d'].pct_change(7)
master_df['vol_change_30d'] = master_df['rolling_vol_7d'].pct_change(7)
master_df['vol_change_60d'] = master_df['rolling_vol_7d'].pct_change(7)

master_df['vol_percentile_7d'] = master_df['rolling_vol_30d'].rolling(30).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
)

master_df['vol_percentile_30d'] = master_df['rolling_vol_30d'].rolling(30).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
)

master_df['vol_percentile_60d'] = master_df['rolling_vol_30d'].rolling(30).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
)


'''# SKEW metrics (if available)
if 'skew_index' in master_df.columns:
    master_df['skew_ma_7d'] = master_df['skew_index'].rolling(7).mean()
    master_df['skew_ma_30d'] = master_df['skew_index'].rolling(30).mean()
    master_df['skew_percentile_30d'] = master_df['skew_index'].rolling(30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
    )
    
    # Regime indicators
    master_df['high_skew_regime'] = (master_df['skew_index'] > 130).astype(int)
    master_df['extreme_skew_regime'] = (master_df['skew_index'] > 140).astype(int)'''

# Volatility regime
median_vol = master_df['rolling_vol_30d'].median()
master_df['high_vol_regime'] = (master_df['rolling_vol_30d'] > median_vol).astype(int)

# Market trend
master_df['trend_7d'] = np.where(master_df['btc_close'] > master_df['btc_ma_7d'], 1, -1)
master_df['trend_30d'] = np.where(master_df['btc_close'] > master_df['btc_ma_30d'], 1, -1)
master_df['trend_60d'] = np.where(master_df['btc_close'] > master_df['btc_ma_60d'], 1, -1)


from fracdiff import frac_diff_rolling_threshold

frac_series, config = frac_diff_rolling_threshold(master_df.btc_close,target_stationarity=0.95)

master_df['btc_frac_diff'] = frac_series['series']

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def add_market_structure_features(df, horizons=[7, 30, 60]):
    """Add market structure and regime features for multiple horizons"""
    print("Adding market structure features...")
    
    for horizon in horizons:
        # Support/Resistance levels
        df[f'support_break_{horizon}d'] = (
            df['btc_close'] < df['btc_close'].rolling(horizon).min().shift(1)
        ).astype(int)
        
        df[f'resistance_break_{horizon}d'] = (
            df['btc_close'] > df['btc_close'].rolling(horizon).max().shift(1)
        ).astype(int)
        
        # Price vs MA relationships  
        if f'btc_ma_{horizon}d' in df.columns:
            df[f'price_above_ma_{horizon}d'] = (df['btc_close'] > df[f'btc_ma_{horizon}d']).astype(int)
            df[f'ma_slope_{horizon}d'] = df[f'btc_ma_{horizon}d'].diff(horizon)
    
    # Cross-MA relationships
    if 'btc_ma_7d' in df.columns and 'btc_ma_30d' in df.columns:
        df['ma7_above_ma30'] = (df['btc_ma_7d'] > df['btc_ma_30d']).astype(int)
    if 'btc_ma_30d' in df.columns and 'btc_ma_60d' in df.columns:
        df['ma30_above_ma60'] = (df['btc_ma_30d'] > df['btc_ma_60d']).astype(int)
    
    # Sequential patterns
    df['lower_lows_sequence_3d'] = (
        (df['btc_close'] < df['btc_close'].shift(1)) & 
        (df['btc_close'].shift(1) < df['btc_close'].shift(2)) &
        (df['btc_close'].shift(2) < df['btc_close'].shift(3))
    ).astype(int)
    
    # Volatility regime changes
    if 'high_vol_regime' in df.columns:
        df['vol_regime_change'] = (df['high_vol_regime'] != df['high_vol_regime'].shift(1)).astype(int)
    
    return df

def add_momentum_features(df, horizons=[7, 14, 30, 60]):
    """Add momentum and divergence features"""
    print("Adding momentum features...")
    
    # MACD family
    ema_12 = df['btc_close'].ewm(span=12).mean()
    ema_26 = df['btc_close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_cross_signal'] = (
        (df['macd'] > df['macd_signal']) & 
        (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    ).astype(int)
    
    # Stochastic RSI
    if 'rsi_14' in df.columns:
        rsi = df['rsi_14']
        df['stoch_rsi'] = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        df['stoch_rsi'] = df['stoch_rsi'].fillna(0.5)
    
    # Williams %R for multiple horizons
    for horizon in horizons:
        highest_high = df['btc_close'].rolling(horizon).max()
        lowest_low = df['btc_close'].rolling(horizon).min()
        df[f'williams_r_{horizon}d'] = -100 * (highest_high - df['btc_close']) / (highest_high - lowest_low)
        df[f'williams_r_{horizon}d'] = df[f'williams_r_{horizon}d'].fillna(-50)
    
    # Rate of Change (ROC) for multiple horizons
    for horizon in horizons:
        df[f'roc_{horizon}d'] = ((df['btc_close'] / df['btc_close'].shift(horizon)) - 1) * 100
        df[f'roc_{horizon}d'] = df[f'roc_{horizon}d'].fillna(0)
    
    # Momentum divergence (price vs RSI)
    if 'rsi_14' in df.columns:
        for horizon in [7, 14, 30]:
            df[f'rsi_divergence_{horizon}d'] = (
                (df['btc_close'] > df['btc_close'].shift(horizon)) & 
                (df['rsi_14'] < df['rsi_14'].shift(horizon))
            ).astype(int)
    
    # Triple momentum confirmation
    df['momentum_confluence'] = (
        (df.get('roc_7d', 0) > 0) & 
        (df.get('roc_14d', 0) > 0) & 
        (df.get('roc_30d', 0) > 0)
    ).astype(int)
    
    return df

def add_volume_features(df, horizons=[7, 14, 30, 60]):
    """Add comprehensive volume analysis features"""
    print("Adding volume features...")
    
    # Volume moving averages for multiple horizons
    for horizon in horizons:
        df[f'volume_ma_{horizon}d'] = df['btc_volume'].rolling(horizon).mean()
        df[f'volume_ratio_current_{horizon}d'] = df['btc_volume'] / df[f'volume_ma_{horizon}d']
    
    # Volume ratios between different periods
    df['volume_ratio_7_30'] = df['volume_ma_7d'] / df['volume_ma_30d']
    df['volume_ratio_7_60'] = df['volume_ma_7d'] / df['volume_ma_60d']
    df['volume_ratio_30_60'] = df['volume_ma_30d'] / df['volume_ma_60d']
    
    # Volume spikes for different thresholds
    for multiplier in [1.5, 2.0, 3.0]:
        df[f'volume_spike_{multiplier}x'] = (df['btc_volume'] > df['volume_ma_30d'] * multiplier).astype(int)
    
    # On-Balance Volume (OBV)
    obv = (df['btc_volume'] * np.sign(df['btc_return'])).cumsum()
    df['obv'] = obv
    for horizon in [7, 14, 30]:
        df[f'obv_ma_{horizon}d'] = obv.rolling(horizon).mean()
        df[f'obv_slope_{horizon}d'] = df[f'obv_ma_{horizon}d'].diff(horizon)
    
    # Volume-Price Trend (VPT)
    vpt = (df['btc_volume'] * df['btc_return']).cumsum()
    df['vpt'] = vpt
    df['vpt_ma_14d'] = vpt.rolling(14).mean()
    df['vpt_slope_14d'] = df['vpt_ma_14d'].diff(14)
    
    # Accumulation/Distribution Line
    # Simplified version using daily high/low approximation
    daily_range = df['btc_close'].rolling(1).max() - df['btc_close'].rolling(1).min()
    # Fix the replace operation - use where instead
    daily_range = daily_range.where(daily_range != 0, df['btc_close'] * 0.001)
    
    clv = ((df['btc_close'] - df['btc_close'].rolling(1).min()) - 
           (df['btc_close'].rolling(1).max() - df['btc_close'])) / daily_range
    clv = clv.fillna(0)
    df['ad_line'] = (clv * df['btc_volume']).cumsum()
    
    # Volume patterns
    df['volume_breakdown'] = (
        (df['btc_volume'] > df['volume_ma_30d'] * 1.5) & 
        (df['btc_return'] < -0.02)
    ).astype(int)
    
    df['volume_thrust'] = (
        (df['btc_volume'] > df['volume_ma_30d'] * 1.5) & 
        (df['btc_return'] > 0.02)
    ).astype(int)
    
    # Volume trend strength
    for horizon in [7, 14, 30]:
        df[f'volume_trend_{horizon}d'] = df['btc_volume'].rolling(horizon).apply(
            lambda x: stats.pearsonr(range(len(x)), x)[0] if len(x) == horizon else 0
        )
    
    return df

def add_volatility_structure_features(df, horizons=[7, 14, 30, 60]):
    """Add advanced volatility structure features"""
    print("Adding volatility structure features...")
    
    # Volatility persistence and clustering
    for horizon in horizons:
        vol_col = f'rolling_vol_{horizon}d'
        if vol_col in df.columns:
            # Volatility of volatility
            df[f'volvol_{horizon}d'] = df[vol_col].rolling(horizon).std()
            
            # Volatility persistence
            vol_mean = df[vol_col].rolling(horizon).mean()
            vol_std = df[vol_col].rolling(horizon).std()
            df[f'vol_persistence_{horizon}d'] = vol_std / vol_mean
            df[f'vol_persistence_{horizon}d'] = df[f'vol_persistence_{horizon}d'].fillna(0)
            
            # Volatility momentum
            df[f'vol_momentum_{horizon}d'] = df[vol_col] / df[vol_col].shift(horizon)
            df[f'vol_momentum_{horizon}d'] = df[f'vol_momentum_{horizon}d'].fillna(1)
    
    # Volatility term structure
    if 'rolling_vol_7d' in df.columns and 'rolling_vol_60d' in df.columns:
        df['vol_term_structure_7_60'] = df['rolling_vol_7d'] / df['rolling_vol_60d']
    if 'rolling_vol_7d' in df.columns and 'rolling_vol_30d' in df.columns:
        df['vol_term_structure_7_30'] = df['rolling_vol_7d'] / df['rolling_vol_30d']
    if 'rolling_vol_30d' in df.columns and 'rolling_vol_60d' in df.columns:
        df['vol_term_structure_30_60'] = df['rolling_vol_30d'] / df['rolling_vol_60d']
    
    # Jump detection for different thresholds
    for threshold in [2, 3, 4]:
        if 'rolling_vol_7d' in df.columns:
            df[f'jump_indicator_{threshold}sigma'] = (
                np.abs(df['btc_return']) > threshold * df['rolling_vol_7d']
            ).astype(int)
    
    # Volatility regime transitions
    if 'rolling_vol_30d' in df.columns:
        vol_30d = df['rolling_vol_30d']
        vol_percentiles = vol_30d.rolling(252).rank(pct=True)
        df['vol_regime_low'] = (vol_percentiles < 0.33).astype(int)
        df['vol_regime_medium'] = ((vol_percentiles >= 0.33) & (vol_percentiles < 0.67)).astype(int)
        df['vol_regime_high'] = (vol_percentiles >= 0.67).astype(int)
    
    return df

def add_statistical_features(df, horizons=[7, 14, 30, 60]):
    """Add statistical anomaly and distribution features"""
    print("Adding statistical features...")
    
    # Z-scores for returns
    for horizon in horizons:
        rolling_mean = df['btc_return'].rolling(horizon).mean()
        rolling_std = df['btc_return'].rolling(horizon).std()
        df[f'return_zscore_{horizon}d'] = (df['btc_return'] - rolling_mean) / rolling_std
        df[f'return_zscore_{horizon}d'] = df[f'return_zscore_{horizon}d'].fillna(0)
    
    # Percentile rankings
    for horizon in [30, 60, 252]:
        df[f'return_percentile_{horizon}d'] = df['btc_return'].rolling(horizon).rank(pct=True)
        df[f'price_percentile_{horizon}d'] = df['btc_close'].rolling(horizon).rank(pct=True)
        if 'btc_volume' in df.columns:
            df[f'volume_percentile_{horizon}d'] = df['btc_volume'].rolling(horizon).rank(pct=True)
    
    # Consecutive patterns
    negative_returns = (df['btc_return'] < 0).astype(int)
    positive_returns = (df['btc_return'] > 0).astype(int)
    
    # Consecutive negative returns
    neg_groups = (negative_returns != negative_returns.shift()).cumsum()
    df['consecutive_negative'] = negative_returns.groupby(neg_groups).cumsum() * negative_returns
    
    # Consecutive positive returns
    pos_groups = (positive_returns != positive_returns.shift()).cumsum()
    df['consecutive_positive'] = positive_returns.groupby(pos_groups).cumsum() * positive_returns
    
    # Tail risk measures
    for horizon in [14, 30, 60]:
        df[f'var_5pct_{horizon}d'] = df['btc_return'].rolling(horizon).quantile(0.05)
        df[f'var_95pct_{horizon}d'] = df['btc_return'].rolling(horizon).quantile(0.95)
        
        # Conditional VaR (Expected Shortfall)
        df[f'cvar_5pct_{horizon}d'] = df['btc_return'].rolling(horizon).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else x.min()
        )
    
    # Maximum adverse excursion
    for horizon in [7, 14, 30]:
        df[f'max_adverse_move_{horizon}d'] = df['btc_return'].rolling(horizon).min()
        df[f'max_favorable_move_{horizon}d'] = df['btc_return'].rolling(horizon).max()
    
    # Skewness and Kurtosis for additional horizons
    for horizon in [14, 21]:
        if f'rolling_skew_{horizon}d' not in df.columns:
            df[f'rolling_skew_{horizon}d'] = df['btc_return'].rolling(horizon).skew()
        if f'rolling_kurt_{horizon}d' not in df.columns:
            df[f'rolling_kurt_{horizon}d'] = df['btc_return'].rolling(horizon).kurt()
    
    return df

def add_cross_asset_features(df):
    """Add time-based and seasonal features"""
    print("Adding cross-asset and temporal features...")
    
    # Create datetime index if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Day of week effects
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Month effects
    df['month'] = df.index.month
    df['is_january'] = (df['month'] == 1).astype(int)
    df['is_december'] = (df['month'] == 12).astype(int)
    df['quarter'] = df.index.quarter
    
    # Year effects (for longer datasets)
    df['year'] = df.index.year
    
    # Holiday effects (simplified - major US holidays)
    # You can expand this based on your specific needs
    df['is_year_end'] = (df.index.month == 12).astype(int)
    df['is_year_start'] = (df.index.month == 1).astype(int)
    
    return df

def add_microstructure_features(df, horizons=[7, 14, 30]):
    """Add market microstructure and behavioral features"""
    print("Adding microstructure features...")
    
    # Price gaps
    df['price_gap'] = df['btc_close'] / df['btc_close'].shift(1) - 1
    df['price_gap'] = df['price_gap'].fillna(0)
    
    for threshold in [0.01, 0.02, 0.03, 0.05]:
        df[f'large_gap_{int(threshold*100)}pct'] = (np.abs(df['price_gap']) > threshold).astype(int)
    
    # Price clustering around round numbers
    df['round_number_1000'] = np.minimum(
        df['btc_close'] % 1000, 1000 - (df['btc_close'] % 1000)
    ) / 1000
    
    df['round_number_100'] = np.minimum(
        df['btc_close'] % 100, 100 - (df['btc_close'] % 100)
    ) / 100
    
    # Momentum exhaustion signals
    if 'rsi_14' in df.columns:
        df['rsi_overbought_negative'] = ((df['rsi_14'] > 70) & (df['btc_return'] < 0)).astype(int)
        df['rsi_oversold_positive'] = ((df['rsi_14'] < 30) & (df['btc_return'] > 0)).astype(int)
    
    # Price acceleration
    df['price_acceleration'] = df['btc_return'].diff()
    for horizon in horizons:
        df[f'acceleration_ma_{horizon}d'] = df['price_acceleration'].rolling(horizon).mean()
    
    # Trend exhaustion
    for horizon in [7, 14, 30]:
        trend_col = f'trend_{horizon}d'
        if trend_col in df.columns:
            df[f'trend_exhaustion_{horizon}d'] = (
                (df[trend_col] > 0.01) & (df['btc_return'] < -0.02)
            ).astype(int)
    
    return df

def add_regime_features(df, horizons=[30, 60]):
    """Add regime change detection features"""
    print("Adding regime features...")
    
    returns = df['btc_return']
    
    # Simple regime classification based on volatility
    for horizon in horizons:
        vol_col = f'rolling_vol_{horizon}d'
        if vol_col in df.columns:
            vol = df[vol_col]
            vol_rolling_quantile = vol.rolling(252).quantile(0.67)
            
            df[f'high_vol_regime_{horizon}d'] = (vol > vol_rolling_quantile).astype(int)
            
            # Regime persistence
            regime_col = f'high_vol_regime_{horizon}d'
            regime_changes = (df[regime_col] != df[regime_col].shift()).cumsum()
            df[f'regime_duration_{horizon}d'] = df[regime_col].groupby(regime_changes).cumsum()
    
    # Structural breaks using rolling correlation
    for horizon in [30, 60]:
        window = horizon * 2
        df[f'structural_break_proxy_{horizon}d'] = returns.rolling(window).apply(
            lambda x: np.abs(np.corrcoef(x[:horizon], x[horizon:])[0,1]) if len(x) == window else 0
        )
    
    # Trend regime classification
    for horizon in [7, 30, 60]:
        trend_col = f'trend_{horizon}d'
        if trend_col in df.columns:
            # Calculate quantiles separately
            trend_q33 = df[trend_col].rolling(252).quantile(0.33)
            trend_q67 = df[trend_col].rolling(252).quantile(0.67)
            df[f'downtrend_regime_{horizon}d'] = (df[trend_col] < trend_q33).astype(int)
            df[f'uptrend_regime_{horizon}d'] = (df[trend_col] > trend_q67).astype(int)
    
    return df

def add_interaction_features(df):
    """Add feature interactions that are particularly relevant for fall prediction"""
    print("Adding interaction features...")
    
    # Volume-Volatility interactions
    if 'btc_volume' in df.columns and 'rolling_vol_7d' in df.columns:
        df['volume_vol_interaction'] = df['btc_volume'] * df['rolling_vol_7d']
        df['volume_vol_ratio'] = df['btc_volume'] / (df['rolling_vol_7d'] + 1e-8)
    
    # RSI-Trend interactions
    if 'rsi_14' in df.columns and 'trend_30d' in df.columns:
        df['rsi_trend_interaction'] = df['rsi_14'] * df['trend_30d']
        df['rsi_downtrend_stress'] = df['rsi_14'] * (df['trend_30d'] < 0).astype(int)
    
    # Drawdown-Volume interaction
    if 'btc_drawdown' in df.columns and 'btc_volume' in df.columns:
        df['drawdown_volume_interaction'] = df['btc_drawdown'] * df['btc_volume']
    
    # Multi-timeframe momentum alignment
    roc_cols = [col for col in df.columns if col.startswith('roc_')]
    if len(roc_cols) >= 3:
        df['momentum_alignment'] = sum([
            (df[col] > 0).astype(int) for col in roc_cols[:3]
        ])
        df['momentum_divergence_count'] = sum([
            (df[col] < 0).astype(int) for col in roc_cols[:3]
        ])
    
    return df

def build_comprehensive_features(master_df):
    """
    Main function to add all features to master_df
    
    Parameters:
    master_df: DataFrame with basic features already computed
    
    Returns:
    DataFrame with all additional features
    """
    print("Starting comprehensive feature engineering...")
    print(f"Initial shape: {master_df.shape}")
    
    # Make a copy to avoid modifying original
    df = master_df.copy()
    
    # Ensure we have required base columns
    required_cols = ['btc_close', 'btc_return', 'btc_volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        # Add each feature group
        df = add_market_structure_features(df)
        df = add_momentum_features(df)
        df = add_volume_features(df)
        df = add_volatility_structure_features(df)
        df = add_statistical_features(df)
        df = add_cross_asset_features(df)
        df = add_microstructure_features(df)
        df = add_regime_features(df)
        df = add_interaction_features(df)
        
        print(f"Final shape: {df.shape}")
        print(f"Added {df.shape[1] - master_df.shape[1]} new features")
        
        # Display new feature summary
        new_features = [col for col in df.columns if col not in master_df.columns]
        print(f"\nNew features added: {len(new_features)}")
        
        # Group by feature type
        feature_groups = {
            'Market Structure': [f for f in new_features if any(x in f for x in ['support', 'resistance', 'ma_', 'lower_lows'])],
            'Momentum': [f for f in new_features if any(x in f for x in ['macd', 'roc_', 'williams', 'stoch', 'divergence'])],
            'Volume': [f for f in new_features if any(x in f for x in ['volume', 'obv', 'vpt', 'ad_line'])],
            'Volatility': [f for f in new_features if any(x in f for x in ['volvol', 'vol_', 'jump'])],
            'Statistical': [f for f in new_features if any(x in f for x in ['zscore', 'percentile', 'var_', 'cvar', 'consecutive'])],
            'Temporal': [f for f in new_features if any(x in f for x in ['day_', 'month', 'is_', 'quarter'])],
            'Microstructure': [f for f in new_features if any(x in f for x in ['gap', 'round_number', 'acceleration'])],
            'Regime': [f for f in new_features if any(x in f for x in ['regime', 'structural', 'trend_regime'])],
            'Interactions': [f for f in new_features if 'interaction' in f or 'alignment' in f]
        }
        
        for group, features in feature_groups.items():
            if features:
                print(f"\n{group} ({len(features)} features):")
                print(f"  {', '.join(features[:5])}" + (f" ... and {len(features)-5} more" if len(features) > 5 else ""))
        
        return df
        
    except Exception as e:
        print(f"Error during feature engineering: {str(e)}")
        print("Returning original DataFrame")
        return master_df

# Usage example:
enhanced_master_df = build_comprehensive_features(master_df)


"""
Horizon-Specific Feature Lists for Bitcoin Fall Prediction
=========================================================

Strategy:
- 7-day model: Focus on short-term momentum, microstructure, and immediate volatility
- 30-day model: Balance between short and medium-term indicators  
- 60-day model: Emphasize longer-term trends, structural breaks, and regime changes

Key Principle: Avoid using features with lookback periods >= prediction horizon
"""

# =============================================================================
# 7-DAY MODEL FEATURES (Short-term focus)
# =============================================================================

features_7d = [
    # Core price data
    'btc_volume',
    'btc_return',
    'btc_log_return',
    'btc_drawdown',
    'btc_frac_diff',
    
    # Short-term volatility (avoid 7d+ lookbacks)
    'rolling_vol_7d',  # Keep as it's exactly our horizon
    'vol_change_7d',
    'vol_percentile_7d',
    'volvol_7d',
    'vol_persistence_7d',
    'vol_momentum_7d',
    
    # Short-term momentum indicators
    'rsi_7',  # Most responsive RSI
    'rsi_14',  # Standard RSI
    'adx_7',   # Short-term trend strength
    'roc_7d',  # Rate of change matching horizon
    'roc_14d', # Slightly longer ROC
    
    # Technical patterns (short-term)
    'macd',
    'macd_signal', 
    'macd_histogram',
    'macd_cross_signal',
    'stoch_rsi',
    'williams_r_7d',
    'williams_r_14d',
    
    # Market structure (immediate)
    'support_break_7d',
    'resistance_break_7d',
    'price_above_ma_7d',
    'ma_slope_7d',
    'lower_lows_sequence_3d',
    'ma7_above_ma30',  # Cross-timeframe signal
    
    # Volume patterns (short-term focus)
    'volume_ma_7d',
    'volume_ratio_current_7d',
    'volume_ratio_7_30',  # Ratio gives context without leakage
    'volume_spike_1.5x',
    'volume_spike_2.0x',
    'volume_spike_3.0x',
    'volume_breakdown',
    'volume_thrust',
    'volume_trend_7d',
    'obv_slope_7d',
    'vpt_slope_14d',
    
    # Statistical measures (short windows)
    'return_zscore_7d',
    'return_zscore_14d',
    'consecutive_negative',
    'consecutive_positive',
    'max_adverse_move_7d',
    'max_favorable_move_7d',
    'rolling_skew_7d',
    
    # Jump and anomaly detection
    'jump_indicator_2sigma',
    'jump_indicator_3sigma',
    'jump_indicator_4sigma',
    'vol_regime_change',
    
    # Microstructure (critical for short-term)
    'price_gap',
    'large_gap_1pct',
    'large_gap_2pct',
    'large_gap_3pct',
    'large_gap_5pct',
    'round_number_1000',
    'round_number_100',
    'price_acceleration',
    'acceleration_ma_7d',
    'acceleration_ma_14d',
    
    # Behavioral indicators
    'rsi_overbought_negative',
    'rsi_oversold_positive',
    'rsi_divergence_7d',
    'trend_exhaustion_7d',
    
    # Temporal features
    'day_of_week',
    'is_weekend',
    'is_monday',
    'is_friday',
    
    # Regime indicators (short-term)
    'vol_regime_low',
    'vol_regime_medium', 
    'vol_regime_high',
    'downtrend_regime_7d',
    'uptrend_regime_7d',
    
    # Interaction features
    'volume_vol_interaction',
    'volume_vol_ratio',
    'rsi_downtrend_stress',
    'momentum_alignment',
    'momentum_divergence_count'
]

# =============================================================================
# 30-DAY MODEL FEATURES (Medium-term focus)
# =============================================================================

features_30d = [
    # Core price data
    'btc_volume',
    'btc_return', 
    'btc_log_return',
    'btc_cumulative_return',
    'btc_drawdown',
    'btc_frac_diff',
    
    # Multi-timeframe volatility (up to 30d)
    'rolling_vol_7d',
    'rolling_vol_30d',
    'vol_change_7d',
    'vol_change_30d',
    'vol_percentile_7d',
    'vol_percentile_30d',
    'vol_term_structure_7_30',
    'volvol_7d',
    'volvol_30d',
    'vol_persistence_7d',
    'vol_persistence_30d',
    'vol_momentum_7d',
    'vol_momentum_30d',
    
    # Momentum indicators (mixed timeframes)
    'rsi_7',
    'rsi_14',
    'rsi_30',  # Add longer RSI for context
    'adx_7',
    'adx_14',
    'adx_30',
    'roc_7d',
    'roc_14d',
    'roc_30d',
    
    # Technical patterns
    'macd',
    'macd_signal',
    'macd_histogram', 
    'macd_cross_signal',
    'stoch_rsi',
    'williams_r_7d',
    'williams_r_14d',
    'williams_r_30d',
    
    # Market structure (multiple timeframes)
    'support_break_7d',
    'resistance_break_7d',
    'support_break_30d',
    'resistance_break_30d',
    'price_above_ma_7d',
    'price_above_ma_30d',
    'ma_slope_7d',
    'ma_slope_30d',
    'lower_lows_sequence_3d',
    'ma7_above_ma30',
    'ma30_above_ma60',  # Add longer-term context
    
    # Trend analysis
    'trend_7d',
    'trend_30d',
    
    # Volume analysis (medium-term)
    'volume_ma_7d',
    'volume_ma_14d',
    'volume_ma_30d',
    'volume_ratio_current_7d',
    'volume_ratio_current_14d',
    'volume_ratio_current_30d',
    'volume_ratio_7_30',
    'volume_spike_1.5x',
    'volume_spike_2.0x',
    'volume_spike_3.0x',
    'volume_breakdown',
    'volume_thrust',
    'volume_trend_7d',
    'volume_trend_14d',
    'volume_trend_30d',
    'obv',
    'obv_ma_7d',
    'obv_ma_14d',
    'obv_ma_30d',
    'obv_slope_7d',
    'obv_slope_14d',
    'obv_slope_30d',
    'vpt',
    'vpt_ma_14d',
    'vpt_slope_14d',
    'ad_line',
    
    # Statistical measures (medium windows)
    'return_zscore_7d',
    'return_zscore_14d',
    'return_zscore_30d',
    'return_percentile_30d',
    'price_percentile_30d',
    'volume_percentile_30d',
    'consecutive_negative',
    'consecutive_positive',
    'var_5pct_14d',
    'var_95pct_14d',
    'cvar_5pct_14d',
    'var_5pct_30d',
    'var_95pct_30d',
    'cvar_5pct_30d',
    'max_adverse_move_7d',
    'max_favorable_move_7d',
    'max_adverse_move_14d',
    'max_favorable_move_14d',
    'max_adverse_move_30d',
    'max_favorable_move_30d',
    'rolling_skew_7d',
    'rolling_skew_14d',
    'rolling_skew_30d',
    'rolling_kurt_14d',
    'rolling_kurt_30d',
    
    # Jump detection
    'jump_indicator_2sigma',
    'jump_indicator_3sigma', 
    'jump_indicator_4sigma',
    
    # Microstructure
    'price_gap',
    'large_gap_1pct',
    'large_gap_2pct',
    'large_gap_3pct',
    'large_gap_5pct',
    'round_number_1000',
    'round_number_100',
    'price_acceleration',
    'acceleration_ma_7d',
    'acceleration_ma_14d',
    'acceleration_ma_30d',
    
    # Behavioral and divergence
    'rsi_overbought_negative',
    'rsi_oversold_positive',
    'rsi_divergence_7d',
    'rsi_divergence_14d',
    'rsi_divergence_30d',
    'trend_exhaustion_7d',
    'trend_exhaustion_30d',
    'momentum_confluence',
    
    # Temporal features
    'day_of_week',
    'is_weekend',
    'is_monday', 
    'is_friday',
    'month',
    'is_january',
    'is_december',
    'quarter',
    
    # Regime analysis (medium-term)
    'high_vol_regime',
    'vol_regime_change',
    'vol_regime_low',
    'vol_regime_medium',
    'vol_regime_high',
    'high_vol_regime_30d',
    'regime_duration_30d',
    'structural_break_proxy_30d',
    'downtrend_regime_7d',
    'uptrend_regime_7d',
    'downtrend_regime_30d',
    'uptrend_regime_30d',
    
    # Interaction features
    'volume_vol_interaction',
    'volume_vol_ratio',
    'rsi_trend_interaction',
    'rsi_downtrend_stress',
    'drawdown_volume_interaction',
    'momentum_alignment',
    'momentum_divergence_count'
]

# =============================================================================
# 60-DAY MODEL FEATURES (Long-term focus)
# =============================================================================

features_60d = [
    # Core price data
    'btc_volume',
    'btc_return',
    'btc_log_return', 
    'btc_cumulative_return',
    'btc_drawdown',
    'btc_frac_diff',
    
    # Full volatility spectrum
    'rolling_vol_7d',
    'rolling_vol_30d',
    'rolling_vol_60d',
    'vol_change_7d',
    'vol_change_30d', 
    'vol_change_60d',
    'vol_percentile_7d',
    'vol_percentile_30d',
    'vol_percentile_60d',
    'vol_term_structure_7_30',
    'vol_term_structure_7_60',
    'vol_term_structure_30_60',
    'volvol_7d',
    'volvol_30d',
    'volvol_60d',
    'vol_persistence_7d',
    'vol_persistence_30d',
    'vol_persistence_60d',
    'vol_momentum_7d',
    'vol_momentum_30d',
    'vol_momentum_60d',
    
    # Full momentum spectrum
    'rsi_7',
    'rsi_14',
    'rsi_30',
    'rsi_60',
    'adx_7',
    'adx_14',
    'adx_30',
    'adx_60',
    'roc_7d',
    'roc_14d',
    'roc_30d',
    'roc_60d',
    
    # Technical patterns
    'macd',
    'macd_signal',
    'macd_histogram',
    'macd_cross_signal',
    'stoch_rsi',
    'williams_r_7d',
    'williams_r_14d',
    'williams_r_30d',
    'williams_r_60d',
    
    # Market structure (all timeframes)
    'support_break_7d',
    'resistance_break_7d',
    'support_break_30d',
    'resistance_break_30d',
    'support_break_60d',
    'resistance_break_60d',
    'price_above_ma_7d',
    'price_above_ma_30d',
    'price_above_ma_60d',
    'ma_slope_7d',
    'ma_slope_30d',
    'ma_slope_60d',
    'lower_lows_sequence_3d',
    'ma7_above_ma30',
    'ma30_above_ma60',
    
    # Trend analysis (all timeframes)
    'trend_7d',
    'trend_30d',
    'trend_60d',
    
    # Comprehensive volume analysis
    'volume_ma_7d',
    'volume_ma_14d',
    'volume_ma_30d',
    'volume_ma_60d',
    'volume_ratio_current_7d',
    'volume_ratio_current_14d',
    'volume_ratio_current_30d',
    'volume_ratio_current_60d',
    'volume_ratio_7_30',
    'volume_ratio_7_60',
    'volume_ratio_30_60',
    'volume_spike_1.5x',
    'volume_spike_2.0x',
    'volume_spike_3.0x',
    'volume_breakdown',
    'volume_thrust',
    'volume_trend_7d',
    'volume_trend_14d',
    'volume_trend_30d',
    'obv',
    'obv_ma_7d',
    'obv_ma_14d',
    'obv_ma_30d',
    'obv_slope_7d',
    'obv_slope_14d',
    'obv_slope_30d',
    'vpt',
    'vpt_ma_14d',
    'vpt_slope_14d',
    'ad_line',
    
    # Statistical measures (all windows)
    'return_zscore_7d',
    'return_zscore_14d',
    'return_zscore_30d',
    'return_zscore_60d',
    'return_percentile_30d',
    'price_percentile_30d',
    'volume_percentile_30d',
    'return_percentile_60d',
    'price_percentile_60d',
    'volume_percentile_60d',
    'return_percentile_252d',  # Annual context
    'price_percentile_252d',
    'volume_percentile_252d',
    'consecutive_negative',
    'consecutive_positive',
    'var_5pct_14d',
    'var_95pct_14d',
    'cvar_5pct_14d',
    'var_5pct_30d',
    'var_95pct_30d',
    'cvar_5pct_30d',
    'var_5pct_60d',
    'var_95pct_60d',
    'cvar_5pct_60d',
    'max_adverse_move_7d',
    'max_favorable_move_7d',
    'max_adverse_move_14d',
    'max_favorable_move_14d',
    'max_adverse_move_30d',
    'max_favorable_move_30d',
    'rolling_skew_7d',
    'rolling_skew_14d',
    'rolling_skew_21d',
    'rolling_skew_30d',
    'rolling_skew_60d',
    'rolling_kurt_14d',
    'rolling_kurt_21d',
    'rolling_kurt_30d',
    
    # Jump detection
    'jump_indicator_2sigma',
    'jump_indicator_3sigma',
    'jump_indicator_4sigma',
    
    # Microstructure (less important for long-term but still useful)
    'price_gap',
    'large_gap_2pct',  # Focus on larger gaps for long-term
    'large_gap_3pct',
    'large_gap_5pct',
    'round_number_1000',
    'price_acceleration',
    'acceleration_ma_14d',
    'acceleration_ma_30d',
    
    # Behavioral and divergence (all timeframes)
    'rsi_overbought_negative',
    'rsi_oversold_positive',
    'rsi_divergence_7d',
    'rsi_divergence_14d',
    'rsi_divergence_30d',
    'trend_exhaustion_7d',
    'trend_exhaustion_30d',
    'momentum_confluence',
    
    # Temporal features (include seasonal)
    'day_of_week',
    'is_weekend',
    'is_monday',
    'is_friday',
    'month',
    'is_january',
    'is_december',
    'quarter',
    'year',  # Include for long-term model
    'is_year_end',
    'is_year_start',
    
    # Comprehensive regime analysis
    'high_vol_regime',
    'vol_regime_change',
    'vol_regime_low',
    'vol_regime_medium',
    'vol_regime_high',
    'high_vol_regime_30d',
    'regime_duration_30d',
    'high_vol_regime_60d',
    'regime_duration_60d',
    'structural_break_proxy_30d',
    'structural_break_proxy_60d',
    'downtrend_regime_7d',
    'uptrend_regime_7d',
    'downtrend_regime_30d',
    'uptrend_regime_30d',
    'downtrend_regime_60d',
    'uptrend_regime_60d',
    
    # All interaction features
    'volume_vol_interaction',
    'volume_vol_ratio',
    'rsi_trend_interaction',
    'rsi_downtrend_stress',
    'drawdown_volume_interaction',
    'momentum_alignment',
    'momentum_divergence_count'
]

# =============================================================================
# FEATURE LIST SUMMARY AND VALIDATION
# =============================================================================

def print_feature_summary():
    """Print summary of feature lists"""
    print("="*80)
    print("HORIZON-SPECIFIC FEATURE LISTS SUMMARY")
    print("="*80)
    
    print(f"\n7-Day Model Features: {len(features_7d)} features")
    print(f"30-Day Model Features: {len(features_30d)} features") 
    print(f"60-Day Model Features: {len(features_60d)} features")
    
    # Find common features across all models
    common_features = set(features_7d) & set(features_30d) & set(features_60d)
    print(f"\nCommon features across all models: {len(common_features)}")
    
    # Find unique features for each model
    unique_7d = set(features_7d) - set(features_30d) - set(features_60d)
    unique_30d = set(features_30d) - set(features_7d) - set(features_60d)
    unique_60d = set(features_60d) - set(features_7d) - set(features_30d)
    
    print(f"Unique to 7-day model: {len(unique_7d)}")
    print(f"Unique to 30-day model: {len(unique_30d)}")
    print(f"Unique to 60-day model: {len(unique_60d)}")
    
    print("\n" + "="*80)
    print("KEY DESIGN PRINCIPLES:")
    print("="*80)
    print("1. 7-day: Emphasizes microstructure, short-term momentum, immediate volatility")
    print("2. 30-day: Balances short and medium-term indicators")
    print("3. 60-day: Includes all timeframes with emphasis on structural breaks and regimes")
    print("4. No temporal leakage: Features don't use future information relative to prediction horizon")
    print("5. Progressive complexity: Longer horizons get more comprehensive feature sets")

def validate_features(all_available_features):
    """Validate that all features in our lists exist in the dataset"""
    print("\n" + "="*80)
    print("FEATURE VALIDATION")
    print("="*80)
    
    available_set = set(all_available_features)
    
    for horizon, feature_list in [("7d", features_7d), ("30d", features_30d), ("60d", features_60d)]:
        missing_features = set(feature_list) - available_set
        if missing_features:
            print(f"\n⚠️  {horizon} model missing features:")
            for feature in sorted(missing_features):
                print(f"   - {feature}")
        else:
            print(f"✅ {horizon} model: All {len(feature_list)} features available")

# Create a dictionary for easy access
HORIZON_FEATURES = {
    7: features_7d,
    30: features_30d, 
    60: features_60d
}

# Print summary
print_feature_summary()



enhanced_master_df = enhanced_master_df.dropna()


def check_future_falls(returns: pd.Series, n: int, d: int, w: int, cumulative: bool = True) -> pd.Series:
    """
    Check if a fall of n*std occurs within the next d days after each day t.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns (daily returns)
    n : int
        Number of standard deviations for the fall threshold
    d : int
        Number of days to look forward
    w : int
        Window size for calculating rolling standard deviation
    cumulative : bool
        If True, check cumulative returns over any period up to d days.
        If False, check only single-day returns.
    
    Returns:
    --------
    pd.Series: Binary series where 1 indicates a fall of n*std occurred within next d days, 0 otherwise
    """
    
    # Calculate rolling standard deviation with window w
    rolling_std = returns.rolling(window=w, min_periods=1).std()
    
    # Initialize result series with zeros
    result = pd.Series(0, index=returns.index)
    
    # For each day, check if there's a fall of n*std in the next d days
    for i in range(len(returns) - d):
        # Current day's standard deviation
        current_std = rolling_std.iloc[i]
        
        # Skip if std is NaN or zero
        if pd.isna(current_std) or current_std == 0:
            continue
        
        # Calculate the threshold for a fall (negative return of n*std)
        threshold = -n * current_std
        
        # Check returns in the next d days
        future_returns = returns.iloc[i+1:i+d+1]
        
        if cumulative:
            # Check cumulative returns for all possible periods
            for j in range(1, len(future_returns) + 1):
                # Calculate cumulative return from day i to day i+j
                cum_return = (1 + future_returns.iloc[:j]).prod() - 1
                
                # Check if cumulative return is below threshold
                if cum_return <= threshold:
                    result.iloc[i] = 1
                    break
        else:
            # Check if any single future return is below the threshold
            if (future_returns <= threshold).any():
                result.iloc[i] = 1
    
    return result


target = check_future_falls(enhanced_master_df.btc_return,6,60,60)

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, precision_score,recall_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
# =============================================================================
# MODIFIED ORIGINAL CELL - Added Precision Outputs
# =============================================================================
warnings.filterwarnings('ignore', category=UserWarning, message='.*Precision is ill-defined.*')

def create_balanced_data(X, y, strategy='moderate'):
    """Create balanced training data"""
    
    if strategy == 'aggressive':
        # Increase minority class to 30% of majority
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
    elif strategy == 'moderate':
        # Increase minority class to 20% of majority  
        smote = SMOTE(sampling_strategy=0.3, random_state=42)
    else:  # conservative
        # Increase minority class to 15% of majority
        smote = SMOTE(sampling_strategy=0.15, random_state=42)
    
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print(f"Original distribution: {np.bincount(y)}")
    print(f"Balanced distribution: {np.bincount(y_balanced)}")
    
    return X_balanced, y_balanced

def calculate_coefficient_significance(model, X, y):
    """Calculate p-values and confidence intervals for logistic regression coefficients"""
    from scipy import stats
    import scipy.linalg as la
    
    # Get predictions and residuals
    predictions = model.predict_proba(X)[:, 1]
    
    # Calculate the Hessian matrix (second derivative of log-likelihood)
    # For logistic regression: H = X^T * W * X where W is diagonal matrix of p(1-p)
    W = predictions * (1 - predictions)
    W_sqrt = np.sqrt(W)
    
    # Weighted design matrix
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    X_weighted = X_with_intercept * W_sqrt[:, np.newaxis]
    
    # Calculate covariance matrix
    try:
        # Use QR decomposition for numerical stability
        Q, R = np.linalg.qr(X_weighted)
        cov_matrix = la.inv(R.T @ R)
        
        # Standard errors are square root of diagonal elements
        std_errors = np.sqrt(np.diag(cov_matrix))
        
        # Get all coefficients (intercept + features)
        all_coeffs = np.concatenate([model.intercept_, model.coef_[0]])
        
        # Calculate z-scores and p-values
        z_scores = all_coeffs / std_errors
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        
        # 95% confidence intervals
        ci_lower = all_coeffs - 1.96 * std_errors
        ci_upper = all_coeffs + 1.96 * std_errors
        
        return std_errors, p_values, ci_lower, ci_upper, z_scores
        
    except (np.linalg.LinAlgError, la.LinAlgError):
        # If matrix is singular, return None values
        n_params = len(model.coef_[0]) + 1
        return (np.full(n_params, np.nan), np.full(n_params, np.nan), 
                np.full(n_params, np.nan), np.full(n_params, np.nan), 
                np.full(n_params, np.nan))

def generate_logistic_equation(model, feature_names, X_train, y_train):
    """Generate the logistic regression equation with coefficients, significance, and feature names"""
    
    intercept = model.intercept_[0]
    coefficients = model.coef_[0]
    
    # Calculate significance statistics
    std_errors, p_values, ci_lower, ci_upper, z_scores = calculate_coefficient_significance(model, X_train, y_train)
    
    print("\n" + "="*100)
    print("LOGISTIC REGRESSION FINAL EQUATION WITH SIGNIFICANCE")
    print("="*100)
    
    print(f"\nIntercept (β₀): {intercept:.6f}")
    if std_errors is not None and not np.isnan(std_errors[0]):
        significance = "***" if p_values[0] < 0.001 else "**" if p_values[0] < 0.01 else "*" if p_values[0] < 0.05 else ""
        print(f"  Standard Error: {std_errors[0]:.6f}, p-value: {p_values[0]:.6f}{significance}")
        print(f"  95% CI: [{ci_lower[0]:.6f}, {ci_upper[0]:.6f}]")
    
    print("\nCoefficients:")
    print("-" * 90)
    print(f"{'Coef':<4} {'Feature Name':<30} {'Coefficient':<12} {'Std Error':<10} {'p-value':<10} {'Significance':<4} {'95% CI':<25}")
    print("-" * 90)
    
    # Create equation string
    equation_parts = [f"{intercept:.6f}"]
    
    for i, (coef, feature_name) in enumerate(zip(coefficients, feature_names)):
        # Determine significance stars
        if std_errors is not None and not np.isnan(p_values[i+1]):
            p_val = p_values[i+1]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            std_err = std_errors[i+1]
            ci_range = f"[{ci_lower[i+1]:.4f}, {ci_upper[i+1]:.4f}]"
        else:
            p_val = np.nan
            significance = ""
            std_err = np.nan
            ci_range = "N/A"
        
        print(f"β{i+1:2d}  {feature_name:<30} {coef:>10.6f}  {std_err:>8.6f}  {p_val:>8.6f}  {significance:<4}  {ci_range:<25}")
        
        if coef >= 0:
            equation_parts.append(f" + {coef:.6f} * {feature_name}")
        else:
            equation_parts.append(f" - {abs(coef):.6f} * {feature_name}")
    
    print("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05")
    
    print("\n" + "="*100)
    print("FINAL LOGISTIC REGRESSION EQUATION:")
    print("="*100)
    print("log-odds = β₀ + β₁*X₁ + β₂*X₂ + ... + βₙ*Xₙ")
    print("\nlog-odds = " + "".join(equation_parts))
    
    print("\nPredicted Probability = 1 / (1 + exp(-(log-odds)))")
    print("="*100)
    
    # Show significant features only
    if std_errors is not None and not np.isnan(p_values).all():
        significant_features = []
        for i, (coef, feature_name, p_val) in enumerate(zip(coefficients, feature_names, p_values[1:])):
            if p_val < 0.05:
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                significant_features.append((abs(coef), feature_name, coef, p_val, significance))
        
        if significant_features:
            significant_features.sort(reverse=True)
            print(f"\nSIGNIFICANT FEATURES (p < 0.05, ranked by |coefficient|):")
            print("-" * 70)
            for rank, (abs_coef, feature_name, coef, p_val, sig) in enumerate(significant_features, 1):
                print(f"{rank:2d}. {feature_name:<30}: {coef:>10.6f} (p={p_val:.6f}){sig}")
    
    # Also show top 10 most important features by absolute coefficient value
    abs_coefs = np.abs(coefficients)
    top_indices = np.argsort(abs_coefs)[::-1][:10]
    
    print(f"\nTOP 10 FEATURES BY |COEFFICIENT| (regardless of significance):")
    print("-" * 70)
    for rank, idx in enumerate(top_indices, 1):
        p_val = p_values[idx+1] if std_errors is not None and not np.isnan(p_values[idx+1]) else np.nan
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "" if not np.isnan(p_val) else ""
        print(f"{rank:2d}. {feature_names[idx]:<30}: {coefficients[idx]:>10.6f} (p={p_val:.6f}){significance}")
    
    return equation_parts

print("\n" + "="*50)
print("LOGISTIC REGRESSION & RANDOM FOREST WITH PRECISION")
print("="*50)

# Multiple feature selection methods
print("Feature Selection Methods:")

# Method 1: F-classif (ANOVA F-test)
selector_f = SelectKBest(score_func=f_classif, k=15)
X_selected_f = selector_f.fit_transform(enhanced_master_df[features_7d], target)
selected_features_f = selector_f.get_support(indices=True)
print(f"F-classif selected features: {selected_features_f}")

# Method 2: Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=30)
X_selected_mi = selector_mi.fit_transform(enhanced_master_df[features_7d], target)
selected_features_mi = selector_mi.get_support(indices=True)
print(f"Mutual Info selected features: {selected_features_mi}")

# Choose the best performing method (using F-classif for consistency)
X_selected = X_selected_mi
selected_features = selected_features_mi
print(selected_features_mi.tolist())

# Get the original feature names for selected features
original_feature_names = [features_7d[i] for i in selected_features]
print(f"Selected feature names: {original_feature_names}")

# Time Series Split for validation
tscv = TimeSeriesSplit(n_splits=10)

# Split data chronologically (last 30% as test)
split_idx = int(0.7 * len(target))
X_train = X_selected[:split_idx]
X_test = X_selected[split_idx:]
y_train = target.iloc[:split_idx]
y_test = target.iloc[split_idx:]

X_train_orig, y_train_orig = X_train, y_train

X_train, y_train = create_balanced_data(X_train,y_train,strategy='aggressive')

# Grid Search parameters
lr_param_grid = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [10,20,30]
}

# Logistic Regression with GridSearch
print("\nLogistic Regression with GridSearch:")
lr_grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid=lr_param_grid,
    cv=tscv,
    scoring='precision_weighted',
    n_jobs=-1
)
lr_grid_search.fit(X_train, y_train)
lr_model = lr_grid_search.best_estimator_

print(f"  Best parameters: {lr_grid_search.best_params_}")
print(f"  Best CV precision: {lr_grid_search.best_score_:.4f}")

# In-sample predictions
y_train_pred_lr = lr_model.predict(X_train_orig)
lr_precision_train = precision_score(y_train_orig, y_train_pred_lr, average='weighted',zero_division=0)
cr_train_lr = classification_report(y_train_orig, y_train_pred_lr)

# Out-of-sample predictions
y_test_pred_lr = lr_model.predict(X_test)
lr_precision_test = precision_score(y_test, y_test_pred_lr, average='weighted',zero_division=0)
cr_ter_lr = classification_report(y_test, y_test_pred_lr)

print(f"  In-sample Precision: {lr_precision_train:.4f}")
print(f"  Out-of-sample Precision: {lr_precision_test:.4f}")

# Feature importance (coefficients)
lr_feature_importance = np.abs(lr_model.coef_[0])
print(f"  Feature Importance (top 5):")
top_indices = np.argsort(lr_feature_importance)[::-1]
for i, idx in enumerate(top_indices):
    original_idx = selected_features[idx]
    print(f"    Feature {original_idx}: {lr_feature_importance[idx]:.4f}")

# Generate the final logistic regression equation with significance
equation_parts = generate_logistic_equation(lr_model, original_feature_names, X_train, y_train)

# Random Forest with GridSearch
print("\nRandom Forest with GridSearch:")
rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=tscv,
    scoring='precision_weighted',
    n_jobs=-1
)
rf_grid_search.fit(X_train, y_train)
rf_model = rf_grid_search.best_estimator_

print(f"  Best parameters: {rf_grid_search.best_params_}")
print(f"  Best CV precision: {rf_grid_search.best_score_:.4f}")

# In-sample predictions
y_train_pred_rf = rf_model.predict(X_train)
rf_precision_train = precision_score(y_train, y_train_pred_rf, average='weighted',zero_division=0)
cr_train_rf = classification_report(y_train, y_train_pred_rf)

# Out-of-sample predictions
y_test_pred_rf = rf_model.predict(X_test)
rf_precision_test = precision_score(y_test, y_test_pred_rf, average='weighted',zero_division=0)
cr_test_rf = classification_report(y_test, y_test_pred_rf)

print(f"  In-sample Precision: {rf_precision_train:.4f}")
print(f"  Out-of-sample Precision: {rf_precision_test:.4f}")

# Feature importance
rf_feature_importance = rf_model.feature_importances_
print(f"  Feature Importance (top 5):")
top_indices = np.argsort(rf_feature_importance)[::-1][:10]
for i, idx in enumerate(top_indices):
    original_idx = selected_features[idx]
    print(f"    Feature {original_idx}: {rf_feature_importance[idx]:.4f}")

# Optional: Save the equation to a file
print("\n" + "="*80)
print("EQUATION SUMMARY")
print("="*80)
print("The logistic regression equation has been generated above.")
print("You can use this equation to make predictions on new data.")
print("Remember: Probability = 1 / (1 + exp(-(log-odds)))")
print("="*80)

# =============================================================================
# SUMMARY COMPARISON
# =============================================================================
print("\n" + "="*50)
print("PRECISION COMPARISON SUMMARY")
print("="*50)

models_summary = {
    'Logistic Regression': (lr_precision_train, lr_precision_test),
    'Random Forest': (rf_precision_train, rf_precision_test)
}

for model_name, (train_prec, test_prec) in models_summary.items():
    overfitting = train_prec - test_prec
    print(f"{model_name}:")
    print(f"  In-sample: {train_prec:.4f}")
    print(f"  Out-of-sample: {test_prec:.4f}")
    print(f"  Overfitting Gap: {overfitting:.4f}")

models_summary_cr = {
    'Logistic Regression': (cr_train_lr, cr_ter_lr),
    'Random Forest': (cr_train_rf, cr_test_rf)
}

for model_name, (cr_train, cr_test) in models_summary_cr.items():
    print("\n" + "="*50)
    print(f"{model_name}:")
    print(cr_train,'\test\n',cr_test)
    print("\n" + "="*50)


full_ensemble_majority = np.concatenate([y_train_pred_lr, y_test_pred_lr])

prediction_df = pd.DataFrame(index=enhanced_master_df.index)
prediction_df['6_60_60'] = full_ensemble_majority

counter = 0
result_exit = []
for i in range(len(prediction_df['6_60_60'])):
    if prediction_df['6_60_60'].iloc[i] == 1 and counter == 0:
        counter = 30
        result_exit.append(1)
        continue
    if counter != 0:
        counter -= 1
        result_exit.append(0)
    elif counter == 0:
        result_exit.append(1)
result_exit = pd.Series(result_exit,index=prediction_df.index)  

binary_exits_df = pd.DataFrame(index=enhanced_master_df.index)

binary_exits_df['6_60_60'] = result_exit

uptrend = detect_uptrend(enhanced_master_df['btc_close'])

final_binary_exit = []
for i in range(len(binary_exits_df)):
    final_binary_exit.append(np.dot(binary_exits_df[['5_30_30', '2_7_7','4_15_15']].iloc[i].values,[0.275,0.45,0.275]))
final_binary_exit = pd.Series(final_binary_exit,index=binary_exits_df.index)
final_binary_exit*= uptrend.shift(1).iloc[-713:]


final_return = master_df.btc_log_return.iloc[-713:-1] * final_binary_exit.iloc[-713:] 

cum_strat = (final_return + 1).cumprod().dropna()

holder_series = master_df.btc_log_return.loc[cum_strat.index[0]:]

cum_bench = (holder_series + 1).cumprod()

plt.figure(figsize=(12,8))

plt.plot(cum_strat.index,cum_strat.values,'orange')

plt.plot(cum_bench.index,cum_bench.values,'red')
