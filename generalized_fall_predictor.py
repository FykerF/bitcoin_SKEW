import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Scientific computing
from scipy import stats
import scipy.linalg as la

# Machine learning
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, classification_report
from imblearn.over_sampling import SMOTE

# Local imports
from fracdiff import frac_diff_rolling_threshold
from uptrend_detector import detect_uptrend

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


@dataclass
class TargetConfig:
    """Configuration for target generation"""
    n_sigma: int  # Number of standard deviations for fall threshold
    horizon_days: int  # Days to look forward for fall
    vol_window: int  # Window for volatility calculation
    name: str  # Target name (e.g., '5_30_30')
    cooldown_days: int = 30  # Cooldown period after signal (default 30 days)


@dataclass
class ModelConfig:
    """Configuration for model training"""
    n_features: int = 30
    test_split: float = 0.3
    smote_strategy: str = 'aggressive'  # 'aggressive', 'moderate', 'conservative'
    cv_folds: int = 10


class BitcoinFallPredictor:
    """
    Generalized Bitcoin fall prediction system that can work with any ticker
    and generate multiple logistic regression models for different target configurations.
    """
    
    def __init__(self, ticker: str = 'BTC-USD', start_date: str = '2018-01-01', random_state: int = 42):
        self.ticker = ticker
        self.start_date = start_date
        self.random_state = random_state
        self.data = None
        self.features_df = None
        self.targets = {}
        self.models = {}
        self.predictions = {}
        self.global_selected_features = None  # Store selected features globally
        self.global_selector = None  # Store selector globally
        
        # Set random seeds for reproducibility - do this once at initialization
        np.random.seed(self.random_state)
        
        # Feature lists for different horizons
        self.feature_lists = self._define_feature_lists()
    
    def collect_data(self) -> pd.DataFrame:
        """Download and prepare basic price data"""
        print(f"Downloading {self.ticker} data from {self.start_date}...")
        
        data = yf.download(self.ticker, start=self.start_date)
        
        # Fix column names if multi-level
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Create master DataFrame
        master_df = pd.DataFrame(index=data.index)
        
        # Add basic price data
        master_df['close'] = data['Close']
        master_df['high'] = data['High']
        master_df['low'] = data['Low']
        master_df['open'] = data['Open']
        master_df['volume'] = np.log(data['Volume'])
        
        # Calculate returns
        master_df['return'] = data['Close'].pct_change()
        master_df['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        self.data = master_df
        print(f"Data collected: {master_df.shape[0]} rows")
        return master_df
    
    def engineer_features(self) -> pd.DataFrame:
        """Create comprehensive feature set"""
        print("Starting feature engineering...")
        
        if self.data is None:
            raise ValueError("Data not collected. Run collect_data() first.")
        
        df = self.data.copy()
        
        # Basic price features
        df = self._add_basic_features(df)
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Market structure features
        df = self._add_market_structure_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Momentum features
        df = self._add_momentum_features(df)
        
        # Regime features
        df = self._add_regime_features(df)
        
        # Temporal features
        df = self._add_temporal_features(df)
        
        # Microstructure features
        df = self._add_microstructure_features(df)
        
        # Interaction features
        df = self._add_interaction_features(df)
        
        # Fractional differencing
        #df = self._add_fractional_diff(df) # This is a greate feature, but the problem with it is thaat 
        
        # Drop NaN values
        df = df.dropna()
        
        self.features_df = df
        print(f"Feature engineering complete: {df.shape[1]} features, {df.shape[0]} rows")
        return df
    
    def generate_targets(self, target_configs: List[TargetConfig]) -> Dict[str, pd.Series]:
        """Generate multiple target variables based on configurations"""
        print("Generating target variables...")
        
        #if self.features_df is None:
            #raise ValueError("Features not engineered. Run engineer_features() first.")
        
        returns = self.features_df['return']
        
        for config in target_configs:
            print(f"  Creating target {config.name} (σ={config.n_sigma}, horizon={config.horizon_days}, vol_window={config.vol_window})")
            
            target = self._check_future_falls(
                returns=returns,
                n=config.n_sigma,
                d=config.horizon_days,
                w=config.vol_window
            )
            
            self.targets[config.name] = target
            print(f"    Target {config.name}: {target.sum()} positive cases ({target.mean():.3%})")
        
        return self.targets
    
    def _perform_feature_selection_once(self, target_configs: List[TargetConfig], n_features: int = 30):
        """Perform feature selection ONCE for all models using the first target - like rouf_baseline"""
        print("Performing feature selection once for all models...")
        
        # Use the first target for feature selection (like rouf_baseline does)
        first_config = target_configs[0]
        first_target = first_config.name
        
        # Get feature list for the first target's horizon
        feature_list = self._select_feature_list(first_config.horizon_days)
        available_features = [f for f in feature_list if f in self.features_df.columns]
        
        if len(available_features) == 0:
            raise ValueError(f"No features available")
        
        # Perform feature selection ONCE
        X = self.features_df[available_features]
        y = self.targets[first_target]
        
        selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(available_features)))
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_feature_names = [available_features[i] for i in selected_indices]
        
        # Store the SAME features for ALL models (like rouf_baseline)
        self.global_selected_features = selected_feature_names
        self.global_selector = selector
        
        print(f"  Selected {len(selected_feature_names)} features from {len(available_features)} available")
        print(f"  Selected features: {selected_feature_names}")
        
        return selected_feature_names
    
    def train_models(self, target_configs: List[TargetConfig], model_config: ModelConfig = ModelConfig(), 
                     smote_threshold: float = None) -> Dict[str, dict]:
        """Train logistic regression models for each target
        
        Args:
            target_configs: List of target configurations
            model_config: Model configuration settings
            smote_threshold: If set, SMOTE will only be applied to targets with positive class 
                           percentage below this threshold (e.g., 0.22 for 22%)
        """
        print("Training logistic regression models...")
        
        if not self.targets:
            raise ValueError("Targets not generated. Run generate_targets() first.")
        
        # Perform feature selection ONCE for all models
        self._perform_feature_selection_once(target_configs, model_config.n_features)
        
        results = {}
        
        for config in target_configs:
            print(f"\n{'='*50}")
            print(f"Training model for target: {config.name}")
            print(f"{'='*50}")
            
            # Use the global random seed set at initialization
            
            # Use the SAME selected features for ALL models
            selected_features = self.global_selected_features
            
            # Prepare data
            X = self.features_df[selected_features].values
            y = self.targets[config.name]
            feature_names = selected_features
            
            # Check if SMOTE should be applied based on threshold
            positive_ratio = y.mean()
            use_smote = True
            if smote_threshold is not None:
                use_smote = positive_ratio < smote_threshold
                print(f"  Positive class ratio: {positive_ratio:.3%}")
                print(f"  SMOTE threshold: {smote_threshold:.3%}")
                print(f"  SMOTE will be {'applied' if use_smote else 'skipped'}")
            
            # Train model
            model_results = self._train_single_model(
                X=X, y=y, 
                feature_names=feature_names,
                config=model_config,
                target_name=config.name,
                use_smote=use_smote
            )
            
            print(f"  Selected {len(feature_names)} features: {feature_names[:5]}..." if len(feature_names) > 5 else f"  Selected features: {feature_names}")
            
            results[config.name] = model_results
            self.models[config.name] = model_results['model']
            # Feature names are already stored by horizon in self.selected_features
        
        return results
    
    def generate_predictions(self, target_configs: List[TargetConfig], ensemble_weights: List[float] = [0.275, 0.45, 0.275], uptrend_dummy: bool = True) -> pd.Series:
        """Generate ensemble predictions and final binary exits"""
        print("Generating ensemble predictions...")
        
        if not self.models:
            raise ValueError("Models not trained. Run train_models() first.")
        
        # Generate individual predictions
        individual_predictions = {}
        
        for config in target_configs:
            if config.name not in self.models:
                raise ValueError(f"Model for {config.name} not found")
            
            model = self.models[config.name]
            
            # Use the same features that were selected globally
            selected_feature_names = self.global_selected_features
            X_selected = self.features_df[selected_feature_names]
            
            # Make predictions
            predictions = model.predict(X_selected)
            
            # Apply cooldown period (using config-specific cooldown)
            binary_exits = self._apply_cooldown(predictions, cooldown_days=config.cooldown_days)
            individual_predictions[config.name] = binary_exits
            print(f"  Applied {config.cooldown_days}-day cooldown for {config.name}")
        
        # Create ensemble
        prediction_df = pd.DataFrame(individual_predictions, index=self.features_df.index)
        
        # Weighted ensemble
        ensemble_scores = []
        for i in range(len(prediction_df)):
            target_names = [config.name for config in target_configs]
            values = prediction_df[target_names].iloc[i].values
            score = np.dot(values, ensemble_weights)
            ensemble_scores.append(score)
        
        ensemble_series = pd.Series(ensemble_scores, index=self.features_df.index)
        
        ens = ensemble_series.copy()

        # Apply uptrend filter
        if uptrend_dummy: 
            uptrend = detect_uptrend(self.features_df['close'])
            filtered_ensemble = ensemble_series * uptrend.shift(1)
        else: 
            filtered_ensemble = ens
        
        self.predictions['ensemble'] = filtered_ensemble
        return filtered_ensemble
    
    def evaluate_strategy(self, ensemble_predictions: pd.Series, lookback_days: int = 713) -> Dict[str, float]:
        """Evaluate the trading strategy performance"""
        print("Evaluating strategy performance...")
        
        # Get returns for the specified lookback period
        returns = self.features_df['log_return'].iloc[-lookback_days:-1]
        signals = ensemble_predictions.iloc[-lookback_days:]
        
        # Calculate strategy returns
        strategy_returns = returns * signals.iloc[:-1]  # Align indices
        
        # Calculate cumulative returns
        strategy_cumret = (strategy_returns + 1).cumprod().dropna()
        benchmark_returns = returns.loc[strategy_cumret.index]
        benchmark_cumret = (benchmark_returns + 1).cumprod()
        
        # Calculate metrics
        metrics = {
            'strategy_total_return': strategy_cumret.iloc[-1] - 1,
            'benchmark_total_return': benchmark_cumret.iloc[-1] - 1,
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
            'benchmark_sharpe': benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252),
            'strategy_volatility': strategy_returns.std() * np.sqrt(252),
            'benchmark_volatility': benchmark_returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(strategy_cumret),
            'win_rate': (strategy_returns > 0).mean(),
            'avg_trade_return': strategy_returns[strategy_returns != 0].mean(),
            'num_trades': (strategy_returns != 0).sum()
        }
        
        return metrics, strategy_cumret, benchmark_cumret
    
    def plot_results(self, strategy_cumret: pd.Series, benchmark_cumret: pd.Series):
        """Plot strategy vs benchmark performance"""
        plt.figure(figsize=(12, 8))
        plt.plot(strategy_cumret.index, strategy_cumret.values, 'orange', label='Strategy', linewidth=2)
        plt.plot(benchmark_cumret.index, benchmark_cumret.values, 'red', label='Buy & Hold', linewidth=2)
        plt.title(f'{self.ticker} Fall Prediction Strategy Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Helper methods
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        # Price features
        df['cumulative_return'] = (1 + df['return']).cumprod() - 1
        df['drawdown'] = (df['close'] / df['close'].cummax()) - 1
        
        # Moving averages
        for window in [7, 30, 60, 200]:
            df[f'ma_{window}d'] = df['close'].rolling(window).mean()
            
        # Volatility measures
        for window in [7, 30, 60]:
            df[f'vol_{window}d'] = df['return'].rolling(window).std() * np.sqrt(252)
            df[f'vol_change_{window}d'] = df[f'vol_{window}d'].pct_change(window)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # RSI
        for period in [7, 14, 30, 60]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # ADX (Average Directional Index)
        for period in [7, 14, 30, 60]:
            df[f'adx_{period}'] = self._calculate_adx(df['high'], df['low'], df['close'], period)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
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
        for horizon in [7, 14, 30, 60]:
            highest_high = df['close'].rolling(horizon).max()
            lowest_low = df['close'].rolling(horizon).min()
            df[f'williams_r_{horizon}d'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
            df[f'williams_r_{horizon}d'] = df[f'williams_r_{horizon}d'].fillna(-50)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive volume analysis features"""
        # Volume moving averages
        for window in [7, 14, 30, 60]:
            df[f'volume_ma_{window}d'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_current_{window}d'] = df['volume'] / df[f'volume_ma_{window}d']
        
        # Volume ratios between different periods
        df['volume_ratio_7_30'] = df['volume_ma_7d'] / df['volume_ma_30d']
        df['volume_ratio_7_60'] = df['volume_ma_7d'] / df['volume_ma_60d']
        df['volume_ratio_30_60'] = df['volume_ma_30d'] / df['volume_ma_60d']
        
        # Volume spikes
        for multiplier in [1.5, 2.0, 3.0]:
            df[f'volume_spike_{multiplier}x'] = (df['volume'] > df['volume_ma_30d'] * multiplier).astype(int)
        
        # On-Balance Volume (OBV)
        obv = (df['volume'] * np.sign(df['return'])).cumsum()
        df['obv'] = obv
        for horizon in [7, 14, 30]:
            df[f'obv_ma_{horizon}d'] = obv.rolling(horizon).mean()
            df[f'obv_slope_{horizon}d'] = df[f'obv_ma_{horizon}d'].diff(horizon)
        
        # Volume-Price Trend (VPT)
        vpt = (df['volume'] * df['return']).cumsum()
        df['vpt'] = vpt
        df['vpt_ma_14d'] = vpt.rolling(14).mean()
        df['vpt_slope_14d'] = df['vpt_ma_14d'].diff(14)
        
        # Accumulation/Distribution Line
        daily_range = df['close'].rolling(1).max() - df['close'].rolling(1).min()
        daily_range = daily_range.where(daily_range != 0, df['close'] * 0.001)
        
        clv = ((df['close'] - df['close'].rolling(1).min()) - 
               (df['close'].rolling(1).max() - df['close'])) / daily_range
        clv = clv.fillna(0)
        df['ad_line'] = (clv * df['volume']).cumsum()
        
        # Volume patterns
        df['volume_breakdown'] = (
            (df['volume'] > df['volume_ma_30d'] * 1.5) & 
            (df['return'] < -0.02)
        ).astype(int)
        
        df['volume_thrust'] = (
            (df['volume'] > df['volume_ma_30d'] * 1.5) & 
            (df['return'] > 0.02)
        ).astype(int)
        
        # Volume trend strength
        for horizon in [7, 14, 30]:
            df[f'volume_trend_{horizon}d'] = df['volume'].rolling(horizon).apply(
                lambda x: stats.pearsonr(range(len(x)), x)[0] if len(x) == horizon else 0
            )
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical anomaly and distribution features"""
        # Rolling statistics
        for window in [7, 14, 21, 30, 60]:
            df[f'rolling_skew_{window}d'] = df['return'].rolling(window).skew()
            if window >= 14:
                df[f'rolling_kurt_{window}d'] = df['return'].rolling(window).kurt()
            
            # Z-scores
            rolling_mean = df['return'].rolling(window).mean()
            rolling_std = df['return'].rolling(window).std()
            df[f'return_zscore_{window}d'] = (df['return'] - rolling_mean) / rolling_std
            df[f'return_zscore_{window}d'] = df[f'return_zscore_{window}d'].fillna(0)
        
        # Percentile rankings
        for window in [30, 60, 252]:
            df[f'return_percentile_{window}d'] = df['return'].rolling(window).rank(pct=True)
            df[f'price_percentile_{window}d'] = df['close'].rolling(window).rank(pct=True)
            if 'volume' in df.columns:
                df[f'volume_percentile_{window}d'] = df['volume'].rolling(window).rank(pct=True)
        
        # Consecutive patterns
        negative_returns = (df['return'] < 0).astype(int)
        positive_returns = (df['return'] > 0).astype(int)
        
        # Consecutive negative returns
        neg_groups = (negative_returns != negative_returns.shift()).cumsum()
        df['consecutive_negative'] = negative_returns.groupby(neg_groups).cumsum() * negative_returns
        
        # Consecutive positive returns
        pos_groups = (positive_returns != positive_returns.shift()).cumsum()
        df['consecutive_positive'] = positive_returns.groupby(pos_groups).cumsum() * positive_returns
        
        # Tail risk measures
        for horizon in [14, 30, 60]:
            df[f'var_5pct_{horizon}d'] = df['return'].rolling(horizon).quantile(0.05)
            df[f'var_95pct_{horizon}d'] = df['return'].rolling(horizon).quantile(0.95)
            
            # Conditional VaR (Expected Shortfall)
            df[f'cvar_5pct_{horizon}d'] = df['return'].rolling(horizon).apply(
                lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else x.min()
            )
        
        # Maximum adverse excursion
        for horizon in [7, 14, 30]:
            df[f'max_adverse_move_{horizon}d'] = df['return'].rolling(horizon).min()
            df[f'max_favorable_move_{horizon}d'] = df['return'].rolling(horizon).max()
        
        return df
    
    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure and regime features for multiple horizons"""
        for horizon in [7, 30, 60]:
            # Support/Resistance levels
            df[f'support_break_{horizon}d'] = (
                df['close'] < df['close'].rolling(horizon).min().shift(1)
            ).astype(int)
            
            df[f'resistance_break_{horizon}d'] = (
                df['close'] > df['close'].rolling(horizon).max().shift(1)
            ).astype(int)
            
            # Price vs MA relationships  
            if f'ma_{horizon}d' in df.columns:
                df[f'price_above_ma_{horizon}d'] = (df['close'] > df[f'ma_{horizon}d']).astype(int)
                df[f'ma_slope_{horizon}d'] = df[f'ma_{horizon}d'].diff(horizon)
        
        # Cross-MA relationships
        if 'ma_7d' in df.columns and 'ma_30d' in df.columns:
            df['ma7_above_ma30'] = (df['ma_7d'] > df['ma_30d']).astype(int)
        if 'ma_30d' in df.columns and 'ma_60d' in df.columns:
            df['ma30_above_ma60'] = (df['ma_30d'] > df['ma_60d']).astype(int)
        
        # Sequential patterns
        df['lower_lows_sequence_3d'] = (
            (df['close'] < df['close'].shift(1)) & 
            (df['close'].shift(1) < df['close'].shift(2)) &
            (df['close'].shift(2) < df['close'].shift(3))
        ).astype(int)
        
        # Volatility regime changes
        if 'high_vol_regime' in df.columns:
            df['vol_regime_change'] = (df['high_vol_regime'] != df['high_vol_regime'].shift(1)).astype(int)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volatility structure features"""
        # Volatility persistence and clustering
        for horizon in [7, 14, 30, 60]:
            vol_col = f'vol_{horizon}d'
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
        
        # Volatility percentiles
        for horizon in [7, 30, 60]:
            vol_col = f'vol_{horizon}d'
            if vol_col in df.columns:
                df[f'vol_percentile_{horizon}d'] = df[vol_col].rolling(30).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
                )
        
        # Volatility term structure
        if 'vol_7d' in df.columns and 'vol_60d' in df.columns:
            df['vol_term_structure_7_60'] = df['vol_7d'] / df['vol_60d']
        if 'vol_7d' in df.columns and 'vol_30d' in df.columns:
            df['vol_term_structure_7_30'] = df['vol_7d'] / df['vol_30d']
        if 'vol_30d' in df.columns and 'vol_60d' in df.columns:
            df['vol_term_structure_30_60'] = df['vol_30d'] / df['vol_60d']
        
        # Jump detection for different thresholds
        for threshold in [2, 3, 4]:
            if 'vol_7d' in df.columns:
                df[f'jump_indicator_{threshold}sigma'] = (
                    np.abs(df['return']) > threshold * df['vol_7d']
                ).astype(int)
        
        # Volatility regime transitions
        if 'vol_30d' in df.columns:
            vol_30d = df['vol_30d']
            vol_percentiles = vol_30d.rolling(252).rank(pct=True)
            df['vol_regime_low'] = (vol_percentiles < 0.33).astype(int)
            df['vol_regime_medium'] = ((vol_percentiles >= 0.33) & (vol_percentiles < 0.67)).astype(int)
            df['vol_regime_high'] = (vol_percentiles >= 0.67).astype(int)
            
            # High volatility regime
            median_vol = vol_30d.median()
            df['high_vol_regime'] = (vol_30d > median_vol).astype(int)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and divergence features"""
        # Rate of Change (ROC) for multiple horizons
        for horizon in [7, 14, 30, 60]:
            df[f'roc_{horizon}d'] = ((df['close'] / df['close'].shift(horizon)) - 1) * 100
            df[f'roc_{horizon}d'] = df[f'roc_{horizon}d'].fillna(0)
        
        # Momentum divergence (price vs RSI)
        if 'rsi_14' in df.columns:
            for horizon in [7, 14, 30]:
                df[f'rsi_divergence_{horizon}d'] = (
                    (df['close'] > df['close'].shift(horizon)) & 
                    (df['rsi_14'] < df['rsi_14'].shift(horizon))
                ).astype(int)
        
        # Triple momentum confirmation
        df['momentum_confluence'] = (
            (df.get('roc_7d', 0) > 0) & 
            (df.get('roc_14d', 0) > 0) & 
            (df.get('roc_30d', 0) > 0)
        ).astype(int)
        
        # Price acceleration
        df['price_acceleration'] = df['return'].diff()
        for horizon in [7, 14, 30]:
            df[f'acceleration_ma_{horizon}d'] = df['price_acceleration'].rolling(horizon).mean()
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime detection features"""
        # Simple regime classification
        returns = df['return']
        
        for window in [30, 60]:
            if f'vol_{window}d' in df.columns:
                vol = df[f'vol_{window}d']
                vol_quantile = vol.rolling(252).quantile(0.67)
                df[f'high_vol_regime_{window}d'] = (vol > vol_quantile).astype(int)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""
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
        df['is_year_end'] = (df.index.month == 12).astype(int)
        df['is_year_start'] = (df.index.month == 1).astype(int)
        
        return df
    
    def _add_fractional_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fractional differencing"""
        try:
            frac_result, _ = frac_diff_rolling_threshold(df['close'], target_stationarity=0.95)
            if frac_result and 'series' in frac_result:
                df['frac_diff'] = frac_result['series']
        except Exception as e:
            print(f"Warning: Could not compute fractional differencing: {e}")
            df['frac_diff'] = 0
        
        return df
    
    def _calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = data.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
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
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure and behavioral features"""
        # Price gaps
        df['price_gap'] = df['close'] / df['close'].shift(1) - 1
        df['price_gap'] = df['price_gap'].fillna(0)
        
        for threshold in [0.01, 0.02, 0.03, 0.05]:
            df[f'large_gap_{int(threshold*100)}pct'] = (np.abs(df['price_gap']) > threshold).astype(int)
        
        # Price clustering around round numbers
        df['round_number_1000'] = np.minimum(
            df['close'] % 1000, 1000 - (df['close'] % 1000)
        ) / 1000
        
        df['round_number_100'] = np.minimum(
            df['close'] % 100, 100 - (df['close'] % 100)
        ) / 100
        
        # Momentum exhaustion signals
        if 'rsi_14' in df.columns:
            df['rsi_overbought_negative'] = ((df['rsi_14'] > 70) & (df['return'] < 0)).astype(int)
            df['rsi_oversold_positive'] = ((df['rsi_14'] < 30) & (df['return'] > 0)).astype(int)
        
        # Trend exhaustion
        for horizon in [7, 14, 30]:
            trend_col = f'trend_{horizon}d'
            if trend_col in df.columns:
                df[f'trend_exhaustion_{horizon}d'] = (
                    (df[trend_col] > 0.01) & (df['return'] < -0.02)
                ).astype(int)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions that are particularly relevant for fall prediction"""
        # Volume-Volatility interactions
        if 'volume' in df.columns and 'vol_7d' in df.columns:
            df['volume_vol_interaction'] = df['volume'] * df['vol_7d']
            df['volume_vol_ratio'] = df['volume'] / (df['vol_7d'] + 1e-8)
        
        # RSI-Trend interactions
        if 'rsi_14' in df.columns and 'trend_30d' in df.columns:
            df['rsi_trend_interaction'] = df['rsi_14'] * df['trend_30d']
            df['rsi_downtrend_stress'] = df['rsi_14'] * (df['trend_30d'] < 0).astype(int)
        
        # Drawdown-Volume interaction
        if 'drawdown' in df.columns and 'volume' in df.columns:
            df['drawdown_volume_interaction'] = df['drawdown'] * df['volume']
        
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
    
    def _check_future_falls(self, returns: pd.Series, n: int, d: int, w: int, cumulative: bool = True) -> pd.Series:
        """Check if a fall of n*std occurs within the next d days"""
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
    
    def _select_feature_list(self, horizon_days: int) -> List[str]:
        """Select appropriate feature list based on horizon"""
        if horizon_days <= 7:
            return self.feature_lists['7d']
        elif horizon_days <= 30:
            return self.feature_lists['30d']
        else:
            return self.feature_lists['60d']
    
    def _define_feature_lists(self) -> Dict[str, List[str]]:
        """Define comprehensive feature lists for different horizons matching rouf_baseline.py"""
        
        # Note: The following feature lists are directly from rouf_baseline.py
        # to ensure we have all the same sophisticated features
        
        return {
            '7d': [
                # Core price data
                'volume', 'return', 'log_return', 'drawdown', 'frac_diff',
                
                # Short-term volatility
                'vol_7d', 'vol_change_7d', 'vol_percentile_7d', 'volvol_7d',
                'vol_persistence_7d', 'vol_momentum_7d',
                
                # Short-term momentum indicators
                'rsi_7', 'rsi_14', 'adx_7', 'roc_7d', 'roc_14d',
                
                # Technical patterns
                'macd', 'macd_signal', 'macd_histogram', 'macd_cross_signal',
                'stoch_rsi', 'williams_r_7d', 'williams_r_14d',
                
                # Market structure
                'support_break_7d', 'resistance_break_7d', 'price_above_ma_7d',
                'ma_slope_7d', 'lower_lows_sequence_3d', 'ma7_above_ma30',
                
                # Volume patterns
                'volume_ma_7d', 'volume_ratio_current_7d', 'volume_ratio_7_30',
                'volume_spike_1.5x', 'volume_spike_2.0x', 'volume_spike_3.0x',
                'volume_breakdown', 'volume_thrust', 'volume_trend_7d',
                'obv_slope_7d', 'vpt_slope_14d',
                
                # Statistical measures
                'return_zscore_7d', 'return_zscore_14d', 'consecutive_negative',
                'consecutive_positive', 'max_adverse_move_7d', 'max_favorable_move_7d',
                'rolling_skew_7d',
                
                # Jump detection
                'jump_indicator_2sigma', 'jump_indicator_3sigma',
                'jump_indicator_4sigma', 'vol_regime_change',
                
                # Microstructure
                'price_gap', 'large_gap_1pct', 'large_gap_2pct', 'large_gap_3pct',
                'large_gap_5pct', 'round_number_1000', 'round_number_100',
                'price_acceleration', 'acceleration_ma_7d', 'acceleration_ma_14d',
                
                # Behavioral indicators
                'rsi_overbought_negative', 'rsi_oversold_positive',
                'rsi_divergence_7d', 'trend_exhaustion_7d',
                
                # Temporal features
                'day_of_week', 'is_weekend', 'is_monday', 'is_friday',
                
                # Regime indicators
                'vol_regime_low', 'vol_regime_medium', 'vol_regime_high',
                'downtrend_regime_7d', 'uptrend_regime_7d',
                
                # Interaction features
                'volume_vol_interaction', 'volume_vol_ratio', 'rsi_downtrend_stress',
                'momentum_alignment', 'momentum_divergence_count'
            ],
            
            '30d': [
                # Core price data
                'volume', 'return', 'log_return', 'cumulative_return',
                'drawdown', 'frac_diff',
                
                # Multi-timeframe volatility
                'vol_7d', 'vol_30d', 'vol_change_7d', 'vol_change_30d',
                'vol_percentile_7d', 'vol_percentile_30d', 'vol_term_structure_7_30',
                'volvol_7d', 'volvol_30d', 'vol_persistence_7d', 'vol_persistence_30d',
                'vol_momentum_7d', 'vol_momentum_30d',
                
                # Momentum indicators
                'rsi_7', 'rsi_14', 'rsi_30', 'adx_7', 'adx_14', 'adx_30',
                'roc_7d', 'roc_14d', 'roc_30d',
                
                # Technical patterns
                'macd', 'macd_signal', 'macd_histogram', 'macd_cross_signal',
                'stoch_rsi', 'williams_r_7d', 'williams_r_14d', 'williams_r_30d',
                
                # Market structure
                'support_break_7d', 'resistance_break_7d', 'support_break_30d',
                'resistance_break_30d', 'price_above_ma_7d', 'price_above_ma_30d',
                'ma_slope_7d', 'ma_slope_30d', 'lower_lows_sequence_3d',
                'ma7_above_ma30', 'ma30_above_ma60',
                
                # Trend analysis
                'trend_7d', 'trend_30d',
                
                # Volume analysis
                'volume_ma_7d', 'volume_ma_14d', 'volume_ma_30d',
                'volume_ratio_current_7d', 'volume_ratio_current_14d',
                'volume_ratio_current_30d', 'volume_ratio_7_30',
                'volume_spike_1.5x', 'volume_spike_2.0x', 'volume_spike_3.0x',
                'volume_breakdown', 'volume_thrust', 'volume_trend_7d',
                'volume_trend_14d', 'volume_trend_30d', 'obv', 'obv_ma_7d',
                'obv_ma_14d', 'obv_ma_30d', 'obv_slope_7d', 'obv_slope_14d',
                'obv_slope_30d', 'vpt', 'vpt_ma_14d', 'vpt_slope_14d', 'ad_line',
                
                # Statistical measures
                'return_zscore_7d', 'return_zscore_14d', 'return_zscore_30d',
                'return_percentile_30d', 'price_percentile_30d', 'volume_percentile_30d',
                'consecutive_negative', 'consecutive_positive', 'var_5pct_14d',
                'var_95pct_14d', 'cvar_5pct_14d', 'var_5pct_30d', 'var_95pct_30d',
                'cvar_5pct_30d', 'max_adverse_move_7d', 'max_favorable_move_7d',
                'max_adverse_move_14d', 'max_favorable_move_14d',
                'max_adverse_move_30d', 'max_favorable_move_30d',
                'rolling_skew_7d', 'rolling_skew_14d', 'rolling_skew_30d',
                'rolling_kurt_14d', 'rolling_kurt_30d',
                
                # Jump detection
                'jump_indicator_2sigma', 'jump_indicator_3sigma', 'jump_indicator_4sigma',
                
                # Microstructure
                'price_gap', 'large_gap_1pct', 'large_gap_2pct', 'large_gap_3pct',
                'large_gap_5pct', 'round_number_1000', 'round_number_100',
                'price_acceleration', 'acceleration_ma_7d', 'acceleration_ma_14d',
                'acceleration_ma_30d',
                
                # Behavioral and divergence
                'rsi_overbought_negative', 'rsi_oversold_positive',
                'rsi_divergence_7d', 'rsi_divergence_14d', 'rsi_divergence_30d',
                'trend_exhaustion_7d', 'trend_exhaustion_30d', 'momentum_confluence',
                
                # Temporal features
                'day_of_week', 'is_weekend', 'is_monday', 'is_friday',
                'month', 'is_january', 'is_december', 'quarter',
                
                # Regime analysis
                'high_vol_regime', 'vol_regime_change', 'vol_regime_low',
                'vol_regime_medium', 'vol_regime_high', 'high_vol_regime_30d',
                'regime_duration_30d', 'structural_break_proxy_30d',
                'downtrend_regime_7d', 'uptrend_regime_7d',
                'downtrend_regime_30d', 'uptrend_regime_30d',
                
                # Interaction features
                'volume_vol_interaction', 'volume_vol_ratio', 'rsi_trend_interaction',
                'rsi_downtrend_stress', 'drawdown_volume_interaction',
                'momentum_alignment', 'momentum_divergence_count'
            ],
            
            '60d': [
                # Core price data
                'volume', 'return', 'log_return', 'cumulative_return',
                'drawdown', 'frac_diff',
                
                # Full volatility spectrum
                'vol_7d', 'vol_30d', 'vol_60d', 'vol_change_7d', 'vol_change_30d',
                'vol_change_60d', 'vol_percentile_7d', 'vol_percentile_30d',
                'vol_percentile_60d', 'vol_term_structure_7_30',
                'vol_term_structure_7_60', 'vol_term_structure_30_60',
                'volvol_7d', 'volvol_30d', 'volvol_60d',
                'vol_persistence_7d', 'vol_persistence_30d', 'vol_persistence_60d',
                'vol_momentum_7d', 'vol_momentum_30d', 'vol_momentum_60d',
                
                # Full momentum spectrum
                'rsi_7', 'rsi_14', 'rsi_30', 'rsi_60', 'adx_7', 'adx_14',
                'adx_30', 'adx_60', 'roc_7d', 'roc_14d', 'roc_30d', 'roc_60d',
                
                # Technical patterns
                'macd', 'macd_signal', 'macd_histogram', 'macd_cross_signal',
                'stoch_rsi', 'williams_r_7d', 'williams_r_14d',
                'williams_r_30d', 'williams_r_60d',
                
                # Market structure (all timeframes)
                'support_break_7d', 'resistance_break_7d', 'support_break_30d',
                'resistance_break_30d', 'support_break_60d', 'resistance_break_60d',
                'price_above_ma_7d', 'price_above_ma_30d', 'price_above_ma_60d',
                'ma_slope_7d', 'ma_slope_30d', 'ma_slope_60d',
                'lower_lows_sequence_3d', 'ma7_above_ma30', 'ma30_above_ma60',
                
                # Trend analysis
                'trend_7d', 'trend_30d', 'trend_60d',
                
                # Comprehensive volume analysis
                'volume_ma_7d', 'volume_ma_14d', 'volume_ma_30d', 'volume_ma_60d',
                'volume_ratio_current_7d', 'volume_ratio_current_14d',
                'volume_ratio_current_30d', 'volume_ratio_current_60d',
                'volume_ratio_7_30', 'volume_ratio_7_60', 'volume_ratio_30_60',
                'volume_spike_1.5x', 'volume_spike_2.0x', 'volume_spike_3.0x',
                'volume_breakdown', 'volume_thrust', 'volume_trend_7d',
                'volume_trend_14d', 'volume_trend_30d', 'obv', 'obv_ma_7d',
                'obv_ma_14d', 'obv_ma_30d', 'obv_slope_7d', 'obv_slope_14d',
                'obv_slope_30d', 'vpt', 'vpt_ma_14d', 'vpt_slope_14d', 'ad_line',
                
                # Statistical measures (all windows)
                'return_zscore_7d', 'return_zscore_14d', 'return_zscore_30d',
                'return_zscore_60d', 'return_percentile_30d', 'price_percentile_30d',
                'volume_percentile_30d', 'return_percentile_60d',
                'price_percentile_60d', 'volume_percentile_60d',
                'return_percentile_252d', 'price_percentile_252d',
                'volume_percentile_252d', 'consecutive_negative',
                'consecutive_positive', 'var_5pct_14d', 'var_95pct_14d',
                'cvar_5pct_14d', 'var_5pct_30d', 'var_95pct_30d', 'cvar_5pct_30d',
                'var_5pct_60d', 'var_95pct_60d', 'cvar_5pct_60d',
                'max_adverse_move_7d', 'max_favorable_move_7d',
                'max_adverse_move_14d', 'max_favorable_move_14d',
                'max_adverse_move_30d', 'max_favorable_move_30d',
                'rolling_skew_7d', 'rolling_skew_14d', 'rolling_skew_21d',
                'rolling_skew_30d', 'rolling_skew_60d', 'rolling_kurt_14d',
                'rolling_kurt_21d', 'rolling_kurt_30d',
                
                # Jump detection
                'jump_indicator_2sigma', 'jump_indicator_3sigma', 'jump_indicator_4sigma',
                
                # Microstructure
                'price_gap', 'large_gap_2pct', 'large_gap_3pct', 'large_gap_5pct',
                'round_number_1000', 'price_acceleration', 'acceleration_ma_14d',
                'acceleration_ma_30d',
                
                # Behavioral and divergence
                'rsi_overbought_negative', 'rsi_oversold_positive',
                'rsi_divergence_7d', 'rsi_divergence_14d', 'rsi_divergence_30d',
                'trend_exhaustion_7d', 'trend_exhaustion_30d', 'momentum_confluence',
                
                # Temporal features (include seasonal)
                'day_of_week', 'is_weekend', 'is_monday', 'is_friday',
                'month', 'is_january', 'is_december', 'quarter',
                'year', 'is_year_end', 'is_year_start',
                
                # Comprehensive regime analysis
                'high_vol_regime', 'vol_regime_change', 'vol_regime_low',
                'vol_regime_medium', 'vol_regime_high', 'high_vol_regime_30d',
                'regime_duration_30d', 'high_vol_regime_60d', 'regime_duration_60d',
                'structural_break_proxy_30d', 'structural_break_proxy_60d',
                'downtrend_regime_7d', 'uptrend_regime_7d',
                'downtrend_regime_30d', 'uptrend_regime_30d',
                'downtrend_regime_60d', 'uptrend_regime_60d',
                
                # All interaction features
                'volume_vol_interaction', 'volume_vol_ratio', 'rsi_trend_interaction',
                'rsi_downtrend_stress', 'drawdown_volume_interaction',
                'momentum_alignment', 'momentum_divergence_count'
            ]
        }
    
    def _prepare_training_data(self, target_name: str, feature_list: List[str], n_features: int) -> Tuple[np.ndarray, pd.Series, List[str]]:
        """Prepare training data with feature selection"""
        # Get available features
        available_features = [f for f in feature_list if f in self.features_df.columns]
        
        if len(available_features) == 0:
            raise ValueError(f"No features available for target {target_name}")
        
        X = self.features_df[available_features]
        y = self.targets[target_name]
        
        # Feature selection with explicit random state
        np.random.seed(self.random_state)  # Ensure deterministic feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(available_features)))
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_feature_names = [available_features[i] for i in selected_indices]
        
        print(f"  Selected {len(selected_feature_names)} features from {len(available_features)} available")
        
        return X_selected, y, selected_feature_names
    
    def _train_single_model(self, X: np.ndarray, y: pd.Series, feature_names: List[str], 
                           config: ModelConfig, target_name: str, use_smote: bool = True) -> dict:
        """Train a single logistic regression model"""
        # Use the global random seed
        
        # Time series split
        split_idx = int((1 - config.test_split) * len(y))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        X_train_orig, y_train_orig = X_train.copy(), y_train.copy()
        
        # Balance training data only if use_smote is True
        if use_smote:
            X_train_balanced, y_train_balanced = self._create_balanced_data(
                X_train, y_train, config.smote_strategy
            )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            print(f"    SMOTE skipped - using original data distribution")
            print(f"    Training distribution: {np.bincount(y_train)}")
        
        # Grid search for hyperparameters
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        tscv = TimeSeriesSplit(n_splits=config.cv_folds)
        grid_search = GridSearchCV(
            LogisticRegression(random_state=self.random_state, max_iter=1000),
            param_grid=param_grid,
            cv=tscv,
            scoring='precision_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_balanced, y_train_balanced)
        model = grid_search.best_estimator_
        
        # Evaluate model
        y_train_pred = model.predict(X_train_orig)
        y_test_pred = model.predict(X_test)
        
        train_precision = precision_score(y_train_orig, y_train_pred, average='weighted', zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Train precision: {train_precision:.4f}")
        print(f"  Test precision: {test_precision:.4f}")
        print(f"  Overfitting gap: {train_precision - test_precision:.4f}")
        
        # Generate equation
        self._generate_logistic_equation(model, feature_names, X_train_balanced, y_train_balanced, target_name)
        
        return {
            'model': model,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'best_params': grid_search.best_params_,
            'feature_names': feature_names
        }
    
    def _create_balanced_data(self, X: np.ndarray, y: pd.Series, strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create balanced training data using SMOTE"""
        strategy_map = {
            'aggressive': 0.5,
            'moderate': 0.3,
            'conservative': 0.15
        }
        
        sampling_strategy = strategy_map.get(strategy, 0.3)
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"    Original distribution: {np.bincount(y)}")
        print(f"    Balanced distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def _apply_cooldown(self, predictions: np.ndarray, cooldown_days: int = 30) -> pd.Series:
        """Apply cooldown period to predictions
        
        When a signal (1) is generated, enforce a cooldown period where no new signals
        can be generated regardless of model predictions.
        """
        result = []
        counter = 0
        
        for pred in predictions:
            if pred == 1 and counter == 0:
                # Signal detected and not in cooldown - emit signal and start cooldown
                counter = cooldown_days
                result.append(1)
            elif counter > 0:
                # In cooldown period - no signal
                counter -= 1
                result.append(0)
            else:
                # No signal and not in cooldown - stay in market
                result.append(1)
        
        return pd.Series(result, index=self.features_df.index[:len(result)])
    
    def _calculate_max_drawdown(self, cumret_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = cumret_series.cummax()
        drawdown = (cumret_series - peak) / peak
        return drawdown.min()
    
    def _generate_logistic_equation(self, model, feature_names: List[str], X_train: np.ndarray, 
                                   y_train: np.ndarray, target_name: str):
        """Generate and print logistic regression equation"""
        print(f"\n{'='*60}")
        print(f"LOGISTIC REGRESSION EQUATION FOR {target_name}")
        print(f"{'='*60}")
        
        intercept = model.intercept_[0]
        coefficients = model.coef_[0]
        
        print(f"Intercept: {intercept:.6f}")
        print("\nTop 10 Features by |Coefficient|:")
        print("-" * 50)
        
        # Get top features by absolute coefficient
        abs_coefs = np.abs(coefficients)
        top_indices = np.argsort(abs_coefs)[::-1][:10]
        
        for i, idx in enumerate(top_indices, 1):
            print(f"{i:2d}. {feature_names[idx]:<25}: {coefficients[idx]:>10.6f}")
        
        print(f"\nEquation: P = 1 / (1 + exp(-({intercept:.4f} + Σ(βᵢ * Xᵢ))))")


def run_bitcoin_fall_prediction():
    """Main function to run the complete Bitcoin fall prediction pipeline"""
    print("="*80)
    print("GENERALIZED BITCOIN FALL PREDICTION SYSTEM")
    print("="*80)
    
    # Initialize predictor with fixed random state for reproducibility
    predictor = BitcoinFallPredictor(ticker='BTC-USD', start_date='2018-01-01', random_state=42)
    
    # Collect and engineer features
    predictor.collect_data()
    predictor.engineer_features()
    
    # Define target configurations with custom cooldown periods
    target_configs = [
        TargetConfig(n_sigma=5, horizon_days=30, vol_window=30, name='5_30_30', cooldown_days=20),
        TargetConfig(n_sigma=2, horizon_days=7, vol_window=7, name='2_7_7', cooldown_days=7),
        TargetConfig(n_sigma=4, horizon_days=15, vol_window=15, name='4_15_15', cooldown_days=12)
    ]
    
    # Generate targets and train models
    predictor.generate_targets(target_configs)
    model_config = ModelConfig(n_features=30, test_split=0.3, smote_strategy='aggressive')
    # Set threshold to 22% - SMOTE will only be applied to targets with less than 22% positive cases
    predictor.train_models(target_configs, model_config, smote_threshold=0.25)
    
    # Generate ensemble predictions
    ensemble_weights = [0.275, 0.45, 0.275]  # weights for 5_30_30, 2_7_7, 4_15_15
    ensemble_predictions = predictor.generate_predictions(target_configs, ensemble_weights)
    
    # Evaluate strategy
    metrics, strategy_cumret, benchmark_cumret = predictor.evaluate_strategy(ensemble_predictions)
    
    # Print results
    print(f"\n{'='*60}")
    print("STRATEGY PERFORMANCE METRICS")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        print(f"{metric:<25}: {value:.4f}")
    
    # Plot results
    predictor.plot_results(strategy_cumret, benchmark_cumret)
    
    return predictor, metrics


if __name__ == "__main__":
    predictor, metrics = run_bitcoin_fall_prediction()